import os
from dotenv import load_dotenv
load_dotenv()
import os
import json
import re
import random
from openai import OpenAI                              # OpenAI 兼容客户端,DeepSeek 走同协议
from transformers import AutoModelForCausalLM, AutoTokenizer

random.seed(42)                                        # 固定随机种子,实验可复现

# ============================================================
# 1. DeepSeek API 客户端
# ============================================================
client = OpenAI(
    api_key=os.getenv("DEEPSEEK_API_KEY"),                                 # DeepSeek API Key
    base_url="https://api.deepseek.com"                # 不指向 OpenAI,而是 DeepSeek 的服务器
)

# ============================================================
# 2. 加载本地模型(baseline + finetune)
# ============================================================
baseline_model_path = "/root/autodl-tmp/40_rlhf_on_llama/Qwen3-4B"               # 未微调的原始 Qwen3-4B
finetune_model_path = "/root/autodl-tmp/40_rlhf_on_llama/LLaMA-Factory/output/llama3_lora_ppo/"   # 你 PPO 训练后的模型

# from_pretrained 会自动从目录里读 config/权重/tokenizer
base_model = AutoModelForCausalLM.from_pretrained(
    baseline_model_path,
    torch_dtype="auto",                                # 让 transformers 按 config 选 dtype(通常 bf16)
    device_map="auto"                                  # 自动决定模型放哪张/哪几张 GPU
)
base_tokenizer = AutoTokenizer.from_pretrained(baseline_model_path)
print('baseline device:', base_model.device)

finetune_model = AutoModelForCausalLM.from_pretrained(
    finetune_model_path,
    torch_dtype="auto",
    device_map="auto"
)
finetune_tokenizer = AutoTokenizer.from_pretrained(finetune_model_path)
print('finetune device:', finetune_model.device)


# ============================================================
# 3. 三个"问一题"函数,分别对应三个被评估的模型
# ============================================================

def run_one_question(question: str):
    """走 DeepSeek API 拿答案"""
    response = client.chat.completions.create(
        model="deepseek-chat",                         # DeepSeek 的对话模型
        messages=[
            # system 给模型设定身份,关键是末尾要求 'The answer is X' 格式,后面正则才能抽出来
            {"role": "system",
             "content": "You are a knowledge expert, you are supposed to answer the multi-choice question to derive your final answer as `The answer is ...`."},
            {"role": "user", "content": question},     # 题目
        ],
        stream=False                                   # 不流式,等完整返回
    )
    return response.choices[0].message.content         # 取第一个生成结果的文本


def run_one_question_qwen3_baseline(question: str):
    """走本地 baseline Qwen3"""
    # Qwen3 是对话模型,需要按 chat 模板组织 messages
    messages = [
        {"role": "system",
         "content": "You are a knowledge expert, you are supposed to answer the multi-choice question to derive your final answer as `The answer is ...`."},
        {"role": "user", "content": question}
    ]

    # apply_chat_template 把 messages 拼成 Qwen3 期望的字符串格式
    # 如: <|im_start|>system\n...<|im_end|>\n<|im_start|>user\n...<|im_end|>\n<|im_start|>assistant\n
    text = base_tokenizer.apply_chat_template(
        messages,
        tokenize=False,                                # 只拼字符串,不转 token(留给下一步)
        add_generation_prompt=True                     # 末尾加 assistant 起始 token,提示模型该生成了
    )

    # 把字符串转成 token id 张量,送到 GPU
    model_inputs = base_tokenizer([text], return_tensors="pt").to('cuda')

    # 让模型生成
    generated_ids = base_model.generate(
        **model_inputs,                                # 展开传入 input_ids 和 attention_mask
        max_new_tokens=512                             # 最多生成 512 个 token
    )

    # generate 返回的是 [输入 token + 生成 token] 的拼接,要去掉输入部分
    generated_ids = [
        output_ids[len(input_ids):]                    # 切掉前面输入的长度
        for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)
    ]

    # token id 解码回人类可读的文本; skip_special_tokens 跳过 <|im_end|> 之类
    response = base_tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]
    return response


def run_one_question_qwen3_finetune(question: str):
    """走微调后的 Qwen3,逻辑同 baseline,只是模型和 tokenizer 换了"""
    messages = [
        {"role": "system",
         "content": "You are a knowledge expert, you are supposed to answer the multi-choice question to derive your final answer as `The answer is ...`."},
        {"role": "user", "content": question}
    ]

    text = finetune_tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True
    )

    model_inputs = finetune_tokenizer([text], return_tensors="pt").to('cuda')

    generated_ids = finetune_model.generate(
        **model_inputs,
        max_new_tokens=512
    )

    generated_ids = [
        output_ids[len(input_ids):]
        for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)
    ]

    response = finetune_tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]
    return response


# ============================================================
# 4. 辅助函数
# ============================================================

def form_options(options: list):
    """把选项 list 格式化成 prompt 用的字符串
    输入: ['apple', 'banana', 'cherry']
    输出:
        Options are:
        (A): apple
        (B): banana
        (C): cherry
    MMLU-Pro 是 10 选 1,所以字母准备到 J
    """
    option_str = 'Options are:\n'
    opts = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J']
    for opt, o in zip(options, opts):                  # zip 同时遍历选项内容和字母
        option_str += f'({o}): {opt}' + '\n'
    return option_str


def get_prediction(output):
    """从模型回答里抽答案字母
    正则解读: r"answer is \(?([ABCDEFGHIJ])\)?"
      answer is        字面匹配
      \(?              一个可选的左括号
      ([ABCDEFGHIJ])   捕获组,匹配一个 A-J 字母
      \)?              一个可选的右括号
    所以 'answer is A'、'answer is (B)' 都能匹配
    """
    pattern = r"answer is \(?([ABCDEFGHIJ])\)?"
    match = re.search(pattern, output)                 # 在 output 全文中找第一个匹配
    if match:
        return match.group(1)                          # 取第一个捕获组(就是那个字母)
    else:
        # 模型没按格式回答,正则抓不到,只能瞎猜兜底,保证流程不卡
        print("extraction failed, do a random guess")
        return random.choice(['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J'])


# ============================================================
# 5. 评估流程封装
# 用一个函数跑一个模型,避免三段重复代码
# ============================================================

def evaluate(name, dataset, ask_fn, output_json, prefix_dict=None):
    """跑一个模型对整个数据集做评估,返回准确率
    参数:
        name:        模型名字(打印用)
        dataset:     题目列表
        ask_fn:      "问一题"的函数(三个模型各传不同的)
        output_json: 评估明细保存路径
    """
    print(f'\n----------------- Start {name} -------------------')
    success, fail = 0, 0                               # 答对/答错计数
    answers = []                                       # 存所有题目的答题明细(便于事后人工查)

    # enumerate(dataset, 1) 表示从 1 开始计数(默认是 0)
    for n, entry in enumerate(dataset, 1):
        prefix = prefix_dict.get(entry['category'], '') if prefix_dict else ''
        # 拼一道题的完整 prompt: 题干 + 选项
        query = prefix + 'Q: ' + entry['question'] + '\n' + form_options(entry['options']) + '\n'

        # 调模型
        answer = ask_fn(query)
        entry['solution'] = answer                     # 把模型答案塞回 entry,留个底
        answers.append(entry)

        # 抽预测字母,跟标准答案比
        prediction = get_prediction(answer)
        if entry["answer"] == prediction:
            success += 1
        else:
            fail += 1

        # 每 10 道题打一次进度
        if n % 10 == 0:
            print(f'[{name}] n={n}, success={success}, fail={fail}')

    # 准确率 = 答对 / 总数; 加个保护避免除零
    acc = success / (success + fail) if (success + fail) > 0 else 0.0

    # 把所有题目的明细保存为 JSON,事后可分析哪些题错了
    # ensure_ascii=False 保证中文不被转义成 \uXXXX
    with open(output_json, 'w', encoding='utf-8') as f:
        json.dump(answers, f, indent=2, ensure_ascii=False)

    print(f'{name} accuracy: {acc:.4f}')
    return acc


# ============================================================
# 6. 主流程
# ============================================================

if __name__ == "__main__":
    # 加载 MMLU-Pro 验证集
    # 列表推导式 + 过滤空值,只保留非空 entry
    with open('data/mmlu_pro_data_validation.json', 'r', encoding='utf-8') as f:
        dataset_val = [line for line in json.load(f) if line]

    print("Total number of data:", len(dataset_val))
    
    with open('data/mmlu_pro_data_test.json', 'r', encoding='utf-8') as f:
        dataset_test = [line for line in json.load(f) if line]

    print("Total number of data:", len(dataset_test))

    # MMLU-Pro 涵盖的 14 个学科
    categories = ['computer science', 'math', 'chemistry', 'engineering', 'law',
                  'biology', 'health', 'physics', 'business', 'philosophy',
                  'economics', 'other', 'psychology', 'history']

    # 为每个学科准备 5-shot 示例(从验证集里抽,拼成 few-shot prompt 前缀)
    # 当前主流程没用到 prompts,所以是 0-shot 评估
    # 想做 5-shot: 在下面 evaluate 之前,把 prompts[entry['category']] 拼到 query 前面
    # prompts = {c: '' for c in categories}
    # for d in dataset_val:
    #     # cot_content 是带思维链的标准解答,作为 few-shot 的"示范回答"
    #     prompts[d['category']] += 'Q: ' + d['question'] + form_options(d['options']) \
    #                               + '\n' + d['cot_content'] + '\n\n'

    from collections import defaultdict

    shot_count = defaultdict(int)
    prompts = {c: '' for c in categories}

    for d in dataset_test:
        if shot_count[d['category']] >= 5:
            continue
        prompts[d['category']] += 'Q: ' + d['question'] + form_options(d['options']) \
                                  + '\n' + d['cot_content'] + '\n\n'
        shot_count[d['category']] += 1

    # 三轮独立评估
    # acc_baseline = evaluate(
    #     'Baseline Qwen3',
    #     dataset_val,
    #     run_one_question_qwen3_baseline,               # 把函数本身作为参数传入(高阶函数)
    #     'outputs_baseline.json'
    # )

    # acc_deepseek = evaluate(
    #     'DeepSeek API',
    #     dataset_val,
    #     run_one_question,
    #     'outputs_deepseek.json'
    # )

    # acc_finetune = evaluate(
    #     'Finetune Qwen3',
    #     dataset_val,
    #     run_one_question_qwen3_finetune,
    #     'outputs_finetune.json'
    # )

    # # 加5-shot的评估：
    acc_baseline = evaluate(
        'Baseline Qwen3',
        dataset_val,
        run_one_question_qwen3_baseline,
        'outputs_baseline.json', 
        prefix_dict=prompts
    )

    acc_deepseek = evaluate(
        'DeepSeek API',
        dataset_val,
        run_one_question,
        'outputs_deepseek.json', 
        prefix_dict=prompts
    )

    acc_finetune = evaluate(
        'Finetune Qwen3',
        dataset_val,
        run_one_question_qwen3_finetune,
        'outputs_finetune.json', 
        prefix_dict=prompts
    )

    # 最终汇总,方便一眼看出微调有没有效
    print('\n========== Final Comparison ==========')
    print(f'Baseline Qwen3 : {acc_baseline:.4f}')
    print(f'DeepSeek API   : {acc_deepseek:.4f}')
    print(f'Finetune Qwen3 : {acc_finetune:.4f}')
    print(f'Improvement    : {(acc_finetune - acc_baseline) * 100:.2f}%')
