# RAG 评估示例代码: BLEU + ROUGE + 支持文档覆盖率

# BLEU (Bilingual Evaluation Understudy, 双语评估替补)
    # 衡量生成文本与参考文本的 n-gram 重叠度
    # > 0.6 说明相似度较高
# ROUGE (Recall-Oriented Understudy for Gisting Evaluation)
    # 面向召回的摘要评估
    # > 0.7 说明覆盖面较好

from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
from rouge import Rouge

# 参考答案 vs 生成答案
ref = "您的汽车保险可赔偿医疗费用、车辆维修费，以及第三方损害赔偿。"
gen = "您的保单通常涵盖车祸后的医疗费用、车辆损失，以及对第三方的赔偿。"

# BLEU: 按字符级别计算 (中文没有空格分词, 所以用 list 拆成单字)
chencherry = SmoothingFunction()  # 平滑函数, 防止 n-gram 匹配为 0 时 BLEU 直接变 0
bleu = sentence_bleu([list(ref)], list(gen), smoothing_function=chencherry.method1)

# ROUGE: 直接传字符串
rouge = Rouge().get_scores(gen, ref)

print(f"BLEU: {bleu:.3f}, ROUGE-1 F1: {rouge[0]['rouge-1']['f']:.3f}")


# 支持文档覆盖率: 生成答案中的 token 有多少来自检索文档
generated = "您的保单通常涵盖车祸后的医疗费用、车辆损失，以及对第三方的赔偿。"
docs = [
    "根据保险条款，医疗费用和车辆损失在车祸理赔中可以获得赔偿。",
    "如果您对第三方造成损害，保险也会提供相应的赔付。"
]

# 按字符拆分, 过滤空白字符
tokens = lambda x: [c for c in x if c.strip()]
a = set(tokens(generated))        # 生成答案的字符集合
d = set(tokens("".join(docs)))    # 检索文档的字符集合
coverage = len(a & d) / len(a)    # 交集 / 生成答案总字符数

print(f"支持文档覆盖率: {coverage:.2f}")

# 判断标准:
    # BLEU > 0.6 + ROUGE > 0.7 → 生成答案与参考答案相似度较高
    # 覆盖率 > 0.7 → 大部分回答来自于检索文档, 可信度较高
    # 更高级的系统还会引入向量相似度计算语义重叠, 或者让另一个 LLM 检查是否引用了具体文档片段
