import os
from dotenv import load_dotenv
load_dotenv()
from transformers import AutoTokenizer, AutoModelForCausalLM
import os
from openai import OpenAI
from zai import ZhipuAiClient
from langchain.embeddings.base import Embeddings




# 模型1: 嵌入模型, 采用清华智谱最新的embedding-3
# 实例化智谱client对象
client = ZhipuAiClient(api_key=os.getenv("ZHIPU_API_KEY"))


class ZhipuAIEmbeddings(Embeddings):
    def __init__(self, client):
        self.client = client

    def embed_documents(self, texts):
        embeddings = []
        for text in texts:
            # 调用清华智谱最新版本的 embeddings 方法
            response = self.client.embeddings.create(
                model="embedding-3",
                input=[text],
            )
            embeddings.append(response.data[0].embedding)
        return embeddings

    def embed_query(self, text):
        # 查询文档
        return self.embed_documents([text])[0]


# 模型2: 大语言模型, 采用 Qwen3-Next-80B-A3B-Thinking
def create_chat_model():
    tokenizer = AutoTokenizer.from_pretrained("./Qwen3-Next-80B-A3B-Thinking",
                                              trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        "./Qwen3-Next-80B-A3B-Thinking",
        torch_dtype="auto",
        device_map="auto",
        trust_remote_code=True
    ).eval()
    return model, tokenizer


def generate_answer(model, tokenizer, question):
    messages = [{"role": "user", "content": question}]
    text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=True
    )
    model_inputs = tokenizer([text], return_tensors="pt").to(model.device)

    generated_ids = model.generate(**model_inputs, max_new_tokens=32768)
    output_ids = generated_ids[0][len(model_inputs.input_ids[0]):].tolist()

    try:
        # rindex finding 151668 (</think>)
        index = len(output_ids) - output_ids[::-1].index(151668)
    except ValueError:
        index = 0

    # thinking_content = tokenizer.decode(output_ids[:index],
    #                                     skip_special_tokens=True).strip("\n")
    content = tokenizer.decode(output_ids[index:],
                               skip_special_tokens=True).strip("\n")
    # print("thinking content:", thinking_content)
    # print("content:", content)
    return content.strip()


def create_deepseek_client():
    # 从环境变量获取 DeepSeek API Key
    deepseek_api_key = os.getenv("DEEPSEEK_API_KEY")
    # 初始化 DeepSeek 模型
    client = OpenAI(
        api_key=deepseek_api_key,                # 你的 DeepSeek API 密钥
        base_url="https://api.deepseek.com/v1",  # DeepSeek 的 API 端点
    )
    return client


def generate_deepseek_answer(client, question):
    response = client.chat.completions.create(
        model="deepseek-chat",
        messages=[
            {"role": "system", "content": "你是一个能力非常强大的助手."},
            {"role": "user", "content": question}
        ],
        stream=False
    )
    # print(response.choices[0].message.content)
    return response.choices[0].message.content


if __name__ == "__main__":
    # bge_m3 = create_embedding_model()
    # print(bge_m3)
    model, tokenizer = create_chat_model()
    output = generate_answer(model, tokenizer, "你好啊,千与千寻")
    print('-' * 50)
    print(output)