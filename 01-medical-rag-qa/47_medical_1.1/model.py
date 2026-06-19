# -*- coding: utf-8 -*-
"""模型层：嵌入模型（智谱 embedding-3）+ 大模型（本地 Qwen2.5-32B / DeepSeek API）。
密钥全部走环境变量；本地基座统一为 Qwen2.5-32B-Instruct，4-bit 量化推理（单卡 4090 约 20GB 可跑）。"""
import os
from dotenv import load_dotenv
from openai import OpenAI
from zai import ZhipuAiClient
from langchain.embeddings.base import Embeddings

from config import LLM_MODEL_PATH, ZHIPU_EMBEDDING_MODEL, get_logger

load_dotenv()
logger = get_logger(__name__)

# 嵌入模型：智谱 embedding-3（开发期）。合规场景可替换为本地 bge-m3。
client = ZhipuAiClient(api_key=os.getenv("ZHIPU_API_KEY"))


class ZhipuAIEmbeddings(Embeddings):
    def __init__(self, client):
        self.client = client

    def embed_documents(self, texts):
        embeddings = []
        for text in texts:
            response = self.client.embeddings.create(
                model=ZHIPU_EMBEDDING_MODEL,
                input=[text],
            )
            embeddings.append(response.data[0].embedding)
        return embeddings

    def embed_query(self, text):
        return self.embed_documents([text])[0]


# 本地大语言模型：Qwen2.5-32B-Instruct（4-bit NF4 量化加载）
def create_chat_model(model_path: str = LLM_MODEL_PATH, load_in_4bit: bool = True):
    import torch
    from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig

    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)

    quant_config = None
    if load_in_4bit:
        quant_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True,
            bnb_4bit_compute_dtype=torch.bfloat16,
        )

    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        quantization_config=quant_config,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        trust_remote_code=True,
    ).eval()
    logger.info("已加载本地大模型: %s (4bit=%s)", model_path, load_in_4bit)
    return model, tokenizer


def generate_answer(model, tokenizer, question, max_new_tokens: int = 1024):
    messages = [{"role": "user", "content": question}]
    text = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    model_inputs = tokenizer([text], return_tensors="pt").to(model.device)
    generated_ids = model.generate(**model_inputs, max_new_tokens=max_new_tokens)
    output_ids = generated_ids[0][len(model_inputs.input_ids[0]):].tolist()
    return tokenizer.decode(output_ids, skip_special_tokens=True).strip()


# DeepSeek API（开发期快速验证）
def create_deepseek_client():
    return OpenAI(
        api_key=os.getenv("DEEPSEEK_API_KEY"),
        base_url="https://api.deepseek.com/v1",
    )


def generate_deepseek_answer(client, question, model: str = "deepseek-chat"):
    response = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": "你是一个能力非常强大的助手."},
            {"role": "user", "content": question},
        ],
        stream=False,
    )
    return response.choices[0].message.content


if __name__ == "__main__":
    model, tokenizer = create_chat_model()
    print(generate_answer(model, tokenizer, "高血压患者饮食上要注意什么?"))
