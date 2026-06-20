import os
import logging

from dotenv import load_dotenv
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
import torch
from openai import OpenAI
from zai import ZhipuAiClient
from langchain.embeddings.base import Embeddings

load_dotenv()
logger = logging.getLogger(__name__)

LLM_MODEL_PATH = os.getenv(
    "LLM_MODEL_PATH", "/root/autodl-tmp/models/Qwen/Qwen2.5-32B-Instruct"
)

# Embedding client (ZhipuAI embedding-3)
client = ZhipuAiClient(api_key=os.getenv("ZHIPU_API_KEY"))


class ZhipuAIEmbeddings(Embeddings):
    def __init__(self, client):
        self.client = client

    def embed_documents(self, texts):
        embeddings = []
        for text in texts:
            response = self.client.embeddings.create(model="embedding-3", input=[text])
            embeddings.append(response.data[0].embedding)
        return embeddings

    def embed_query(self, text):
        return self.embed_documents([text])[0]


def create_chat_model(model_path: str = LLM_MODEL_PATH):
    """Load Qwen2.5-32B-Instruct with 4-bit NF4 quantization (fits 4x4090)."""
    logger.info("Loading local LLM from %s", model_path)
    quant_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
        bnb_4bit_compute_dtype=torch.bfloat16,
    )
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        quantization_config=quant_config,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        trust_remote_code=True,
    ).eval()
    logger.info("Local LLM loaded")
    return model, tokenizer


def generate_answer(model, tokenizer, question, max_new_tokens=2048):
    messages = [{"role": "user", "content": question}]
    text = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    model_inputs = tokenizer([text], return_tensors="pt").to(model.device)
    generated_ids = model.generate(**model_inputs, max_new_tokens=max_new_tokens)
    output_ids = generated_ids[0][len(model_inputs.input_ids[0]):].tolist()
    return tokenizer.decode(output_ids, skip_special_tokens=True).strip()


def create_deepseek_client():
    client = OpenAI(
        api_key=os.getenv("DEEPSEEK_API_KEY"),
        base_url="https://api.deepseek.com/v1",
    )
    return client


def generate_deepseek_answer(client, question):
    response = client.chat.completions.create(
        model="deepseek-chat",
        messages=[
            {"role": "system", "content": "你是一个能力非常强大的助手."},
            {"role": "user", "content": question},
        ],
        stream=False,
    )
    return response.choices[0].message.content


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    model, tokenizer = create_chat_model()
    output = generate_answer(model, tokenizer, "高血压患者饮食上要注意什么?")
    logger.info("Sample answer: %s", output)
