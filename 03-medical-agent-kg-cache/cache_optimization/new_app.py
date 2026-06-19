import os
from dotenv import load_dotenv
load_dotenv()
# 代码运行前需要开启redis服务，这里直接写在app.py中，--daemonize yes可以保证子进程运行也生效。
import subprocess
subprocess.run("redis-server --daemonize yes --protected-mode no", shell=True)

# 等 Redis 真正起来再 import
import time
time.sleep(1)

import os
import uvicorn
from fastapi import FastAPI, Request, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
import json
import requests
import datetime

import sys
sys.path.append('../agent_kg')
from config import uri, auth
from neo4j import GraphDatabase
from zai import ZhipuAiClient
from langchain_milvus import Milvus, BM25BuiltInFunction

# 导入新的 redis 工具类
from new_redis import redis_manager
sys.path.append('../../01-medical-rag-qa/rag_pipeline')
from model import ZhipuAIEmbeddings, create_deepseek_client, generate_deepseek_answer

os.environ["TOKENIZERS_PARALLELISM"] = "false"
app = FastAPI()

deepseek_api_key = os.getenv("DEEPSEEK_API_KEY")

# 初始化模型和数据库连接
client_embedding = ZhipuAiClient(api_key=os.getenv("ZHIPU_API_KEY"))

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

embedding_model = ZhipuAIEmbeddings(client_embedding)
print("创建 Embedding 模型成功......")

URI = "./milvus_agent.db"
milvus_vectorstore = Milvus(
    embedding_function=embedding_model,
    builtin_function=BM25BuiltInFunction(),
    vector_field=["dense", "sparse"],
    index_params=[
        {"metric_type": "IP", "index_type": "IVF_FLAT"},
        {"metric_type": "BM25", "index_type": "SPARSE_INVERTED_INDEX"}
    ],
    connection_args={"uri": URI},
)
retriever = milvus_vectorstore.as_retriever()
print("创建 Milvus 连接成功......")

client_llm = create_deepseek_client()
print("创建 DeepSeek 成功......")

# 注意: Redis 连接已经在 new_redis 导入时自动初始化了


def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)


# 图灵君课堂小朱老师 • 独家讲义
# 核心封装: 将 RAG + LLM 的耗时逻辑封装为一个函数
def perform_rag_and_llm(query: str) -> str:
    """
    执行完整的 RAG 流程:
    1. Milvus 召回
    2. DeepSeek 推理
    """
    global milvus_vectorstore, retriever

    # 1. 向量数据库 Milvus 召回
    recall_milvus = milvus_vectorstore.similarity_search(
        query,
        k=10,
        ranker_type="rrf",
        ranker_params={"k": 100}
    )
    context = format_docs(recall_milvus) if recall_milvus else ""

    # 2. 构造 Prompt 并调用 LLM
    SYSTEM_PROMPT = """
    System: 你是一个非常得力的医学助手, 你可以通过从数据库中检索出的信息找到问题的答案.
    """

    USER_PROMPT = f"""
    User: 利用介于<context>和</context>之间的从数据库中检索出的信息来回答问题, 具体的问题介于<question>和</question>之间. 如果提供的信息为空, 则按照你的经验知识来给出尽可能严谨准确的回答, 不知道的时候坦诚的承认不了解, 不要编造不真实的信息.
    <context>
    {context}
    </context>

    <question>
    {query}
    </question>
    """

    # 调用 DeepSeek 生成回答
    response = generate_deepseek_answer(client_llm, SYSTEM_PROMPT + USER_PROMPT)
    return response


@app.post("/")
async def chatbot(request: Request):
    try:
        json_post_raw = await request.json()
        # 处理可能的双重编码问题
        if isinstance(json_post_raw, str):
            json_post_list = json.loads(json_post_raw)
        else:
            json_post_list = json_post_raw

        query = json_post_list.get('question')
        if not query:
            return {"status": 400, "error": "Question is required"}

        # 使用 redis_manager 接管流程
        # 1. 尝试从缓存读
        # 2. 如果没有, 自动加锁
        # 3. 执行 perform_rag_and_llm 回调函数
        # 4. 将结果写入缓存并返回
        # 5. 自动释放锁

        # 使用 lambda 将 query 参数闭包传给 compute 函数
        compute_callback = lambda: perform_rag_and_llm(query)
        response = redis_manager.get_or_compute(query, compute_callback)

        now = datetime.datetime.now()
        time_str = now.strftime("%Y-%m-%d %H:%M:%S")
        return {
            "response": response,
            "status": 200,
            "time": time_str
        }
    except Exception as e:
        print(f"Server Error: {e}")
        return {"status": 500, "error": str(e)}


if __name__ == '__main__':
    uvicorn.run(app, host='0.0.0.0', port=8103, workers=1)
