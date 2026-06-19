import os
from dotenv import load_dotenv
load_dotenv()
import os
import uvicorn
from fastapi import FastAPI, Request, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
import json
import requests
import datetime

from config import uri, auth
from neo4j import GraphDatabase
from zai import ZhipuAiClient
from langchain_milvus import Milvus, BM25BuiltInFunction
from vectors import get_redis_client, cache_set, cache_get
import sys
sys.path.append('../../01-medical-rag-qa/rag_pipeline')
from model import ZhipuAIEmbeddings, create_deepseek_client, generate_deepseek_answer
# from GraphDatabase.models import CypherResponse, ValidationResponse
from pydantic_models import CypherResponse, ValidationResponse

import subprocess
# 代码运行前需要开启redis服务，这里直接写在app.py中，--daemonize yes可以保证子进程运行也生效。
subprocess.run("redis-server --daemonize yes --protected-mode no", shell=True)

os.environ["TOKENIZERS_PARALLELISM"] = "false"
app = FastAPI()

deepseek_api_key = os.getenv("DEEPSEEK_API_KEY")

# 模型1: 嵌入模型, 采用清华智谱最新的embedding-3, 实例化智谱client对象
client_embedding = ZhipuAiClient(api_key=os.getenv("ZHIPU_API_KEY"))

# 获取 neo4j 图数据库的连接
driver = GraphDatabase.driver(uri=uri, auth=auth, max_connection_lifetime=1000)

# 允许所有域的请求
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 创建 Embedding 模型
embedding_model = ZhipuAIEmbeddings(client_embedding)
print("创建 Embedding 模型成功......")

# 设置默认的 Milvus 数据库文件路径
URI = "./milvus_agent.db"

# 创建 Milvus 连接
milvus_vectorstore = Milvus(
    embedding_function=embedding_model,
    builtin_function=BM25BuiltInFunction(),
    vector_field=["dense", "sparse"],
    index_params=[
        {
            "metric_type": "IP",
            "index_type": "IVF_FLAT",
        },
        {
            "metric_type": "BM25",
            "index_type": "SPARSE_INVERTED_INDEX"
        }
    ],
    connection_args={"uri": URI},
)
retriever = milvus_vectorstore.as_retriever()
print("创建 Milvus 连接成功......")

# 创建大语言模型, 采用 DeepSeek
client_llm = create_deepseek_client()
print("创建 DeepSeek 成功......")

# 获取 redis 连接
client_redis = get_redis_client()
print("创建 Redis 连接成功......")


def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)


@app.post("/")
async def chatbot(request: Request):
    global milvus_vectorstore, retriever
    json_post_raw = await request.json()
    json_post = json.dumps(json_post_raw)
    json_post_list = json.loads(json_post)
    query = json_post_list.get('question')

    # 1: 先查 redis 缓存, 如果缓存命中, 直接返回结果
    response_redis = cache_get(client_redis, query)
    # print('response_redis: ', response_redis)

    if response_redis is not None:
        # redis 返回的字符串是以十六进制显示的, 需要按 utf-8 解码
        response = response_redis.decode('utf-8')

        now = datetime.datetime.now()
        time = now.strftime("%Y-%m-%d %H:%M:%S")
        answer = {
            "response": response,
            "status": 200,
            "time": time
        }
        print('REDIS HIT !!! ✅😊')

        return answer

    # 2: 向量数据库 Milvus 模糊召回 & 重排序
    # 在集合中搜索问题并检索语义 top-10 匹配项, 而且已经配置了 reranker 的处理
    recall_milvus = milvus_vectorstore.similarity_search(
        query,
        k=10,
        ranker_type="rrf",
        ranker_params={"k": 100}
    )

    if recall_milvus:
        # 检索结果存放在列表中
        # context = [r.page_content for r in recall_milvus]
        context = format_docs(recall_milvus)
    else:
        # context = []
        context = ""

    # 3: 图数据库 neo4j 精准召回
    # 访问 neo4j API 服务, 生成 Cypher 命令
    res = ""
    data = {"natural_language_query": query}
    data_json = json.dumps(data)
    # cypher_response = requests.post("http://0.0.0.0:8101/generate", data_json)
    cypher_response = requests.post("http://localhost:8101/generate", json={"natural_language_query": query})
    if cypher_response.status_code == 200:
        cypher_response_data = cypher_response.json()

        cypher_query = cypher_response_data["cypher_query"]
        confidence = cypher_response_data["confidence"]
        is_valid = cypher_response_data["validated"]

        res = ""

        cypher_valid = None
        
        if cypher_query is not None and float(confidence) >= 0.9 and is_valid == True:
            print("neo4j Cyhper 初步生成成功 !!!")
            # print(f"生成的 Cypher: {cypher_query}")

            # 验证 neo4j 生成的 Cypher 命令完全正确 ✅
            data = {"cypher_query": cypher_query}
            data_json = json.dumps(data)
            # cypher_valid = requests.post("http://localhost:8101/validate", data_json)
            cypher_valid = requests.post("http://localhost:8101/validate", json={"cypher_query": cypher_query})
            

            if cypher_valid.status_code == 200:
                cypher_valid_data = cypher_valid.json()

                # print(f"cypher_valid_data: {cypher_valid_data}")
                if cypher_valid_data["is_valid"] == True:
                    with driver.session() as session:
                        try:
                            record = session.run(cypher_query)
                            result = list(map(lambda x: x[0], record))
                            res = ','.join(result)

                            print(f"Neo4j 查询结果: {res}")
                        except Exception as e:
                            print(e)
                            print("neo4j查询失败 ❌!!")
                            res = ""
    else:
        # raise HTTPException(status_code=cypher_response.status_code, detail="生成Cypher查询失败 ❌!!")
        print("生成Cypher查询失败 ❌!!")
        pass

    # 合并 Milvus 和 neo4j 的召回结果, 共同作为 DeepSeek-V3.1 的输入 prompt
    context = context + "\n" + res

    # 为LLM定义系统和用户提示, 这个提示是由从Milvus检索到的文档组装而成的.
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

    # 4: 使用 DeepSeek 最新版本模型, 根据提示生成回复
    response = generate_deepseek_answer(client_llm, SYSTEM_PROMPT + USER_PROMPT.format(context, query))

    # 5: 写入缓存
    cache_set(client_redis, query, response)

    # 6: 组装服务返回数据
    now = datetime.datetime.now()
    time = now.strftime("%Y-%m-%d %H:%M:%S")
    answer = {
        "response": response,
        "status": 200,
        "time": time
    }

    return answer


if __name__ == '__main__':
    # 主函数中直接启动fastapi服务
    uvicorn.run(app, host='0.0.0.0', port=8103, workers=1)
