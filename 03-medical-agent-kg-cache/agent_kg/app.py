import os
import sys
import json
import logging
import datetime
import subprocess

from dotenv import load_dotenv
import uvicorn
from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
import requests

from config import uri, auth
from neo4j import GraphDatabase
from zai import ZhipuAiClient
from langchain_milvus import Milvus, BM25BuiltInFunction
from vectors import get_redis_client, cache_set, cache_get

load_dotenv()
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
)
logger = logging.getLogger(__name__)

sys.path.append('../../01-medical-rag-qa/rag_pipeline')
from model import ZhipuAIEmbeddings, create_deepseek_client, generate_deepseek_answer

# Start a local Redis server (idempotent) before serving
subprocess.run("redis-server --daemonize yes --protected-mode no", shell=True)

os.environ["TOKENIZERS_PARALLELISM"] = "false"
app = FastAPI(title="Medical Agent (KG + Cache)")

# Embedding client (ZhipuAI embedding-3)
client_embedding = ZhipuAiClient(api_key=os.getenv("ZHIPU_API_KEY"))

# Neo4j graph database connection
driver = GraphDatabase.driver(uri=uri, auth=auth, max_connection_lifetime=1000)

# 允许所有域的请求
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

embedding_model = ZhipuAIEmbeddings(client_embedding)
logger.info("Embedding model initialized")

URI = os.getenv("MILVUS_TEXT_URI", "./milvus_agent.db")

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
logger.info("Milvus vector store connected: %s", URI)

client_llm = create_deepseek_client()
logger.info("LLM client initialized")

client_redis = get_redis_client()
logger.info("Redis client initialized")


def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)


@app.post("/")
async def chatbot(request: Request):
    global milvus_vectorstore, retriever
    json_post_raw = await request.json()
    json_post = json.dumps(json_post_raw)
    json_post_list = json.loads(json_post)
    query = json_post_list.get('question')

    # Step 1: serve from Redis cache if present
    response_redis = cache_get(client_redis, query)

    if response_redis is not None:
        response = response_redis.decode('utf-8')
        logger.info("Cache hit, returning cached answer")
        return {
            "response": response,
            "status": 200,
            "time": datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        }

    # Step 2: dense + sparse recall with RRF reranking over Milvus (top-10)
    recall_milvus = milvus_vectorstore.similarity_search(
        query,
        k=10,
        ranker_type="rrf",
        ranker_params={"k": 100}
    )
    context = format_docs(recall_milvus) if recall_milvus else ""

    # Step 3: precise recall from the Neo4j knowledge graph via NL2Cypher service
    res = ""
    cypher_response = requests.post(
        "http://localhost:8101/generate", json={"natural_language_query": query}
    )
    if cypher_response.status_code == 200:
        cypher_response_data = cypher_response.json()

        cypher_query = cypher_response_data["cypher_query"]
        confidence = cypher_response_data["confidence"]
        is_valid = cypher_response_data["validated"]

        res = ""

        if cypher_query is not None and float(confidence) >= 0.9 and is_valid == True:
            logger.info("Cypher generated, validating before execution")

            # Validate the generated Cypher before running it against Neo4j
            cypher_valid = requests.post(
                "http://localhost:8101/validate", json={"cypher_query": cypher_query}
            )

            if cypher_valid.status_code == 200:
                cypher_valid_data = cypher_valid.json()
                if cypher_valid_data["is_valid"] == True:
                    with driver.session() as session:
                        try:
                            record = session.run(cypher_query)
                            result = list(map(lambda x: x[0], record))
                            res = ','.join(result)
                            logger.info("Neo4j query result: %s", res)
                        except Exception as e:
                            logger.error("Neo4j query failed: %s", e)
                            res = ""
    else:
        logger.warning("Cypher generation failed")

    # Merge Milvus and Neo4j recall as the LLM context
    context = context + "\n" + res

    system_prompt = (
        "你是一个非常得力的医学助手, 你可以通过从数据库中检索出的信息找到问题的答案."
    )
    user_prompt = f"""利用介于<context>和</context>之间的从数据库中检索出的信息来回答问题, 具体的问题介于<question>和</question>之间. 如果提供的信息为空, 则按照你的经验知识来给出尽可能严谨准确的回答, 不知道的时候坦诚的承认不了解, 不要编造不真实的信息.
    <context>
    {context}
    </context>
    <question>
    {query}
    </question>
    """

    # Step 4: generate the answer with the merged context
    response = generate_deepseek_answer(client_llm, system_prompt + user_prompt)

    # Step 5: write back to cache
    cache_set(client_redis, query, response)

    return {
        "response": response,
        "status": 200,
        "time": datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
    }


if __name__ == '__main__':
    uvicorn.run(app, host='0.0.0.0', port=8103, workers=1)
