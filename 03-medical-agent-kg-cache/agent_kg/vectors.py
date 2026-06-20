import os
import json
import uuid
import time
import logging

from dotenv import load_dotenv
from tqdm import tqdm
import redis
from zai import ZhipuAiClient
from langchain.embeddings.base import Embeddings
from langchain_core.documents import Document
from langchain_milvus import Milvus, BM25BuiltInFunction

load_dotenv()
logger = logging.getLogger(__name__)


def get_redis_client():
    # Connection pool (recommended for production)
    pool = redis.ConnectionPool(
        host=os.getenv("REDIS_HOST", "0.0.0.0"),
        port=int(os.getenv("REDIS_PORT", "6379")),
        db=0,
        password=os.getenv("REDIS_PASSWORD") or None,
        max_connections=10,
    )
    r = redis.StrictRedis(connection_pool=pool)

    try:
        r.ping()
        logger.info("Connected to Redis")
    except redis.ConnectionError:
        logger.error("Failed to connect to Redis")

    return r


# 将 (question, answer) 问答对, 存入 redis
def cache_set(r, question: str, answer: str):
    r.hset("qa", question, answer)
    r.expire("qa", 3600)


# 通过 question, 读取存在 redis 中的 answer
def cache_get(r, question: str):
    return r.hget("qa", question)

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


class Milvus_vector():
    def __init__(self, client, uri="./milvus_agent.db"):
        self.URI = uri
        self.embeddings = ZhipuAIEmbeddings(client=client)

        # 定义索引类型
        self.dense_index = {
            "metric_type": "IP",
            "index_type": "IVF_FLAT",
        }
        self.sparse_index = {
            "metric_type": "BM25",
            "index_type": "SPARSE_INVERTED_INDEX"
        }

    def create_vector_store(self, docs):
        init_docs = docs[:10]
        self.vectorstore = Milvus.from_documents(
            documents=init_docs,
            embedding=self.embeddings,
            builtin_function=BM25BuiltInFunction(),  # output_field_names="sparse",
            index_params=[self.dense_index, self.sparse_index],
            vector_field=["dense", "sparse"],
            connection_args={
                "uri": self.URI,
            },
            consistency_level="Bounded",  # 支持 ("Strong", "Session", "Bounded", "Eventually")
            drop_old=False,
        )
        logger.info("Milvus collection initialized")

        count = 10
        temp = []
        for doc in tqdm(docs[10:]):
            temp.append(doc)
            if len(temp) >= 5:
                self.vectorstore.aadd_documents(temp)
                count += len(temp)
                temp = []
                logger.info("Inserted %d documents", count)
                time.sleep(1)

        logger.info("Total %d documents inserted; Milvus index built", count)

        return self.vectorstore


# 插入数据
def prepare_document(file_path=['./data/data.jsonl', './data/train.jsonl']):
    # 逐条取出文本数据, 创建嵌入张量, 然后将张量数据插入Milvus
    file_path1 = file_path[0]

    count = 0
    docs = []
    with open(file_path1, 'r', encoding='utf-8') as f:
        for line in f:
            content = json.loads(line.strip())
            prompt = content['query'] + "\n" + content['response']

            temp_doc = Document(page_content=prompt, metadata={"doc_id": str(uuid.uuid4())})
            docs.append(temp_doc)
            count += 1

    logger.info("Loaded %d documents", count)

    return docs


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
    )
    docs = prepare_document()
    milvus_vectorstore = Milvus_vector(client)
    vectorstore = milvus_vectorstore.create_vector_store(docs)
    get_redis_client()
    logger.info("Initialization complete")
