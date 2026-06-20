import os
import json
import logging
import datetime

import uvicorn
from dotenv import load_dotenv
from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from zai import ZhipuAiClient
from langchain_milvus import Milvus, BM25BuiltInFunction

from model import ZhipuAIEmbeddings, create_deepseek_client, generate_deepseek_answer

load_dotenv()
os.environ["TOKENIZERS_PARALLELISM"] = "false"

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
)
logger = logging.getLogger(__name__)

MILVUS_URI = os.getenv("MILVUS_TEXT_URI", "./milvus_agent.db")

app = FastAPI(title="Medical RAG Baseline")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Embedding: ZhipuAI embedding-3
client_embedding = ZhipuAiClient(api_key=os.getenv("ZHIPU_API_KEY"))
embedding_model = ZhipuAIEmbeddings(client_embedding)
logger.info("Embedding model initialized")

# Dense + sparse hybrid retrieval over Milvus
milvus_vectorstore = Milvus(
    embedding_function=embedding_model,
    builtin_function=BM25BuiltInFunction(),
    vector_field=["dense", "sparse"],
    index_params=[
        {"metric_type": "IP", "index_type": "IVF_FLAT"},
        {"metric_type": "BM25", "index_type": "SPARSE_INVERTED_INDEX"},
    ],
    connection_args={"uri": MILVUS_URI},
)
retriever = milvus_vectorstore.as_retriever()
logger.info("Milvus vector store connected: %s", MILVUS_URI)

# LLM client (DeepSeek API)
client_llm = create_deepseek_client()
logger.info("LLM client initialized")


def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)


@app.post("/")
async def chatbot(request: Request):
    json_post_raw = await request.json()
    json_post_list = json.loads(json.dumps(json_post_raw))
    query = json_post_list.get("question")
    logger.info("Received query: %s", query)

    # Hybrid recall + RRF reranking, top-10
    recall_rerank = milvus_vectorstore.similarity_search(
        query,
        k=10,
        ranker_type="rrf",
        ranker_params={"k": 100},
    )
    context = format_docs(recall_rerank) if recall_rerank else ""
    logger.info("Retrieved %d candidate documents", len(recall_rerank))

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

    response = generate_deepseek_answer(client_llm, system_prompt + user_prompt)

    answer = {
        "response": response,
        "status": 200,
        "time": datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
    }
    return answer


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8103, workers=1)
