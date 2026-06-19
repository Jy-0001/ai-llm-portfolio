import os
from dotenv import load_dotenv
load_dotenv()
import os
import uvicorn
from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
import json
import datetime
from zai import ZhipuAiClient
from langchain_milvus import Milvus, BM25BuiltInFunction
from model import ZhipuAIEmbeddings, create_deepseek_client, generate_deepseek_answer


os.environ["TOKENIZERS_PARALLELISM"] = "false"

app = FastAPI()

deepseek_api_key = os.getenv("DEEPSEEK_API_KEY")


# 模型1: 嵌入模型, 采用清华智谱最新的embedding-3
# 实例化智谱client对象
client_embedding = ZhipuAiClient(api_key=os.getenv("ZHIPU_API_KEY"))


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


# 创建大语言模型, 采用 DeepSeek-V3.2
client_llm = create_deepseek_client()
print("创建 DeepSeek-V3.2 成功......")


def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)


@app.post("/")
async def chatbot(request: Request):
    global milvus_vectorstore, retriever
    json_post_raw = await request.json()
    json_post = json.dumps(json_post_raw)
    json_post_list = json.loads(json_post)
    query = json_post_list.get('question')

    # 召回 & 排序
    # 在集合中搜索问题并检索语义 top-10 匹配项, 而且已经配置了 reranker 的处理, 采用RRF算法
    recall_rerank_milvus = milvus_vectorstore.similarity_search(
        query,
        k=10,
        ranker_type="rrf",
        ranker_params={"k": 100}
    )

    if recall_rerank_milvus:
        # 检索结果存放在列表中
        # context = [r.page_content for r in recall_rerank_milvus]
        context = format_docs(recall_rerank_milvus)
    else:
        context = []

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

    # 使用DeepSeek V3.2最新版本模型, 根据提示生成回复
    response = generate_deepseek_answer(client_llm,
                                        SYSTEM_PROMPT + USER_PROMPT.format(context, query))

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