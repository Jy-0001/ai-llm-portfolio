# -*- coding: utf-8 -*-
"""RAG 多路召回服务（FastAPI）：
  路1  文本 QA：Milvus dense+sparse 双索引 + 内置 RRF 重排
  路2  PDF 教材：Milvus + 父文档检索器（持久化 docstore）
  两路 context 融合 → LLM（本地 Qwen2.5-32B 或 DeepSeek API）生成。
"""
import os
import datetime

import uvicorn
from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from langchain_milvus import Milvus, BM25BuiltInFunction
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_classic.retrievers.parent_document_retriever import ParentDocumentRetriever
from langchain_classic.storage import LocalFileStore, create_kv_docstore

from model import (
    ZhipuAIEmbeddings,
    client as embedding_client,
    create_deepseek_client,
    generate_deepseek_answer,
)
from config import (
    MILVUS_TEXT_URI, MILVUS_PDF_URI, DOCSTORE_PATH,
    CHILD_CHUNK_SIZE, CHILD_CHUNK_OVERLAP,
    PARENT_CHUNK_SIZE, PARENT_CHUNK_OVERLAP,
    API_HOST, API_PORT, USE_LOCAL_LLM, get_logger,
)

os.environ["TOKENIZERS_PARALLELISM"] = "false"
logger = get_logger(__name__)

app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], allow_credentials=True,
    allow_methods=["*"], allow_headers=["*"],
)

_DENSE_INDEX = {"metric_type": "IP", "index_type": "IVF_FLAT"}
_SPARSE_INDEX = {"metric_type": "BM25", "index_type": "SPARSE_INVERTED_INDEX"}

# ---- 嵌入 ----
embedding_model = ZhipuAIEmbeddings(embedding_client)
logger.info("创建 Embedding 模型成功")

# ---- 路1：文本 QA 向量库 ----
milvus_vectorstore = Milvus(
    embedding_function=embedding_model,
    builtin_function=BM25BuiltInFunction(),
    vector_field=["dense", "sparse"],
    index_params=[_DENSE_INDEX, _SPARSE_INDEX],
    connection_args={"uri": MILVUS_TEXT_URI},
)
logger.info("创建文本 Milvus 连接成功: %s", MILVUS_TEXT_URI)

# ---- 路2：PDF 教材向量库 + 父文档检索器 ----
# 修复原 bug：父文档存到持久化 docstore（原 InMemoryStore 进程结束即丢，导致重启后第二路召回为空）
docstore = create_kv_docstore(LocalFileStore(DOCSTORE_PATH))
child_splitter = RecursiveCharacterTextSplitter(
    chunk_size=CHILD_CHUNK_SIZE, chunk_overlap=CHILD_CHUNK_OVERLAP,
    length_function=len,
    separators=["\n\n", "\n", "。", "！", "？", "；", "，", " ", ""],
)
parent_splitter = RecursiveCharacterTextSplitter(
    chunk_size=PARENT_CHUNK_SIZE, chunk_overlap=PARENT_CHUNK_OVERLAP,
)
pdf_vectorstore = Milvus(
    embedding_function=embedding_model,
    builtin_function=BM25BuiltInFunction(),
    vector_field=["dense", "sparse"],
    index_params=[_DENSE_INDEX, _SPARSE_INDEX],
    connection_args={"uri": MILVUS_PDF_URI},
    consistency_level="Bounded",
    drop_old=False,
)
parent_retriever = ParentDocumentRetriever(
    vectorstore=pdf_vectorstore, docstore=docstore,
    child_splitter=child_splitter, parent_splitter=parent_splitter,
)
logger.info("创建 PDF 父文档检索器成功: %s", MILVUS_PDF_URI)

# ---- LLM：本地 Qwen2.5-32B 或 DeepSeek API ----
_local_model = _local_tokenizer = _deepseek_client = None
if USE_LOCAL_LLM:
    from model import create_chat_model
    _local_model, _local_tokenizer = create_chat_model()
    logger.info("使用本地 Qwen2.5-32B 推理")
else:
    _deepseek_client = create_deepseek_client()
    logger.info("使用 DeepSeek API 推理")


def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)


def llm_generate(prompt: str) -> str:
    if USE_LOCAL_LLM:
        from model import generate_answer
        return generate_answer(_local_model, _local_tokenizer, prompt)
    return generate_deepseek_answer(_deepseek_client, prompt)


@app.post("/")
async def chatbot(request: Request):
    json_post = await request.json()
    query = json_post.get("question", "")

    # 路1：Milvus dense+sparse + 内置 RRF
    recall = milvus_vectorstore.similarity_search(
        query, k=10, ranker_type="rrf", ranker_params={"k": 100}
    )
    context = format_docs(recall) if recall else ""

    # 路2：PDF 父文档检索
    res = ""
    retrieved_docs = parent_retriever.invoke(query)
    if retrieved_docs:
        res = retrieved_docs[0].page_content

    context = (context + "\n" + res).strip()

    system_prompt = "你是一个非常得力的医学助手, 你可以通过从数据库中检索出的信息找到问题的答案."
    # 注意：直接 f-string 拼接，不再二次 .format()（原代码 context 含 {} 时会崩）
    user_prompt = (
        "利用<context></context>之间检索到的信息回答<question></question>之间的问题。"
        "若提供的信息为空，按你的医学知识严谨作答，不确定时坦诚说明，不要编造。\n"
        f"<context>\n{context}\n</context>\n"
        f"<question>\n{query}\n</question>"
    )
    response = llm_generate(system_prompt + "\n" + user_prompt)

    logger.info("query=%s | ctx_len=%d", query[:30], len(context))
    return {
        "response": response,
        "status": 200,
        "time": datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
    }


if __name__ == "__main__":
    uvicorn.run(app, host=API_HOST, port=API_PORT, workers=1)
