# -*- coding: utf-8 -*-
"""集中配置：模型 / 向量库 / 分块 / 服务 / 日志。
所有可变项走环境变量（见仓库根 .env.example），不再硬编码。"""
import os
import sys
import logging
from dotenv import load_dotenv

load_dotenv()

# ---------------- 模型 ----------------
# 本地大模型基座（与作品集统一为 Qwen2.5-32B-Instruct）。
# AutoDL 上用 modelscope 下载到数据盘后，把路径配进 .env 的 LLM_MODEL_PATH。
LLM_MODEL_PATH = os.getenv(
    "LLM_MODEL_PATH",
    "/root/autodl-tmp/models/Qwen/Qwen2.5-32B-Instruct",
)
# 推理走哪个：本地 Qwen（合规/离线）还是 DeepSeek API（开发期快速验证）
USE_LOCAL_LLM = os.getenv("USE_LOCAL_LLM", "0") == "1"
# 嵌入模型（开发期用智谱 embedding-3 API；合规场景可换本地 bge-m3）
ZHIPU_EMBEDDING_MODEL = os.getenv("ZHIPU_EMBEDDING_MODEL", "embedding-3")

# ---------------- 向量库 / 文档存储 ----------------
MILVUS_TEXT_URI = os.getenv("MILVUS_TEXT_URI", "./milvus_agent.db")
MILVUS_PDF_URI = os.getenv("MILVUS_PDF_URI", "./pdf_agent.db")
# 父文档持久化目录（修复原 InMemoryStore 重启即丢父文档的 bug）
DOCSTORE_PATH = os.getenv("DOCSTORE_PATH", "./parent_docstore")

# ---------------- 分块 ----------------
CHILD_CHUNK_SIZE = int(os.getenv("CHILD_CHUNK_SIZE", "200"))
CHILD_CHUNK_OVERLAP = int(os.getenv("CHILD_CHUNK_OVERLAP", "50"))
PARENT_CHUNK_SIZE = int(os.getenv("PARENT_CHUNK_SIZE", "1000"))
PARENT_CHUNK_OVERLAP = int(os.getenv("PARENT_CHUNK_OVERLAP", "200"))

# ---------------- 服务 ----------------
API_HOST = os.getenv("API_HOST", "0.0.0.0")
API_PORT = int(os.getenv("API_PORT", "8103"))

# ---------------- 日志 ----------------
_FMT = "%(asctime)s | %(levelname)s | %(name)s | %(message)s"


def get_logger(name: str = "rag") -> logging.Logger:
    """同时输出到控制台 + logs/rag.log（logs/ 已被 .gitignore）。"""
    logger = logging.getLogger(name)
    if logger.handlers:
        return logger
    logger.setLevel(logging.INFO)
    ch = logging.StreamHandler(sys.stdout)
    ch.setFormatter(logging.Formatter(_FMT))
    logger.addHandler(ch)
    try:
        os.makedirs("logs", exist_ok=True)
        fh = logging.FileHandler(os.path.join("logs", "rag.log"), encoding="utf-8")
        fh.setFormatter(logging.Formatter(_FMT))
        logger.addHandler(fh)
    except OSError:
        pass
    return logger
