# -*- coding: utf-8 -*-
"""用 modelscope 把本地基座 / 重排模型下载到数据盘（国内比 HuggingFace 快很多）。
用法:  python download_models.py
下载目标由 .env 的 MODEL_CACHE_DIR 控制（默认 /root/autodl-tmp/models）。
"""
import os
from dotenv import load_dotenv
from modelscope import snapshot_download

load_dotenv()
CACHE = os.getenv("MODEL_CACHE_DIR", "/root/autodl-tmp/models")

MODELS = [
    "Qwen/Qwen2.5-32B-Instruct",      # 本地基座（USE_LOCAL_LLM=1 时用）
    "maidalun/bce-reranker-base_v1",  # 重排模型（可选）
]


def main():
    for mid in MODELS:
        print(f"downloading {mid} ...")
        path = snapshot_download(mid, cache_dir=CACHE)
        print(f"  -> {path}")
    print("done. 把 .env 的 LLM_MODEL_PATH 指向 Qwen2.5-32B-Instruct 的实际目录。")


if __name__ == "__main__":
    main()
