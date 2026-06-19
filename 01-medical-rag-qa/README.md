# 01 · 医疗垂直领域 RAG 知识库智能问答系统

面向医疗问诊场景的检索增强问答系统：以医疗问答语料与临床医学教材为知识源，构建覆盖疾病、用药、检查、饮食等多类问答的垂直知识库。推理基座统一为 **Qwen2.5-32B-Instruct**（本地，4-bit 量化），开发期亦可切换 **DeepSeek API**。

## 解决的核心问题
- **按字数硬切导致语义割裂** → 多级分块：`RecursiveCharacterTextSplitter` 基线 + `MarkdownHeader` 结构感知分块 + **父文档检索器（ParentDocumentRetriever）**（子块匹配保精度、回溯父块保上下文）。
- **query 与文档"问句↔陈述"弱匹配** → 召回增强：用 LLM 对每个文档块**反向生成代理问题**并向量化、经 UUID/doc_id 回溯原文；叠加 **HyDE / Step-back / RAG-Fusion** 扩大语义覆盖。
- **单一检索器召回不全 / 分数不可比** → **稠密 + 稀疏混合检索**（Milvus dense 向量 + BM25），用 **RRF** 融合多路排名，再经 **bce-reranker** 重排 + 上下文压缩提升信噪比、降 Token 成本。
- **医疗高风险幻觉** → 幻觉护栏 Prompt + 保留引用 `doc_id` 支持答案可追溯。

## 运行（AutoDL）
> 底座环境（Python 3.10 / CUDA 12.1 / torch 2.3.x cu121）见仓库根 `ENVIRONMENT.md`。

```bash
# 1) 装依赖
pip install -r 01-medical-rag-qa/requirements.txt

# 2) 配密钥（密钥不入库，从环境变量读）
cp .env.example .env        # 然后填入你自己的 ZHIPU_API_KEY / DEEPSEEK_API_KEY

# 3) 进项目目录
cd 01-medical-rag-qa/47_medical_1.1

# 4)（可选，USE_LOCAL_LLM=1 时）下载本地基座到数据盘
python ../download_models.py

# 5) 建索引：文本 QA + PDF 教材（父文档持久化到 DOCSTORE_PATH）
python vectors.py

# 6) 起服务
python agent2.py            # 默认 0.0.0.0:8103

# 7) 提问
curl -X POST http://127.0.0.1:8103/ -H "Content-Type: application/json" \
     -d '{"question":"高血压患者饮食上要注意什么?"}'
```

环境变量（全部见 `.env.example`）：`USE_LOCAL_LLM`（0=DeepSeek API / 1=本地 Qwen2.5-32B）、`LLM_MODEL_PATH`、`MILVUS_TEXT_URI`、`MILVUS_PDF_URI`、`DOCSTORE_PATH`、`API_PORT`。

## 评估
五维度 RAG 评估：检索层（Recall@K / Precision@K / MRR / nDCG）、生成一致性层（Faithfulness / Groundedness）、系统层（P95/P99 延迟 / QPS / 缓存命中率），离线评估 → 压测回放 → 指标对比驱动迭代。评估脚本见 `50_agent_evaluate/eval.py`。

## 目录
- `47_medical_1.1/` — **主体（推荐入口）**：`config.py` 集中配置、`model.py` 模型层、`vectors.py` 建索引、`agent2.py` 多路召回服务；`知识点单独实现/` 为各召回增强单点 demo。
- `46_langchain_medical_1/` — 早期基线版本（单路检索链）。
- `50_agent_evaluate/` — RAG 评估。
- `41_llm_RAG/` — SimCSE 语义检索 + FastAPI 流式服务与压测。

## 技术栈
LangChain · Milvus(dense+sparse) · BM25 · RRF · bce-reranker · HyDE/Step-back/RAG-Fusion · 智谱 embedding-3 · Qwen2.5-32B / DeepSeek · FastAPI

> 说明：本项目为学习/作品集项目，密钥经环境变量管理；模型权重与数据集不入库（见 `.gitignore`）。
