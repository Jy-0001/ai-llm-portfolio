# AI / LLM 算法工程作品集

> 围绕**医疗垂直领域**,从 NLP 基础到大模型微调对齐、RAG 检索增强、AI Agent 工作流的端到端实践作品集。
> 这些项目完成于一段约六个月的 AI 算法工程集中实训，主线是"把一个垂直领域问答系统，从判别式范式一路做到生成式 Agent，并对标工业级落地"。

---

## 🧭 项目总览

| # | 项目 | 一句话 | 核心技术 |
|---|------|--------|----------|
| 01 | [医疗垂直领域 RAG 知识库问答](./01-medical-rag-qa) | 多级分块 + 混合检索 + 重排 + 五维评估的 RAG 系统 | LangChain · Milvus · BM25+Dense · RRF · bce-reranker |
| 02 | [大模型 SFT + RLHF/DPO 微调](./02-llm-sft-rlhf) | 指令微调到人类偏好对齐的完整训练链路 | LoRA · DPO/PPO/GRPO/GSPO · DeepSpeed ZeRO · 多卡 |
| 03 | [医疗 AI Agent（知识图谱 + 生产级缓存）](./03-medical-agent-kg-cache) | NL2Cypher 多路召回 Agent + Redis 四类并发隐患修复 | neo4j · Redis · LangChain · 分布式锁 |
| 04 | [知识图谱问答机器人（BERT + neo4j）](./04-kg-qa-bot-bert) | LLM 时代之前的判别式 KGQA 范式 | BERT · CRF/IDCNN · neo4j · 爬虫 |
| 05 | [中文文本纠错与推理加速](./05-text-correction-inference) | CSC 纠错 + 模型压缩/ONNX 部署优化 | MacBERT4CSC · SpanPointer · SimCSE · ONNX · 量化/蒸馏/剪枝 |

每个子目录有独立 README，说明该项目解决的问题、技术选型理由、关键实现与踩过的坑。

---

## 🛠 技术栈

- **语言/框架**：Python、PyTorch、FastAPI
- **大模型**：Qwen3、Llama、ChatGLM、DeepSeek；LoRA / SFT / DPO / PPO / GRPO / GSPO
- **训练**：LLaMA-Factory、DeepSpeed ZeRO（多卡分布式）、混合精度
- **RAG / 检索**：LangChain、Milvus、Chroma、BM25、RRF 融合、bce-reranker、HyDE / Step-back / RAG-Fusion
- **Agent / 图谱**：neo4j、NL2Cypher、Redis（分布式锁 / 连接池 / 缓存击穿穿透雪崩防护）
- **NLP 基础**：BERT / ALBERT / T5、CRF、SpanPointer、SimCSE、ONNX、知识蒸馏 / 量化 / 剪枝

---

## ⚠️ 说明

- 本仓库为**学习与作品集性质**，仅包含本人编辑的源代码；模型权重、训练数据、第三方开源仓库（LLaMA-Factory / DeepSpeedExamples 等）均未纳入。
- 部分脚本依赖本地数据 / 模型路径与外部服务（neo4j、Redis、向量库），克隆后需自行配置环境与数据方可运行；代码以展示思路与实现为主。
