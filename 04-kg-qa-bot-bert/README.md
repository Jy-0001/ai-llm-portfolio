# 04 · 医疗知识图谱智能问答机器人（BERT 意图识别 + neo4j）

LLM 时代之前的判别式 KGQA 范式：自建爬虫采集结构化医疗数据，构建知识图谱，再以"命名实体识别 + 意图分类 → 组装 Cypher → 检索图谱 → 生成回答"的链路完成问答。与项目 03 的生成式 Agent 形成"判别式/规则驱动 → 生成式"的技术演进对照。

## 实现要点
- **数据与图谱**：医疗数据爬虫采集，构建覆盖近 8800 种疾病、7 类实体节点（疾病/药品/食物/症状/检查/科室/厂商）的 neo4j 知识图谱。
- **命名实体识别**：BERT+CRF / IDCNN+CRF 序列标注（BERT 出发射分数，CRF 学转移矩阵 + Viterbi 解码全局最优标签序列，约束非法转移）。
- **意图识别**：从规则触发词匹配升级为 **BERT 微调的 8 分类**（问症状/问用药/问饮食/问检查/疾病-药品/疾病-食物 等）。
- **查询与回答**：`question_parser` 组装 Cypher 查询图谱，`answer_search` 生成回答。

## 关键文件
- `17_knowledge_graph_NER/bert_IDCNN/` — BERT+IDCNN+CRF 序列标注（`crf.py`、`idcnn_crf.py`、`train.py`）
- `18_knowledge_graph_RE/` — 关系抽取
- `19_red_spiderv1/` · `20_bert_redspider/` · `20_redspiderv1.1andv2/` — 数据采集与文本分类

## 技术栈
BERT · CRF / IDCNN · neo4j · 网络爬虫 · PyTorch · Flask / FastAPI
