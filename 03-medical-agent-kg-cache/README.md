# 03 · 医疗健康 AI Agent 智能体工作流（知识图谱 + 生产级缓存）

在 RAG 基础上升级为多路召回 Agent：用户 query → FastAPI 入口 → Redis 缓存查询（命中直返）→ 未命中则并行走 Milvus 文本召回与 neo4j 知识图谱精准召回 → 合并上下文喂 LLM 生成 → 结果写回缓存。

## NL2Cypher 知识图谱召回
针对"LLM 生成查询不可靠 vs 数据库要求绝对正确"的矛盾，设计**四重约束工程模式**：
1. 严格 schema 约束（Disease / Drug / Food / Symptom 等节点与关系）
2. LLM 生成后规则校验
3. neo4j `EXPLAIN` 解释器语法验证
4. 返回结果格式验证

配合置信度衰减（基线 0.9，每发现一处错误 -0.1）与 LLM 自审改进，杜绝错误 Cypher 落库执行。

## 【核心】生产级 Redis 缓存四类并发隐患修复
| 隐患 | 方案 |
|------|------|
| **BigKey** | "单 Hash 存全部 QA" → `prefix + MD5(question)` 独立 String Key，解决百万级单 Key 阻塞、无法按条过期 |
| **连接池泄漏** | 单例模式 + 类变量全局复用 `ConnectionPool`，杜绝 TIME_WAIT 堆积、FD 耗尽 |
| **缓存击穿** | `SETNX` 分布式互斥锁 + UUID 身份标识 + Lua 原子释放 + Double Check，使热点 Key 失效瞬间 N 个并发仅 1 个穿透到 LLM |
| **缓存雪崩 / 穿透** | 过期时间 ±10% 随机抖动打散；`<EMPTY>` 负向缓存拦截不存在 Key |

**收益**：缓存命中后相同问题响应时间约 10s → 5ms（≈2000x）；API 缓存命中成本约为未命中的 1/10。
**沉淀**：将"查缓存 → 加锁 → 计算 → 写回"封装为 `get_or_compute` 模板方法 + 回调，业务层一行调用屏蔽全部并发复杂度。

## 运行（AutoDL）
> 底座见仓库根 `ENVIRONMENT.md`；本项目复用 01 的模型层（`rag_pipeline/model.py`），推理可走 DeepSeek API 或本地 Qwen2.5-32B。
```bash
pip install -r 03-medical-agent-kg-cache/requirements.txt
cp .env.example .env        # 填 ZHIPU/DEEPSEEK key 与 NEO4J_*、REDIS_* 配置
# 需本机/容器内有 neo4j 与 redis-server
cd 03-medical-agent-kg-cache/agent_kg
python build_medicalgraph.py   # 构建医疗知识图谱
python main.py                 # 起 Agent 服务
# Redis 优化版见 ../cache_optimization/new_app.py
```
Neo4j 口令、Redis 地址等全部走环境变量（见 `.env.example` 的 `NEO4J_*` / `REDIS_*`），不再硬编码。

## 关键文件
- `agent_kg/` — Agent 主体：`main.py`、`build_medicalgraph.py`、`validators.py`、`vectors.py`、`config.py`（env 化）、`知识图谱与缓存_笔记.py`
- `cache_optimization/` — Redis 生产级优化（`new_redis.py` / `new_app.py`）

## 技术栈
LangChain · neo4j · Redis · Milvus · FastAPI · Pydantic · Lua · Qwen2.5-32B / DeepSeek
