# 医疗健康AI Agent项目开发：知识图谱neo4j + Redis缓存 (Agent开发4 / 项目V1.2)
    # 工具：Redis, neo4j, pymilvus 2.6.6, FastAPI
    # 架构升级核心变化
        # 旧 (V1.1, Agent开发3)：Milvus文本召回 + Milvus PDF父子检索 → LLM
        # 新 (V1.2, Agent开发4)：Redis缓存 → Milvus文本召回 + neo4j精准召回 → LLM → 写回Redis
        # 注意：PDF父子检索这一路被砍了 (大概率因为上节课 docstore 不持久化的坑)
        #       换成知识图谱, 走 NL2Cypher 的路子
    # 知识图谱 neo4j 优化 RAG
        # 核心矛盾: LLM不可靠性 vs 数据库查询正确性
            # 工业一线绝不允许"错误的查询命令执行" → 要么直接报错, 要么严重后果
        # 四种约束方法 (LLM生成SQL/Cypher的通用工程模式)：
            # 1. 严格定义schema, LLM只在有限集合中选
            # 2. 严格检查LLM生成的查询是否满足规范
            # 3. 用解释器实际验证 (EXPLAIN cypher)
            # 4. 验证返回结果格式
        # 五个代码模块 (NL2Cypher 完整架构)
            # schemas.py     图schema定义 (Pydantic NodeSchema / RelationshipSchema / GraphSchema)
                # 实际schema: Disease/Drug/Food/Symptom四节点
                #            has_symptom / recommand_drug / recommand_eat三关系
            # prompts.py     两个prompt工厂函数
                # create_system_prompt: 注入schema + 7条规则 + 3个few-shot示例 → 生成Cypher
                # create_validation_prompt: 让LLM对生成的Cypher自审, 找错误+给建议
            # models.py      请求/响应 Pydantic 类
                # NL2CypherRequest / CypherResponse        : /generate 端点
                # ValidationRequest / ValidationResponse   : /validate 端点
                # QueryType (Enum)                          : 限定MATCH/CREATE/MERGE/DELETE等
            # validators.py  两套验证器
                # CypherValidator      接neo4j, EXPLAIN验证语法 + 正则查schema匹配
                # RuleBasedValidator   不连neo4j时的fallback, 纯正则规则
            # main.py        FastAPI 服务 (port 8101)
                # /generate  NL → Cypher + 解释 + 置信度 + 验证
                # /validate  单独验证已有Cypher + LLM生成改进建议
                # /schema    返回当前图schema
                # lifespan(): 启动时根据环境变量择优用 CypherValidator / RuleBasedValidator
        # 关键设计点
            # 置信度衰减: 基线0.9, 每发现一个错误扣0.1, 最低0.3
            # 双重把关: LLM生成 → schema匹配验证 → EXPLAIN语法验证 → 真正执行
            # 端口分工: neo4j服务 8101, Agent主服务 8103
    # Redis 缓存命中
        # 商业动机: DeepSeek API 缓存命中价 ≈ 未命中价 1/10 (0.2元 vs 2元 / 百万tokens)
        # 实现
            # vectors.py 新增 get_redis_client / cache_set / cache_get
            # app.py 在主接口最前面 cache_get → 命中直接返回; LLM生成后 cache_set 写入
        # 实测效果: 首查 10.6 秒, 二查 0.006 秒, 提速 ~2000 倍
        # 实现细节
            # 用 hset("qa", question, answer) 把所有问答塞进同一个 Hash 表
            # r.expire("qa", 3600) 每次写入刷新整个 hash 过期时间
            # redis 返回 bytes, 需 .decode('utf-8') 转字符串
    # 项目完整流程代码调用逻辑：
        # 调用逻辑：query通过fastapi post 到app.py > 搜索redis缓存 > 命中：直接输出
            #                                           ↓
            #    rag检索：milvus with BM25双路重排序检索 + fastapi post 到 NL2Cypher服务 > 将两路召回结果合并通过prompt输入LLM > 存入redis缓存(哈希存入) > 输出结果
            #               ↓                              ↓                    ↑如果有效
            #           未找到则记录为空       generate先将NL生成Cypher语句 > validate验证生成的Cypher语句的有效性/安全性
            #                                              ↓未生成              ↓无效
            #                                            未生成则为空         无效则为空
        # 代码逻辑：
            # neo4j数据库准备：
                # neo4j配置：config.py
                # 数据：data/medical.json
                # 代码：build_medicalgraph录入数据
                    # 前提：neo4j start 开启服务
            # query 请求发出：test.py
            # 流程主入口：app.py
                # RAG检索模型以及数据库准备：vectors.py
                # NL2Cypher入口以及生成/验证函数定义：main.py
                    # 定义cypher prompt格式：prompt.py
                    # 定义cypher 相关函数格式：pydantic_models.py
                    # 定义schema格式：schemas.py
                    # 验证cypher语句代码逻辑：validators.py
                        # neo4j EXPLAIN语句以及schema格式验证：CypherValidator
                        # 规则验证：RuleBasedValidator
    # 老师自己反思的 V1.2 生产隐患 (下节课 V2.0 要解决)
        # ① 大Key问题: 单 Hash 存所有 QA, 百万级时阻塞, 且 Redis 只能对 Key 过期不能对 Field 过期
        # ② 连接池管理: 每次调函数都新建连接池, 最终耗尽连接拒绝服务
        # ③ 缓存击穿: Hot Key 失效瞬间, 并发请求全打到 LLM
        # ④ 缓存雪崩: 过期时间固定 3600 秒, 容易大量缓存同时失效
    # 关键认知
        # NL2SQL / NL2Cypher 通用模式: 提示工程 + schema约束 + 语法验证 + 结果验证
        #   LLM 把自然语言转结构化查询的本质都是这个套路, 数据库种类不影响
        # 缓存价值不只是省钱, 更是省时间 (10秒 → 5ms, 用户体验完全不同档)
        # "演示 vs 生产"鸿沟: 讲义代码能跑通"happy path", 但生产环境暴露的 bug
        #   (类型不匹配/方法名错位/连接池失控/缓存策略不当) 才是真正学到东西的地方
