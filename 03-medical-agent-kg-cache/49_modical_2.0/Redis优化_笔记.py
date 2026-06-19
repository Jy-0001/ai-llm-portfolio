# Agent开发5: Redis 生产环境优化 (项目 v1.2 → v2.0)

# 课程核心
    # 上一节 v1.2 把 Redis 缓存接进来了, 单线程跑通了, 但放到生产环境会立刻崩
    # 这节给 v1.2 的四个致命隐患做工业级改造, 升级到 v2.0
    # 改造的核心文件: new_redis.py (Redis 工具类) + new_app.py (主服务调用方式)

# v1.2 的四个致命生产隐患
    # 1: BigKey (大 Key) 问题
        # 老代码: r.hset("qa", question, answer)  把所有问答塞进同一个 Hash, Key="qa"
        # 后果三连:
            # ① 百万条数据时 Hash 体积巨大, 任何操作都阻塞 Redis 单线程
            # ② 数据迁移困难 (单 Key 太大, 网络传输/序列化都慢)
            # ③ 无法对单条问答设过期 (Redis 的 expire 只能作用在 Key 级别, 不能作用在 Hash 的 Field 上)
    # 2: 连接池管理混乱
        # 老代码: def get_redis_client(): pool = redis.ConnectionPool(...); return ...
        # 每次调用都新建连接池
        # 后果: 高并发下每秒创建几千个连接池 → TCP 三次握手不断 → TIME_WAIT 堆积 → 端口/fd 耗尽 → Redis 拒绝连接
    # 3: 缺乏并发保护 (缓存击穿)
        # 热点 Key 过期那一瞬间, 所有请求同时打到 LLM, 浪费成本 + 拖垮服务
    # 4: 缺乏雪崩保护
        # 老代码: r.expire("qa", 3600)  固定过期时间
        # 后果: 同一时间灌入的缓存, 同一时间集体过期, 大批请求同时穿透到 LLM

# 词源 & 概念锚点
    # BigKey: 单个 Redis Key 体积过大的状态, "Big" 指数据量, 不是 Key 字符串本身长
    # Hash:   Redis 的一种数据结构, 一个 Key 下面挂多个 field-value 对, 类似字典
    # TCP:    Transmission Control Protocol, 传输控制协议
    # TIME_WAIT: TCP 连接关闭后的等待状态, 默认 60 秒才真正释放, 占着 fd 不放
    # fd:     File Descriptor, 文件描述符, OS 给每个连接/文件分配的编号, 总数有上限
    # 单例模式 (Singleton Pattern): 一个类全局只有一个实例, 所有调用方共享
    # 类变量 (Class Variable): 写在 class 里、方法外的变量, 该类所有实例共享同一份
    # 分布式互斥锁 (Distributed Mutex Lock): 跨进程/跨机器共享的锁, 同一时刻只允许一个持有
    # 缓存击穿 vs 缓存穿透 vs 缓存雪崩 (容易混)
        # 击穿 Breakdown: 热点 Key 过期瞬间, 大量请求打穿 → 单点
        # 穿透 Penetration: 大量请求查"根本不存在的 Key", 每次都打到后端 → 恶意/Bug
        # 雪崩 Avalanche: 大量 Key 同一时间过期, 大批请求同时穿透 → 集体
    # MD5: Message Digest 5, 一种哈希算法, 把任意长度输入压成固定 32 位十六进制字符串
    # Lua 脚本: Redis 服务端支持直接跑 Lua 代码, 多条命令打包成一个原子操作, 中间不会被打断
    # NX (nx=True): Not eXist, 仅当 Key 不存在时才设置, Redis 的原子化"加锁"原语
    # UUID: Universally Unique IDentifier, 全球唯一标识符, uuid4() 生成随机版本


# Part1: 问题1 - BigKey 解决方案
    # 思路: 把一个巨型 Hash 打散成无数个独立的小 String Key
    # Key 命名规范: prefix + MD5(question)
        # 例: "llm:cache:e10adc3949ba59abbe56e057f20f883e"
        # 用 MD5 是因为 question 长度不定、可能含中文/空格, 哈希后变成固定 32 位 ASCII, 可控
    # 实现 (new_redis.py 中):
        # def _generate_key(self, text, prefix="llm:cache:"):
        #     hash_obj = hashlib.md5(text.encode('utf-8'))
        #     return f"{prefix}{hash_obj.hexdigest()}"
    # 收益:
        # ① 每个 Key 独立, 不存在大 Key 阻塞
        # ② 每个 Key 可以单独设过期时间
        # ③ 数据迁移/分片都按 Key 走, 天然分散


# Part2: 问题2 - 连接池单例化
    # 思路: 全局只创建一次 ConnectionPool, 所有调用共享
    # 实现技术: 单例模式 + 类变量
    # 关键代码片段
        # class RedisClientWrapper:
        #     _pool = None                  # 类变量, 所有实例共享
        #     def __init__(self, host=..., port=...):
        #         if not RedisClientWrapper._pool:        # 第一次才建
        #             RedisClientWrapper._pool = redis.ConnectionPool(
        #                 max_connections=100,            # 上限 100, 超出排队
        #                 decode_responses=True,          # 自动 bytes→str
        #                 socket_timeout=5,
        #                 socket_connect_timeout=5,
        #             )
        #         self.client = redis.StrictRedis(connection_pool=RedisClientWrapper._pool)
        # redis_manager = RedisClientWrapper()  # 模块加载时实例化一次, 之后 import 都拿同一个
    # 类比 (老师的井和桶):
        # v1.2: 每次喝水就新打一口井, 喝完填掉 → 地上全是废井, 资源耗尽
        # v2.0: 全村就一口井 + 100 个桶, 用桶借水还桶 → 桶循环使用
    # 连接池行为
        # 池里维护一组 TCP 长连接 (不每次握手, 不每次 TIME_WAIT)
        # 请求来 "借" 一个空闲连接, 用完 "还" 回去 (不关闭)
        # 并发太高时新请求排队等待, 不会无限创建


# Part3: 问题3 - 缓存击穿 (分布式互斥锁 + Double Check)
    # 击穿场景
        # 热点 Key 过期那一瞬间, N 个请求同时 Cache Miss → 全部去调 LLM → 一份工作做 N 次, 浪费 N 倍成本和时间
    # 解决: 让其中"只有 1 个"请求去调 LLM, 其他 N-1 个等着, 等第 1 个把缓存写回再一起读
    # 锁的实现 (Redis SETNX + 过期时间)
        # def acquire_lock(self, lock_name, acquire_timeout=3, lock_timeout=10):
        #     identifier = str(uuid.uuid4())            # 锁的"身份标签", 用于安全释放
        #     lock_key = f"lock:{lock_name}"
        #     end = time.time() + acquire_timeout       # 抢锁总时间不超过 3 秒
        #     while time.time() < end:
        #         if self.client.set(lock_key, identifier, ex=lock_timeout, nx=True):
        #             return identifier                  # 抢到了, 返回 token
        #         time.sleep(0.01)                       # 没抢到, 等 10ms 再抢
        #     return None                                # 3 秒还没抢到, 放弃
    # 关键参数解释
        # nx=True (NOT eXist):  仅当 Key 不存在时才 SET → SET 成功 = 抢到锁 (原子操作, 无竞争)
        # ex=lock_timeout:      锁本身也设过期, 防止持锁线程死了锁永远不释放
        # identifier (UUID):    锁的归属标识, 释放锁时校验, 防止误删别人的锁 (见下面思考题2)
    # Double Check (双重验证) - 抢到锁之后必须再查一次缓存!!!
        # 为什么? 100 个线程同时 Cache Miss → A 抢到锁 → A 调 LLM 5 秒写入缓存 → A 释放锁
        # → B 抢到锁, 此时缓存其实已经有了!!! 如果 B 不再查一次直接调 LLM, 就是重复劳动
        # 加上 Double Check: B 抢到锁后再查 Redis 发现已有 → 直接返回, 不调 LLM
        # 收益: 100 个请求只调 1 次 LLM, 用户 B~ZZ 几乎毫秒返回
    # 原子释放锁 (Lua 脚本)
        # 释放锁 = "先 GET 比对 identifier, 再 DEL", 这两步如果不是原子的, 中间会插入别人的锁导致误删
        # Redis 支持服务端跑 Lua 脚本, 整段 Lua 是一个原子操作
        # if redis.call("get", KEYS[1]) == ARGV[1] then return redis.call("del", KEYS[1]) else return 0 end


# Part4: 问题4 - 缓存雪崩 (随机过期时间)
    # 雪崩场景
        # 批量灌入的缓存, 都设了同样的 3600 秒过期 → 1 小时后这批 Key 同时失效 → 同一时刻大批请求穿透
    # 解决: 给过期时间加随机抖动 ±10%
        # def set_answer(self, question, answer, expire_time=3600):
        #     key = self._generate_key(question)
        #     jitter = random.randint(int(-expire_time * 0.1), int(expire_time * 0.1))
        #     real_expire = expire_time + jitter         # 3240 ~ 3960 秒之间随机
        #     self.client.setex(key, real_expire, answer)
    # 效果: 原本 1 小时同步过期, 现在均匀分散在 12 分钟窗口内过期, 后端压力被打散


# Part5: 附加问题 - 缓存穿透 (<EMPTY> 负向缓存)
    # 穿透场景
        # 用户/恶意攻击者反复查询根本不存在的 Key (e.g. "如何用 Python 实现三体人的技术?")
        # 每次 Cache Miss → 每次都打到 LLM → 持续烧钱
    # 解决: "查无结果"也写缓存, 用占位符 <EMPTY> 标记
        # if not answer:  # LLM 返回空
        #     self.client.setex(self._generate_key(question), 60, "<EMPTY>")
        # 下次同样请求查到 <EMPTY>, 直接返回 None, 不再调 LLM
    # 过期时间设短 (60 秒): 因为"无答案"可能是临时的, 不能永久拒绝
    # 业务层处理: get_answer 读到 <EMPTY> 返回 None, 业务层根据 None 决定显示什么


# Part6: 三个思考题 (面试高频题)
    # 思考题1: 为什么要加 Double Check
        # 不加: 100 个 Cache Miss → 抢锁 → A 调 LLM 写缓存 → 释放锁 → B 抢到 → 又调 LLM → 重复 100 次
        # 加上: B 抢到锁后再查一次, 发现 A 已经写入了 → 直接返回, 只调 1 次 LLM
        # 本质: 缓存的状态在"抢锁等待期间"已经变了, 必须重新读取最新状态
    # 思考题2: UUID 是为了区分用户还是区分线程实例
        # 答案: 区分"持有锁的线程实例", 不是区分用户
        # 灾难场景 (无 UUID):
        #   A 抢锁(10s 过期) → A 卡了 15s → 第10s 锁自动过期 → B 抢到锁 → 第15s A 醒来 DEL 锁 → 误删 B 的锁 → C 又抢到 → B/C 并发执行, 互斥失效
        # 有 UUID 的安全释放:
        #   A 拿自己的 uuid_A 去 Redis: "锁的 value 还是 uuid_A 吗?" → 是 uuid_B → 不删
        #   Lua 脚本保证 GET + DEL 原子化
        # 核心: 锁的安全性 (Safety), 防止误删后来者的锁
    # 思考题3: 为什么用 <EMPTY> 而不是不写缓存
        # 不写: 同样的无效请求每次都打 LLM, 攻击者可以无限烧你的钱
        # 写 <EMPTY>: 第一次 1ms 拦截后续所有同样的请求
        # 这是工业界标准模式, 叫"缓存空对象 (Cache Empty Object)" 或"负向缓存 (Negative Cache)"


# Part7: 一站式封装 - get_or_compute 流程总览
    # 这是 RedisClientWrapper 的核心方法, 把所有保护逻辑封装成一个函数
    # 业务代码只需要传 (question, 一个回调函数 compute_func)
    # 流程
        # 1. 查缓存 (get_answer)
        #    命中 → 直接返回 (REDIS HIT)
        #    命中 <EMPTY> → 返回 None (穿透保护)
        # 2. Cache Miss → acquire_lock 抢锁
        # 3. 抢到锁
        #    Double Check 再查一次, 命中 → 返回 (REDIS HIT Double Check)
        #    没命中 → 调 compute_func() (回调里跑 RAG + LLM)
        #    answer 非空 → set_answer 写缓存 (带 jitter 防雪崩)
        #    answer 为空 → 写 <EMPTY> 占位 60 秒 (防穿透)
        #    finally → release_lock (Lua 原子释放)
        # 4. 没抢到锁 → sleep 100ms → 再查一次缓存
        #    有 → 返回
        #    没有 → 返回 "System busy" 提示
    # 业务调用方式 (new_app.py 中)
        # compute_callback = lambda: perform_rag_and_llm(query)
        # response = redis_manager.get_or_compute(query, compute_callback)
        # 业务层只关心"问题"和"算答案的方法", 所有并发/缓存逻辑都被 redis_manager 屏蔽了


# Part8: new_app.py 相比旧 app.py 的关键变化
    # 旧 v1.2: 缓存逻辑散落在 chatbot 函数里, 串行写死, 没有锁/雪崩/穿透保护
    # 新 v2.0:
        # ① import 新工具类: from new_redis import redis_manager
        # ② 把 RAG + LLM 整段逻辑抽成函数 perform_rag_and_llm(query) -> str
        # ③ chatbot 里只写一行: redis_manager.get_or_compute(query, lambda: perform_rag_and_llm(query))
        # ④ 注意: 这版示例代码砍掉了 NL2Cypher (neo4j) 那一路, 退回到只用 Milvus
        #    实际生产里应该把 neo4j 那路也封装进 perform_rag_and_llm
    # 设计模式: 回调函数 (Callback) + 模板方法 (Template Method)
        # get_or_compute 是模板, 定义了"缓存→锁→计算→写回"的固定骨架
        # compute_func 是回调, 由业务方填具体的"怎么计算"


# Part9: 工业界常见心法 (从这节抽象出的元规律)
    # 资源管理三原则
        # ① 建立成本低的多建, 维护成本高的共享 (TCP连接昂贵 → 共享池; StrictRedis实例轻量 → 多建)
        # ② 任何"可能并发"的操作都要分析竞态 (Race Condition Analysis)
        # ③ 任何"批量同时"的事件都要加随机抖动 (Jitter), 打散到时间窗口
    # 并发推演心法 (三遍法)
        # 第一遍: 单用户正常流程
        # 第二遍: N 个用户同时跑同一流程 → 找竞态/重复劳动
        # 第三遍: 某一步失败 → 降级方案 (e.g. Redis 挂了走全流程, neo4j 超时返回空)
    # 锁的安全性
        # 加锁必须原子 (SETNX/SET NX)
        # 锁本身要有过期时间 (防持锁方崩溃)
        # 释放锁要验身份 (UUID + Lua 原子化)
        # 持锁时间应短 (LLM 推理 5 秒 vs 锁过期 10 秒, 留缓冲)


# 关键代码文件
    # new_redis.py:  RedisClientWrapper 类
        # 类变量 _pool + 单例 redis_manager
        # _generate_key (MD5)
        # get_answer / set_answer (基础读写, 含 <EMPTY> 拦截 + jitter)
        # acquire_lock / release_lock (含 Lua 原子释放)
        # get_or_compute (一站式入口, 双重验证)
    # new_app.py:   主服务
        # from new_redis import redis_manager
        # perform_rag_and_llm 抽出 RAG+LLM 逻辑
        # chatbot 一行调用 redis_manager.get_or_compute
    # new_test.py:  发请求测试 (跟旧 test.py 几乎一样)


# 启动顺序 & 测试预期
    # 启动顺序: redis-server (后台) → python new_app.py (8103 端口)
    # 第一次请求: REDIS Cache Miss → 走完整 RAG+LLM → 耗时约 10 秒
    # 第二次相同请求: REDIS HIT → 直接返回缓存 → 耗时 5ms 以内 (提升约 2000 倍)
    # 并发请求同一 query: 只有 1 个调 LLM, 其他都从缓存读
