import redis
import json
import hashlib
import logging
import random
import time
import uuid
from typing import Optional, Callable, Any

# 配置日志
logger = logging.getLogger("RedisManager")
if not logger.handlers:
    logging.basicConfig(level=logging.INFO)


# 图灵君课堂小朱老师 • 独家讲义
class RedisClientWrapper:
    # 类变量, 所有实例共享, 实现单例连接池
    _pool = None

    def __init__(self, host='0.0.0.0', port=6379, db=0, password=None,
                 max_connections=100):
        # 1. 连接池单例化
        if not RedisClientWrapper._pool:
            RedisClientWrapper._pool = redis.ConnectionPool(
                host=host,
                port=port,
                db=db,
                password=password,
                max_connections=max_connections,
                decode_responses=True,  # 自动解码, 方便直接处理字符串
                socket_timeout=5,
                socket_connect_timeout=5
            )

        self.client = redis.StrictRedis(connection_pool=RedisClientWrapper._pool)

        # 2. 预加载 Lua 脚本 (原子性释放锁)
        self.unlock_script = self.client.register_script("""
            if redis.call("get", KEYS[1]) == ARGV[1] then
                return redis.call("del", KEYS[1])
            else
                return 0
            end
        """)

        # 测试连接
        try:
            self.client.ping()
            logger.info("Redis Connected Successfully ✅")
        except redis.ConnectionError:
            logger.error("Redis Connection Failed ❌")

    def _generate_key(self, text: str, prefix: str = "llm:cache:") -> str:
        # 生成缓存 Key, 使用 MD5 将任意长度的问题(text) 转换成固定长度的字符串
        hash_obj = hashlib.md5(text.encode('utf-8'))
        # 结果示例: "llm:cache:e10adc3949ba59abbe56e057f20f883e"
        return f"{prefix}{hash_obj.hexdigest()}"

    # 图灵君课堂小朱老师 • 独家讲义
    def get_answer(self, question: str) -> Optional[str]:
        # 基础读取缓存
        key = self._generate_key(question)
        try:
            val = self.client.get(key)
            if val:
                logger.info(f"Cache Hit ✅: {key}")
                # 防穿透占位符
                if val == "<EMPTY>":
                    # 返回 None, 表示"缓存命中了, 但命中了一个空结果"
                    # 业务层收到 None 后, 再决定当"未找到答案"时如何处理, 而不是直接去调 LLM
                    return None
                return val
        except redis.RedisError as e:
            logger.error(f"Redis Read Error: {e}")

        return None

    def set_answer(self, question: str, answer: str, expire_time: int = 3600):
        # 基础写入缓存 (带随机过期时间, 防止雪崩)
        key = self._generate_key(question)
        # 增加随机抖动 +/- 10%
        jitter = random.randint(int(-expire_time * 0.1), int(expire_time * 0.1))
        real_expire = expire_time + jitter

        try:
            self.client.setex(key, real_expire, answer)
        except redis.RedisError as e:
            logger.error(f"Redis Write Error: {e}")

    # 图灵君课堂小朱老师 • 独家讲义
    def acquire_lock(self, lock_name: str, acquire_timeout=3, lock_timeout=10) -> Optional[str]:
        # 获取分布式锁
        # 思考题: uuid 是为了区分用户呢? 还是区分线程实例呢?
        # 答: 区分线程实例, 用于安全释放, 防止误删别人的锁
        identifier = str(uuid.uuid4())
        lock_key = f"lock:{lock_name}"
        end = time.time() + acquire_timeout

        while time.time() < end:
            # 尝试获取锁 (nx=True 表示仅当键不存在时才设置, not exist)
            if self.client.set(lock_key, identifier, ex=lock_timeout, nx=True):
                return identifier
            # 获取不到锁, 做短暂的 10ms 停留, 然后重新尝试获取
            time.sleep(0.01)

        # 3 秒获取不到锁, 返回 None
        return None

    def release_lock(self, lock_name: str, identifier: str) -> bool:
        # 原子性释放锁
        lock_key = f"lock:{lock_name}"
        try:
            result = self.unlock_script(keys=[lock_key], args=[identifier])
            return bool(result)
        except redis.RedisError as e:
            logger.error(f"Lock Release Error: {e}")
            return False

    def get_or_compute(self, question: str, compute_func: Callable[[], str]) -> str:
        """
        核心: 防击穿/防穿透/防雪崩的智能获取
        :param question: 用户问题
        :param compute_func: 如果缓存未命中, 需要执行的耗时函数 (例如 LLM 推理)
        """
        # 1. 查缓存
        cached_ans = self.get_answer(question)
        if cached_ans:
            print('REDIS HIT !!! ✅😊')
            return cached_ans

        # 2. 缓存未命中, 加锁
        hash_key = hashlib.md5(question.encode('utf-8')).hexdigest()
        lock_token = self.acquire_lock(hash_key)

        if lock_token:
            try:
                # 双重验证 (思考题: 为什么要加上 Double Check?)
                # 答: 抢锁等待期间 cache 可能已经被别的线程写入, 必须再确认一次
                cached_ans_retry = self.get_answer(question)
                if cached_ans_retry:
                    print('REDIS HIT (Double Check) !!! ✅😊')
                    return cached_ans_retry

                print("Cache Miss ❌, Computing LLM...")
                # 3. 执行 LLM 推理 (回调函数)
                answer = compute_func()

                # 4. 写回缓存
                if answer:
                    self.set_answer(question, answer)
                else:
                    # 防缓存穿透 (思考题: 为什么要用一个 <EMPTY> 来防穿透呢?)
                    # 答: 拦截后续相同的无效查询, 1ms 直接返回不再调 LLM
                    self.client.setex(self._generate_key(question), 60, "<EMPTY>")

                return answer
            finally:
                self.release_lock(hash_key, lock_token)
        else:
            # 5. 获取锁失败 (有人在跑), 等待并重试
            time.sleep(0.1)
            # 再次尝试获取, 如果还是没有, 说明系统繁忙或刚才那次失败了
            return self.get_answer(question) or "System busy, calculating..."


# 单例模式初始化 (模块被 import 时执行一次, 全局共享)
redis_manager = RedisClientWrapper()
