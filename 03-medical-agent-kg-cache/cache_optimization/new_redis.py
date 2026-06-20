import redis
import hashlib
import logging
import random
import time
import uuid
from typing import Optional, Callable

logger = logging.getLogger("RedisManager")
if not logger.handlers:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
    )


class RedisClientWrapper:
    # Class-level connection pool shared by all instances (singleton)
    _pool = None

    def __init__(self, host='0.0.0.0', port=6379, db=0, password=None,
                 max_connections=100):
        if not RedisClientWrapper._pool:
            RedisClientWrapper._pool = redis.ConnectionPool(
                host=host,
                port=port,
                db=db,
                password=password,
                max_connections=max_connections,
                decode_responses=True,
                socket_timeout=5,
                socket_connect_timeout=5,
            )

        self.client = redis.StrictRedis(connection_pool=RedisClientWrapper._pool)

        # Preload Lua script for atomic lock release
        self.unlock_script = self.client.register_script("""
            if redis.call("get", KEYS[1]) == ARGV[1] then
                return redis.call("del", KEYS[1])
            else
                return 0
            end
        """)

        try:
            self.client.ping()
            logger.info("Redis connected successfully")
        except redis.ConnectionError:
            logger.error("Redis connection failed")

    def _generate_key(self, text: str, prefix: str = "llm:cache:") -> str:
        # MD5 the question into a fixed-length cache key
        hash_obj = hashlib.md5(text.encode('utf-8'))
        return f"{prefix}{hash_obj.hexdigest()}"

    def get_answer(self, question: str) -> Optional[str]:
        key = self._generate_key(question)
        try:
            val = self.client.get(key)
            if val:
                logger.info("Cache hit: %s", key)
                # Negative-cache placeholder: hit, but the answer is empty.
                # The caller receives None and decides how to handle "not found"
                # instead of calling the LLM again.
                if val == "<EMPTY>":
                    return None
                return val
        except redis.RedisError as e:
            logger.error("Redis read error: %s", e)

        return None

    def set_answer(self, question: str, answer: str, expire_time: int = 3600):
        # Write with +/-10% TTL jitter to avoid synchronized expiry (cache avalanche)
        key = self._generate_key(question)
        jitter = random.randint(int(-expire_time * 0.1), int(expire_time * 0.1))
        real_expire = expire_time + jitter

        try:
            self.client.setex(key, real_expire, answer)
        except redis.RedisError as e:
            logger.error("Redis write error: %s", e)

    def acquire_lock(self, lock_name: str, acquire_timeout=3, lock_timeout=10) -> Optional[str]:
        # Distributed lock. The UUID identifies the holder so release only
        # deletes our own lock, never someone else's.
        identifier = str(uuid.uuid4())
        lock_key = f"lock:{lock_name}"
        end = time.time() + acquire_timeout

        while time.time() < end:
            # nx=True: set only if the key does not already exist
            if self.client.set(lock_key, identifier, ex=lock_timeout, nx=True):
                return identifier
            time.sleep(0.01)

        return None

    def release_lock(self, lock_name: str, identifier: str) -> bool:
        # Atomic release via the preloaded Lua script
        lock_key = f"lock:{lock_name}"
        try:
            result = self.unlock_script(keys=[lock_key], args=[identifier])
            return bool(result)
        except redis.RedisError as e:
            logger.error("Lock release error: %s", e)
            return False

    def get_or_compute(self, question: str, compute_func: Callable[[], str]) -> str:
        """Cache-aside read protecting against stampede / penetration / avalanche.

        :param question: user question
        :param compute_func: expensive fallback (e.g. an LLM call) on cache miss
        """
        cached_ans = self.get_answer(question)
        if cached_ans:
            logger.info("Cache hit, returning cached answer")
            return cached_ans

        # Cache miss: acquire a per-question lock so only one worker computes
        hash_key = hashlib.md5(question.encode('utf-8')).hexdigest()
        lock_token = self.acquire_lock(hash_key)

        if lock_token:
            try:
                # Double check: another worker may have filled the cache while
                # we were waiting for the lock.
                cached_ans_retry = self.get_answer(question)
                if cached_ans_retry:
                    logger.info("Cache hit on double check")
                    return cached_ans_retry

                logger.info("Cache miss, computing via fallback")
                answer = compute_func()

                if answer:
                    self.set_answer(question, answer)
                else:
                    # Negative cache: block repeated invalid queries from hitting
                    # the LLM by storing a short-lived <EMPTY> marker.
                    self.client.setex(self._generate_key(question), 60, "<EMPTY>")

                return answer
            finally:
                self.release_lock(hash_key, lock_token)
        else:
            # Lock not acquired (another worker is computing): wait and retry
            time.sleep(0.1)
            return self.get_answer(question) or "System busy, calculating..."


# Module-level singleton, initialized once on import
redis_manager = RedisClientWrapper()
