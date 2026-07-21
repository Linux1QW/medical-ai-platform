"""
全局测试配置与 fixture

关键：session 级别 mock 掉 llm_cache 的 Redis 连接，
避免 CI 环境中真实 Redis 容器干扰测试。
"""

import fnmatch
from unittest.mock import AsyncMock

import pytest

import app.services.llm_cache as llm_cache_module

# ── 全局 mock Redis（llm_cache 模块专用）────────────────────────────────────

def _build_mock_redis():
    """构建一个基于内存字典的 mock Redis 客户端"""
    store = {}
    counters = {}  # 用于 INCR 操作的持久化计数器

    async def mock_get(key):
        # 优先检查计数器（INCR 写入）
        if key in counters:
            return str(counters[key])
        return store.get(key)

    async def mock_set(key, value, ex=None):
        store[key] = value

    async def mock_setex(key, ttl, value):
        store[key] = value

    async def mock_ping():
        return True

    async def mock_scan(cursor, match, count):
        matched = [k for k in store.keys() if fnmatch.fnmatch(k, match)]
        return 0, matched

    async def mock_delete(*keys):
        for k in keys:
            store.pop(k, None)

    async def mock_ttl(key):
        return 3600 if key in store else -2

    async def mock_incr(key):
        counters[key] = counters.get(key, 0) + 1
        return counters[key]

    async def mock_hgetall(key):
        return store.get(key, {})

    async def mock_hincrby(key, field, amount=1):
        if key not in store:
            store[key] = {}
        store[key][field] = store[key].get(field, 0) + amount
        return store[key][field]

    async def mock_expire(key, ttl):
        pass

    redis_mock = AsyncMock()
    redis_mock.get = mock_get
    redis_mock.set = mock_set
    redis_mock.setex = mock_setex
    redis_mock.ping = mock_ping
    redis_mock.scan = mock_scan
    redis_mock.delete = mock_delete
    redis_mock.ttl = mock_ttl
    redis_mock.incr = mock_incr
    redis_mock.hgetall = mock_hgetall
    redis_mock.hincrby = mock_hincrby
    redis_mock.expire = mock_expire
    redis_mock._store = store
    redis_mock._counters = counters

    return redis_mock, store


@pytest.fixture(scope="session", autouse=True)
def mock_llm_cache_redis():
    """
    全局 mock LLM 缓存的 Redis 连接（session 级别）

    在所有测试开始前就将 _get_redis 替换为返回 mock 客户端的异步函数，
    确保 CI 上有真实 Redis 服务时也不会被测试代码连接。
    """
    redis_mock, store = _build_mock_redis()

    # 保存原始引用，用于测试结束后恢复
    original_get_redis = llm_cache_module._get_redis
    original_redis_client = llm_cache_module._redis_client

    # 直接替换：_get_redis 始终返回 mock 客户端
    async def mocked_get_redis():
        return redis_mock

    llm_cache_module._get_redis = mocked_get_redis
    llm_cache_module._redis_client = redis_mock

    # Mock 掉 _enforce_max_size，防止随机触发 1% 概率清理导致测试不稳定
    original_enforce_max_size = llm_cache_module._enforce_max_size

    async def noop_enforce_max_size(r):
        pass

    llm_cache_module._enforce_max_size = noop_enforce_max_size

    # 将 store 和 mock 暴露到模块上，方便测试 fixture 清空和方法覆盖
    llm_cache_module._mock_redis_store = store
    llm_cache_module._mock_redis = redis_mock

    yield redis_mock

    # 恢复原始状态
    llm_cache_module._get_redis = original_get_redis
    llm_cache_module._redis_client = original_redis_client
    llm_cache_module._enforce_max_size = original_enforce_max_size
    if hasattr(llm_cache_module, '_mock_redis_store'):
        delattr(llm_cache_module, '_mock_redis_store')
    if hasattr(llm_cache_module, '_mock_redis'):
        delattr(llm_cache_module, '_mock_redis')
