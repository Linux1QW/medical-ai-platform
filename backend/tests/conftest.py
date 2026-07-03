"""
全局测试配置与 fixture

关键：session 级别 mock 掉 llm_cache 的 Redis 连接，
避免 CI 环境中真实 Redis 容器干扰测试。
"""

import fnmatch
import pytest
from unittest.mock import AsyncMock

import app.services.llm_cache as llm_cache_module


# ── 全局 mock Redis（llm_cache 模块专用）────────────────────────────────────

def _build_mock_redis():
    """构建一个基于内存字典的 mock Redis 客户端"""
    store = {}

    async def mock_get(key):
        return store.get(key)

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

    redis_mock = AsyncMock()
    redis_mock.get = mock_get
    redis_mock.setex = mock_setex
    redis_mock.ping = mock_ping
    redis_mock.scan = mock_scan
    redis_mock.delete = mock_delete
    redis_mock.ttl = mock_ttl
    redis_mock._store = store

    return redis_mock, store


@pytest.fixture(scope="session", autouse=True)
def mock_llm_cache_redis():
    """
    全局 mock LLM 缓存的 Redis 连接（session 级别）

    在所有测试开始前就将 _get_redis 替换为返回 mock 客户端的异步函数，
    确保 CI 中有真实 Redis 服务时也不会被测试代码连接。
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

    # 将 store 和 mock 暴露到模块上，方便测试 fixture 清空和方法覆盖
    llm_cache_module._mock_redis_store = store
    llm_cache_module._mock_redis = redis_mock

    yield redis_mock

    # 恢复原始状态
    llm_cache_module._get_redis = original_get_redis
    llm_cache_module._redis_client = original_redis_client
    if hasattr(llm_cache_module, '_mock_redis_store'):
        delattr(llm_cache_module, '_mock_redis_store')
    if hasattr(llm_cache_module, '_mock_redis'):
        delattr(llm_cache_module, '_mock_redis')
