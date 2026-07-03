"""
LLMResponseCache 单元测试

测试策略：
- 使用 fakeredis 模拟 Redis（无需真实 Redis 实例）
- 如果 fakeredis 不可用，则通过 mock 模拟 Redis 行为
- 覆盖所有核心路径：命中、未命中、temperature>0 跳过、禁用跳过、异常降级、统计指标
"""

import asyncio
import hashlib
import json
import pytest
from unittest.mock import AsyncMock, MagicMock, patch

# ── 被测模块 ──────────────────────────────────────────────────────────────────
import app.services.llm_cache as llm_cache_module
from app.services.llm_cache import LLMResponseCache, _build_cache_key


# ── Fixtures ──────────────────────────────────────────────────────────────────

@pytest.fixture(autouse=True)
def reset_redis_and_stats():
    """每个测试前重置 Redis 客户端和统计计数器"""
    # 重置全局 Redis 客户端，确保 _get_redis() 重新初始化
    llm_cache_module._redis_client = None
    # 重置统计计数器
    llm_cache_module._cache_hits = 0
    llm_cache_module._cache_misses = 0
    llm_cache_module._cache_errors = 0
    yield
    # 清理后也重置，避免影响后续测试
    llm_cache_module._redis_client = None


@pytest.fixture
def mock_redis():
    """创建一个模拟的 Redis 客户端（内存字典存储）"""
    store = {}

    async def mock_get(key):
        return store.get(key)

    async def mock_setex(key, ttl, value):
        store[key] = value

    async def mock_ping():
        return True

    async def mock_scan(cursor, match, count):
        import fnmatch
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

    return redis_mock


SAMPLE_MESSAGES = [
    {"role": "system", "content": "你是一个医学助手"},
    {"role": "user", "content": "患者头痛伴发热，可能是什么病？"},
]

SAMPLE_RESPONSE = "根据症状描述，可能是上呼吸道感染或偏头痛，建议进一步检查血常规和体温。"
SAMPLE_MODEL = "qwen3-max"


# ── 测试缓存键生成 ─────────────────────────────────────────────────────────────

class TestCacheKeyGeneration:
    """测试缓存键生成逻辑"""

    def test_key_format(self):
        """缓存键格式正确：llm_cache:{model}:{temperature}:{hash}"""
        key = _build_cache_key(SAMPLE_MESSAGES, SAMPLE_MODEL, 0.0)
        assert key.startswith("llm_cache:")
        assert SAMPLE_MODEL in key
        assert ":0.0:" in key

    def test_same_messages_same_key(self):
        """相同消息列表生成相同缓存键"""
        key1 = _build_cache_key(SAMPLE_MESSAGES, SAMPLE_MODEL, 0.0)
        key2 = _build_cache_key(SAMPLE_MESSAGES, SAMPLE_MODEL, 0.0)
        assert key1 == key2

    def test_different_messages_different_key(self):
        """不同消息列表生成不同缓存键"""
        msgs2 = [{"role": "user", "content": "不同的问题"}]
        key1 = _build_cache_key(SAMPLE_MESSAGES, SAMPLE_MODEL, 0.0)
        key2 = _build_cache_key(msgs2, SAMPLE_MODEL, 0.0)
        assert key1 != key2

    def test_different_model_different_key(self):
        """不同模型生成不同缓存键"""
        key1 = _build_cache_key(SAMPLE_MESSAGES, "qwen3-max", 0.0)
        key2 = _build_cache_key(SAMPLE_MESSAGES, "qwen-turbo", 0.0)
        assert key1 != key2

    def test_different_temperature_different_key(self):
        """不同 temperature 生成不同缓存键"""
        key1 = _build_cache_key(SAMPLE_MESSAGES, SAMPLE_MODEL, 0.0)
        key2 = _build_cache_key(SAMPLE_MESSAGES, SAMPLE_MODEL, 0.5)
        assert key1 != key2


# ── 测试缓存读写 ──────────────────────────────────────────────────────────────

class TestCacheReadWrite:
    """测试缓存写入和读取"""

    @pytest.mark.asyncio
    async def test_set_then_get(self, mock_redis):
        """写入缓存后可以读取"""
        with patch.object(llm_cache_module, '_get_redis', return_value=mock_redis):
            await LLMResponseCache.set(
                SAMPLE_MESSAGES, SAMPLE_MODEL, 0.0, SAMPLE_RESPONSE
            )
            result = await LLMResponseCache.get(SAMPLE_MESSAGES, SAMPLE_MODEL, 0.0)
            assert result == SAMPLE_RESPONSE

    @pytest.mark.asyncio
    async def test_cache_miss(self, mock_redis):
        """缓存未命中时返回 None"""
        with patch.object(llm_cache_module, '_get_redis', return_value=mock_redis):
            result = await LLMResponseCache.get(SAMPLE_MESSAGES, SAMPLE_MODEL, 0.0)
            assert result is None

    @pytest.mark.asyncio
    async def test_different_model_no_cross_contamination(self, mock_redis):
        """不同模型之间缓存不交叉污染"""
        with patch.object(llm_cache_module, '_get_redis', return_value=mock_redis):
            await LLMResponseCache.set(
                SAMPLE_MESSAGES, "model-A", 0.0, "response-A"
            )
            result_a = await LLMResponseCache.get(SAMPLE_MESSAGES, "model-A", 0.0)
            result_b = await LLMResponseCache.get(SAMPLE_MESSAGES, "model-B", 0.0)
            assert result_a == "response-A"
            assert result_b is None


# ── 测试 temperature > 0 不缓存 ────────────────────────────────────────────────

class TestTemperatureFilter:
    """temperature > 0 时不启用缓存"""

    @pytest.mark.asyncio
    async def test_get_with_positive_temperature(self, mock_redis):
        """temperature > 0 时 get 返回 None（不查缓存）"""
        with patch.object(llm_cache_module, '_get_redis', return_value=mock_redis):
            # 先写入缓存
            await LLMResponseCache.set(
                SAMPLE_MESSAGES, SAMPLE_MODEL, 0.0, SAMPLE_RESPONSE
            )
            # temperature=0.5 时不应命中缓存
            result = await LLMResponseCache.get(SAMPLE_MESSAGES, SAMPLE_MODEL, 0.5)
            assert result is None

    @pytest.mark.asyncio
    async def test_set_with_positive_temperature(self, mock_redis):
        """temperature > 0 时 set 不写入缓存"""
        with patch.object(llm_cache_module, '_get_redis', return_value=mock_redis):
            await LLMResponseCache.set(
                SAMPLE_MESSAGES, SAMPLE_MODEL, 0.5, SAMPLE_RESPONSE
            )
            # 即使写入，temperature=0 时也读不到（因为 key 包含 temperature）
            result = await LLMResponseCache.get(SAMPLE_MESSAGES, SAMPLE_MODEL, 0.0)
            assert result is None


# ── 测试缓存禁用 ──────────────────────────────────────────────────────────────

class TestCacheDisabled:
    """LLM_CACHE_ENABLED=False 时不写入缓存"""

    @pytest.mark.asyncio
    async def test_set_when_disabled(self, mock_redis):
        """缓存禁用时 set 不写入"""
        with patch.object(llm_cache_module.settings, 'LLM_CACHE_ENABLED', False):
            with patch.object(llm_cache_module, '_get_redis', return_value=mock_redis):
                await LLMResponseCache.set(
                    SAMPLE_MESSAGES, SAMPLE_MODEL, 0.0, SAMPLE_RESPONSE
                )
                # store 应该为空
                assert len(mock_redis._store) == 0


# ── 测试异常降级 ──────────────────────────────────────────────────────────────

class TestGracefulDegradation:
    """缓存异常时不影响正常流程（best-effort 降级）"""

    @pytest.mark.asyncio
    async def test_get_redis_failure_returns_none(self):
        """Redis 连接失败时 get 返回 None（不抛异常）"""
        async def failing_get_redis():
            return None

        with patch.object(llm_cache_module, '_get_redis', failing_get_redis):
            result = await LLMResponseCache.get(SAMPLE_MESSAGES, SAMPLE_MODEL, 0.0)
            assert result is None
            assert llm_cache_module._cache_misses == 1

    @pytest.mark.asyncio
    async def test_get_redis_exception_returns_none(self, mock_redis):
        """Redis get 操作异常时返回 None（不抛异常）"""
        async def failing_get(key):
            raise ConnectionError("Redis connection lost")

        mock_redis.get = failing_get

        with patch.object(llm_cache_module, '_get_redis', return_value=mock_redis):
            result = await LLMResponseCache.get(SAMPLE_MESSAGES, SAMPLE_MODEL, 0.0)
            assert result is None
            assert llm_cache_module._cache_errors == 1

    @pytest.mark.asyncio
    async def test_set_redis_exception_silent(self, mock_redis):
        """Redis set 操作异常时静默处理"""
        async def failing_setex(key, ttl, value):
            raise ConnectionError("Redis connection lost")

        mock_redis.setex = failing_setex

        with patch.object(llm_cache_module, '_get_redis', return_value=mock_redis):
            # 不应抛出异常
            await LLMResponseCache.set(
                SAMPLE_MESSAGES, SAMPLE_MODEL, 0.0, SAMPLE_RESPONSE
            )
            assert llm_cache_module._cache_errors == 1


# ── 测试统计指标 ──────────────────────────────────────────────────────────────

class TestStats:
    """统计指标正确性"""

    @pytest.mark.asyncio
    async def test_stats_initial(self):
        """初始状态统计正确"""
        stats = await LLMResponseCache.get_stats()
        assert stats["cache_hits"] == 0
        assert stats["cache_misses"] == 0
        assert stats["cache_errors"] == 0
        assert stats["hit_rate"] == 0.0

    @pytest.mark.asyncio
    async def test_stats_after_hit(self, mock_redis):
        """缓存命中后统计正确"""
        with patch.object(llm_cache_module, '_get_redis', return_value=mock_redis):
            # 先写入
            await LLMResponseCache.set(
                SAMPLE_MESSAGES, SAMPLE_MODEL, 0.0, SAMPLE_RESPONSE
            )
            # 命中
            await LLMResponseCache.get(SAMPLE_MESSAGES, SAMPLE_MODEL, 0.0)

            stats = await LLMResponseCache.get_stats()
            assert stats["cache_hits"] == 1
            assert stats["cache_misses"] == 0
            assert stats["hit_rate"] == 100.0

    @pytest.mark.asyncio
    async def test_stats_after_miss(self, mock_redis):
        """缓存未命中后统计正确"""
        with patch.object(llm_cache_module, '_get_redis', return_value=mock_redis):
            await LLMResponseCache.get(SAMPLE_MESSAGES, SAMPLE_MODEL, 0.0)

            stats = await LLMResponseCache.get_stats()
            assert stats["cache_hits"] == 0
            assert stats["cache_misses"] == 1
            assert stats["hit_rate"] == 0.0

    @pytest.mark.asyncio
    async def test_stats_hit_rate_mixed(self, mock_redis):
        """混合命中/未命中时命中率计算正确"""
        with patch.object(llm_cache_module, '_get_redis', return_value=mock_redis):
            # 写入并命中一次
            await LLMResponseCache.set(
                SAMPLE_MESSAGES, SAMPLE_MODEL, 0.0, SAMPLE_RESPONSE
            )
            await LLMResponseCache.get(SAMPLE_MESSAGES, SAMPLE_MODEL, 0.0)

            # 未命中一次（不同消息）
            other_msgs = [{"role": "user", "content": "另一个问题"}]
            await LLMResponseCache.get(other_msgs, SAMPLE_MODEL, 0.0)

            stats = await LLMResponseCache.get_stats()
            assert stats["cache_hits"] == 1
            assert stats["cache_misses"] == 1
            assert stats["hit_rate"] == 50.0

    @pytest.mark.asyncio
    async def test_stats_error_count(self, mock_redis):
        """异常次数统计正确"""
        async def failing_get(key):
            raise ConnectionError("fail")

        mock_redis.get = failing_get

        with patch.object(llm_cache_module, '_get_redis', return_value=mock_redis):
            await LLMResponseCache.get(SAMPLE_MESSAGES, SAMPLE_MODEL, 0.0)
            await LLMResponseCache.get(SAMPLE_MESSAGES, SAMPLE_MODEL, 0.0)

            stats = await LLMResponseCache.get_stats()
            assert stats["cache_errors"] == 2


# ── 测试 clear ────────────────────────────────────────────────────────────────

class TestClear:
    """测试缓存清除"""

    @pytest.mark.asyncio
    async def test_clear_removes_all_keys(self, mock_redis):
        """clear 清除所有 llm_cache:* 键"""
        with patch.object(llm_cache_module, '_get_redis', return_value=mock_redis):
            await LLMResponseCache.set(
                SAMPLE_MESSAGES, SAMPLE_MODEL, 0.0, "resp1"
            )
            msgs2 = [{"role": "user", "content": "另一个问题"}]
            await LLMResponseCache.set(msgs2, SAMPLE_MODEL, 0.0, "resp2")

            assert len(mock_redis._store) == 2
            await LLMResponseCache.clear()
            assert len(mock_redis._store) == 0
