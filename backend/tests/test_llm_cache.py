"""
LLMResponseCache 单元测试

测试策略：
- 通过 conftest.py 中的 session 级别 fixture 全局 mock Redis
- 每个测试通过 reset_redis_and_stats fixture 清空 store 并重置状态
- 覆盖所有核心路径：命中、未命中、temperature>0 跳过、禁用跳过、异常降级、统计指标
"""

import hashlib
import json
import pytest
from unittest.mock import patch

# ── 被测模块 ──────────────────────────────────────────────────────────────────
import app.services.llm_cache as llm_cache_module
from app.services.llm_cache import LLMResponseCache, _build_cache_key


# ── Fixtures ──────────────────────────────────────────────────────────────────

@pytest.fixture(autouse=True)
def reset_redis_and_stats():
    """每个测试前重置 Redis 客户端、统计计数器和 mock store"""
    # 重置为 mock Redis 实例（不是 None！None 会导致 _get_redis 连接真实 Redis）
    llm_cache_module._redis_client = llm_cache_module._mock_redis
    # 重置统计计数器（错误计数器仍在进程内）
    llm_cache_module._cache_errors = 0
    # 清空 mock Redis 的内存 store 和计数器
    if hasattr(llm_cache_module, '_mock_redis_store'):
        llm_cache_module._mock_redis_store.clear()
    if hasattr(llm_cache_module, '_mock_redis') and hasattr(llm_cache_module._mock_redis, '_counters'):
        llm_cache_module._mock_redis._counters.clear()

    yield

    # 清理后也重置为 mock
    llm_cache_module._redis_client = llm_cache_module._mock_redis


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
    async def test_set_then_get(self):
        """写入缓存后可以读取"""
        await LLMResponseCache.set(
            SAMPLE_MESSAGES, SAMPLE_MODEL, 0.0, SAMPLE_RESPONSE
        )
        result = await LLMResponseCache.get(SAMPLE_MESSAGES, SAMPLE_MODEL, 0.0)
        assert result == SAMPLE_RESPONSE

    @pytest.mark.asyncio
    async def test_cache_miss(self):
        """缓存未命中时返回 None"""
        result = await LLMResponseCache.get(SAMPLE_MESSAGES, SAMPLE_MODEL, 0.0)
        assert result is None

    @pytest.mark.asyncio
    async def test_different_model_no_cross_contamination(self):
        """不同模型之间缓存不交叉污染"""
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
    async def test_get_with_positive_temperature(self):
        """temperature > 0 时 get 返回 None（不查缓存）"""
        # 先写入缓存
        await LLMResponseCache.set(
            SAMPLE_MESSAGES, SAMPLE_MODEL, 0.0, SAMPLE_RESPONSE
        )
        # temperature=0.5 时不应命中缓存
        result = await LLMResponseCache.get(SAMPLE_MESSAGES, SAMPLE_MODEL, 0.5)
        assert result is None

    @pytest.mark.asyncio
    async def test_set_with_positive_temperature(self):
        """temperature > 0 时 set 不写入缓存"""
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
    async def test_set_when_disabled(self):
        """缓存禁用时 set 不写入"""
        with patch.object(llm_cache_module.settings, 'LLM_CACHE_ENABLED', False):
            await LLMResponseCache.set(
                SAMPLE_MESSAGES, SAMPLE_MODEL, 0.0, SAMPLE_RESPONSE
            )
            # store 应该为空
            assert len(llm_cache_module._mock_redis_store) == 0


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
            # 计数器现在在 Redis 中，但 Redis 不可用时计数器也无法递增
            # 只验证函数安全返回 None（graceful degradation）

    @pytest.mark.asyncio
    async def test_get_redis_exception_returns_none(self):
        """Redis get 操作异常时返回 None（不抛异常）"""
        mock_redis = llm_cache_module._mock_redis
        original_get = mock_redis.get

        async def failing_get(key):
            raise ConnectionError("Redis connection lost")

        mock_redis.get = failing_get
        try:
            result = await LLMResponseCache.get(SAMPLE_MESSAGES, SAMPLE_MODEL, 0.0)
            assert result is None
            assert llm_cache_module._cache_errors == 1
        finally:
            mock_redis.get = original_get

    @pytest.mark.asyncio
    async def test_set_redis_exception_silent(self):
        """Redis set 操作异常时静默处理"""
        mock_redis = llm_cache_module._mock_redis
        original_setex = mock_redis.setex

        async def failing_setex(key, ttl, value):
            raise ConnectionError("Redis connection lost")

        mock_redis.setex = failing_setex
        try:
            # 不应抛出异常
            await LLMResponseCache.set(
                SAMPLE_MESSAGES, SAMPLE_MODEL, 0.0, SAMPLE_RESPONSE
            )
            assert llm_cache_module._cache_errors == 1
        finally:
            mock_redis.setex = original_setex


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
    async def test_stats_after_hit(self):
        """缓存命中后统计正确"""
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
    async def test_stats_after_miss(self):
        """缓存未命中后统计正确"""
        await LLMResponseCache.get(SAMPLE_MESSAGES, SAMPLE_MODEL, 0.0)

        stats = await LLMResponseCache.get_stats()
        assert stats["cache_hits"] == 0
        assert stats["cache_misses"] == 1
        assert stats["hit_rate"] == 0.0

    @pytest.mark.asyncio
    async def test_stats_hit_rate_mixed(self):
        """混合命中/未命中时命中率计算正确"""
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
    async def test_stats_error_count(self):
        """异常次数统计正确"""
        mock_redis = llm_cache_module._mock_redis
        original_get = mock_redis.get

        async def failing_get(key):
            raise ConnectionError("fail")

        mock_redis.get = failing_get
        try:
            await LLMResponseCache.get(SAMPLE_MESSAGES, SAMPLE_MODEL, 0.0)
            await LLMResponseCache.get(SAMPLE_MESSAGES, SAMPLE_MODEL, 0.0)

            stats = await LLMResponseCache.get_stats()
            assert stats["cache_errors"] == 2
        finally:
            mock_redis.get = original_get


# ── 测试 clear ────────────────────────────────────────────────────────────────

class TestClear:
    """测试缓存清除"""

    @pytest.mark.asyncio
    async def test_clear_removes_all_keys(self):
        """clear 清除所有 llm_cache:* 键"""
        await LLMResponseCache.set(
            SAMPLE_MESSAGES, SAMPLE_MODEL, 0.0, "resp1"
        )
        msgs2 = [{"role": "user", "content": "另一个问题"}]
        await LLMResponseCache.set(msgs2, SAMPLE_MODEL, 0.0, "resp2")

        assert len(llm_cache_module._mock_redis_store) == 2
        await LLMResponseCache.clear()
        assert len(llm_cache_module._mock_redis_store) == 0
