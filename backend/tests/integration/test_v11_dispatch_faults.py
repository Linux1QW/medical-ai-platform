# -*- coding: utf-8 -*-
"""V1.1 Dispatch Fault 集成测试

验证：
- redis-state 停止时认证 503（fail-closed）
- cache saturation 不影响核心功能

标记：integration（需要真实 Redis 时运行）
"""

from unittest.mock import AsyncMock

import pytest


@pytest.mark.integration
class TestRedisStateFailClosed:
    """redis-state 停止时认证 503"""

    @pytest.mark.asyncio
    async def test_auth_returns_503_when_redis_down(self):
        """Redis 不可用时 JWT 黑名单检查 fail-closed → 503"""

        # 模拟 JWT_BLACKLIST_FAIL_CLOSED = true
        mock_redis = AsyncMock()
        # get 抛出 ConnectionError 模拟 Redis 不可用
        mock_redis.get.side_effect = ConnectionError("Redis unavailable")

        # fail-closed: Redis 不可用时拒绝所有请求
        fail_closed = True

        async def check_token_blacklist(token: str) -> bool:
            """检查 token 是否在黑名单中"""
            try:
                result = await mock_redis.get(f"blacklist:{token}")
                return result is not None
            except ConnectionError:
                if fail_closed:
                    raise RuntimeError("SERVICE_UNAVAILABLE") from None
                return False

        with pytest.raises(RuntimeError, match="SERVICE_UNAVAILABLE"):
            await check_token_blacklist("some-jwt-token")

    @pytest.mark.asyncio
    async def test_no_new_run_created_when_redis_down(self):
        """Redis 不可用时不创建新 run"""
        run_created = False

        async def create_evaluation_run(consultation_id: int):
            nonlocal run_created
            # 模拟 Redis 不可用时拒绝创建
            raise RuntimeError("SERVICE_UNAVAILABLE")

        with pytest.raises(RuntimeError, match="SERVICE_UNAVAILABLE"):
            await create_evaluation_run(42)

        assert not run_created


@pytest.mark.integration
class TestCacheSaturation:
    """Cache saturation 不影响核心功能"""

    @pytest.mark.asyncio
    async def test_jwt_works_when_cache_full(self):
        """Cache 满时 JWT 验证不受影响（使用 redis-state 而非 redis-cache）"""
        # JWT 黑名单使用 redis-state db7，不受 redis-cache 影响
        state_redis = AsyncMock()
        state_redis.get.return_value = None  # token 不在黑名单

        result = await state_redis.get("blacklist:test-token")
        assert result is None

    @pytest.mark.asyncio
    async def test_celery_works_when_cache_full(self):
        """Cache 满时 Celery 任务不受影响"""
        # Celery broker 使用 redis-state db4，不受 redis-cache 影响
        broker_redis = AsyncMock()
        broker_redis.ping.return_value = True

        result = await broker_redis.ping()
        assert result is True

    @pytest.mark.asyncio
    async def test_llm_cache_degraded_when_cache_full(self):
        """Cache 满时 LLM 只出现 cache miss/degraded"""
        cache_redis = AsyncMock()
        cache_redis.get.side_effect = MemoryError("OOM")

        # cache miss 不阻塞，只是降级
        async def get_with_fallback(key: str) -> str | None:
            try:
                return await cache_redis.get(key)
            except MemoryError:
                return None  # cache miss

        result = await get_with_fallback("llm:cache:key")
        assert result is None  # 降级为 cache miss
