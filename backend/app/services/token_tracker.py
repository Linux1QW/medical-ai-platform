"""
Token 用量追踪器 — 基于 Redis 的按日/按模型 Token 统计

Key 格式：
  - token_usage:{model}:{date}   某模型某天的用量
  - token_usage:daily:{date}     当天全局总用量
"""

import logging
from datetime import date, timedelta
from typing import Optional

import redis.asyncio as aioredis

from app.core.config import settings

logger = logging.getLogger(__name__)

# ── Redis 连接（懒初始化，复用 llm_cache 同一 Redis 实例 db=2）───────────────

_redis_client: Optional[aioredis.Redis] = None


async def _get_redis() -> Optional[aioredis.Redis]:
    """获取 Redis 客户端（与 llm_cache 同实例 db=2）"""
    global _redis_client
    if _redis_client is not None:
        return _redis_client

    try:
        redis_url = settings.REDIS_CHECKPOINT_URL
        if "/1" in redis_url:
            redis_url = redis_url.replace("/1", "/2")
        elif redis_url.endswith("redis://localhost:6379"):
            redis_url = redis_url + "/2"
        _redis_client = aioredis.from_url(
            redis_url,
            decode_responses=True,
            socket_connect_timeout=3.0,
            socket_timeout=3.0,
            retry_on_timeout=False,
        )
        await _redis_client.ping()
        logger.info("TokenTracker Redis 连接成功")
    except Exception as e:
        logger.warning(f"TokenTracker Redis 连接失败，Token 统计不可用: {e}")
        _redis_client = None
    return _redis_client


def _today_str() -> str:
    return date.today().isoformat()


class TokenTracker:
    """Token 用量追踪器"""

    async def record_usage(
        self, model: str, prompt_tokens: int, completion_tokens: int
    ) -> None:
        """记录单次调用的 Token 用量（best-effort，异常静默）"""
        redis = await _get_redis()
        if redis is None:
            return

        total = prompt_tokens + completion_tokens
        today = _today_str()

        try:
            # 按模型+日期
            model_key = f"token_usage:{model}:{today}"
            await redis.hincrby(model_key, "prompt_tokens", prompt_tokens)
            await redis.hincrby(model_key, "completion_tokens", completion_tokens)
            await redis.hincrby(model_key, "total_tokens", total)
            await redis.expire(model_key, 86400 * 7)  # 保留 7 天

            # 全局日汇总
            daily_key = f"token_usage:daily:{today}"
            await redis.hincrby(daily_key, "prompt_tokens", prompt_tokens)
            await redis.hincrby(daily_key, "completion_tokens", completion_tokens)
            await redis.hincrby(daily_key, "total_tokens", total)
            await redis.expire(daily_key, 86400 * 7)
        except Exception as e:
            logger.debug(f"Token 用量记录异常: {e}")

    async def get_daily_usage(self, date_str: str = None) -> dict:
        """获取某天的 Token 用量统计"""
        redis = await _get_redis()
        if redis is None:
            return {"error": "Redis 不可用"}

        target = date_str or _today_str()
        key = f"token_usage:daily:{target}"
        try:
            data = await redis.hgetall(key)
            if not data:
                return {"date": target, "prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}
            return {
                "date": target,
                "prompt_tokens": int(data.get("prompt_tokens", 0)),
                "completion_tokens": int(data.get("completion_tokens", 0)),
                "total_tokens": int(data.get("total_tokens", 0)),
            }
        except Exception as e:
            logger.debug(f"获取日用量异常: {e}")
            return {"error": str(e)}

    async def get_model_usage(self, model: str, days: int = 7) -> list:
        """获取某模型最近 N 天的用量趋势"""
        redis = await _get_redis()
        if redis is None:
            return []

        result = []
        today = date.today()
        for i in range(days):
            d = (today - timedelta(days=i)).isoformat()
            key = f"token_usage:{model}:{d}"
            try:
                data = await redis.hgetall(key)
                if data:
                    result.append({
                        "date": d,
                        "prompt_tokens": int(data.get("prompt_tokens", 0)),
                        "completion_tokens": int(data.get("completion_tokens", 0)),
                        "total_tokens": int(data.get("total_tokens", 0)),
                    })
                else:
                    result.append({"date": d, "prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0})
            except Exception as e:
                logger.debug(f"获取模型用量异常: {e}")
        return result

    async def check_budget(self, daily_limit: float = None) -> dict:
        """检查是否超出每日预算"""
        limit = daily_limit or settings.TOKEN_DAILY_LIMIT
        usage = await self.get_daily_usage()
        total = usage.get("total_tokens", 0)
        estimated_cost = round(total / 1000 * settings.COST_PER_1K_TOKENS, 4)
        return {
            "daily_limit": limit,
            "used_tokens": total,
            "remaining_tokens": max(0, limit - total),
            "estimated_cost": estimated_cost,
            "budget_exceeded": total > limit,
        }

    async def get_summary(self) -> dict:
        """获取 Token 用量摘要（供 /health 端点使用）"""
        daily = await self.get_daily_usage()
        budget = await self.check_budget()
        return {
            "today": daily,
            "budget": budget,
        }


# 全局单例
token_tracker = TokenTracker()
