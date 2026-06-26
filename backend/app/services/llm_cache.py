"""
LLM 响应缓存层 — 基于 Redis 的精确哈希缓存

仅对 temperature=0 的调用启用缓存（temperature>0 具有随机性，不适合缓存）。
缓存键格式：llm_cache:{model}:{temperature}:{sha256_hash}
任何缓存异常均静默处理，不影响正常 LLM 调用（best-effort 降级）。
"""

import hashlib
import json
import logging
from typing import Optional, List, Dict

import redis.asyncio as aioredis

from app.core.config import settings

logger = logging.getLogger(__name__)

# ── Redis 连接（懒初始化，与 checkpointer 使用同一实例但 db 不同）──────────────

_redis_client: Optional[aioredis.Redis] = None

# ── 统计指标（进程内计数器，重启归零）─────────────────────────────────────────

_cache_hits: int = 0
_cache_misses: int = 0
_cache_errors: int = 0


async def _get_redis() -> Optional[aioredis.Redis]:
    """获取 Redis 客户端实例（懒初始化，best-effort：连接失败返回 None）"""
    global _redis_client
    if _redis_client is not None:
        return _redis_client
    try:
        # 复用 REDIS_CHECKPOINT_URL 指向的 Redis 实例，使用 db=2 避免与 checkpointer(db=1) 冲突
        redis_url = settings.REDIS_CHECKPOINT_URL
        # 替换 db 编号为 2（保留 db=1 给 checkpointer）
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
        # 验证连接
        await _redis_client.ping()
        logger.info(f"LLM 缓存 Redis 连接已建立: {redis_url}")
        return _redis_client
    except Exception as e:
        logger.warning(f"LLM 缓存 Redis 连接失败，缓存功能已禁用: {e}")
        _redis_client = None
        return None


def _build_cache_key(
    messages: List[Dict[str, str]],
    model: str,
    temperature: float,
) -> str:
    """生成缓存键：llm_cache:{model}:{temperature}:{sha256_hash}"""
    # 对 messages 做确定性 JSON 序列化（排序 key，ensure_ascii=False 保留中文）
    messages_json = json.dumps(messages, sort_keys=True, ensure_ascii=False)
    hash_digest = hashlib.sha256(messages_json.encode("utf-8")).hexdigest()[:16]
    return f"llm_cache:{model}:{temperature}:{hash_digest}"


class LLMResponseCache:
    """
    LLM 响应缓存（Redis 后端）

    设计原则：
    - best-effort：所有 Redis 异常均被捕获并记录日志，不向上层传播
    - 仅缓存 temperature=0 的调用（确定性输出）
    - 缓存键包含 model + temperature，避免跨模型/温度污染
    """

    @staticmethod
    async def get(
        messages: List[Dict[str, str]],
        model: str,
        temperature: float,
    ) -> Optional[str]:
        """
        查询缓存

        Returns:
            缓存的 LLM 响应文本，未命中或异常时返回 None
        """
        global _cache_hits, _cache_misses, _cache_errors

        # temperature > 0 不缓存
        if temperature > 0:
            _cache_misses += 1
            return None

        cache_key = _build_cache_key(messages, model, temperature)

        try:
            r = await _get_redis()
            if r is None:
                _cache_misses += 1
                return None

            value = await r.get(cache_key)
            if value is not None:
                _cache_hits += 1
                logger.debug(f"LLM 缓存命中: key={cache_key}")
                return value
            else:
                _cache_misses += 1
                logger.debug(f"LLM 缓存未命中: key={cache_key}")
                return None

        except Exception as e:
            _cache_errors += 1
            logger.warning(f"LLM 缓存读取异常（静默降级）: {e}")
            return None

    @staticmethod
    async def set(
        messages: List[Dict[str, str]],
        model: str,
        temperature: float,
        response: str,
        ttl: Optional[int] = None,
    ) -> None:
        """
        写入缓存（best-effort，异常静默）

        Args:
            messages: OpenAI 格式消息列表
            model: 模型名称
            temperature: 采样温度（>0 时不写入）
            response: LLM 响应文本
            ttl: 过期时间（秒），默认使用 LLM_CACHE_TTL
        """
        global _cache_errors

        # temperature > 0 不缓存
        if temperature > 0:
            return

        if not settings.LLM_CACHE_ENABLED:
            return

        cache_key = _build_cache_key(messages, model, temperature)
        _ttl = ttl if ttl is not None else settings.LLM_CACHE_TTL

        try:
            r = await _get_redis()
            if r is None:
                return

            await r.setex(cache_key, _ttl, response)
            logger.debug(f"LLM 缓存已写入: key={cache_key}, ttl={_ttl}s")

            # 维护缓存大小（近似 LRU：超过 max_size 时随机淘汰旧键）
            # 使用 SCAN + TTL 检查，避免阻塞 Redis
            await _enforce_max_size(r)

        except Exception as e:
            _cache_errors += 1
            logger.warning(f"LLM 缓存写入异常（静默降级）: {e}")

    @staticmethod
    async def clear() -> None:
        """清除所有 LLM 缓存键（best-effort）"""
        global _cache_errors
        try:
            r = await _get_redis()
            if r is None:
                return

            # 使用 SCAN 避免 KEYS 命令阻塞 Redis
            cursor = 0
            while True:
                cursor, keys = await r.scan(
                    cursor=cursor, match="llm_cache:*", count=100
                )
                if keys:
                    await r.delete(*keys)
                if cursor == 0:
                    break

            logger.info("LLM 缓存已全部清除")

        except Exception as e:
            _cache_errors += 1
            logger.warning(f"LLM 缓存清除异常（静默降级）: {e}")

    @staticmethod
    async def get_stats() -> dict:
        """
        获取缓存统计指标

        Returns:
            包含 cache_hits, cache_misses, cache_errors, hit_rate, cache_size 的字典
        """
        total = _cache_hits + _cache_misses
        hit_rate = (_cache_hits / total * 100) if total > 0 else 0.0

        # 异步获取当前缓存条目数（近似值）
        cache_size = 0
        try:
            r = await _get_redis()
            if r is not None:
                cursor = 0
                count = 0
                while True:
                    cursor, keys = await r.scan(
                        cursor=cursor, match="llm_cache:*", count=100
                    )
                    count += len(keys)
                    if cursor == 0:
                        break
                cache_size = count
        except Exception:
            pass

        return {
            "cache_hits": _cache_hits,
            "cache_misses": _cache_misses,
            "cache_errors": _cache_errors,
            "hit_rate": round(hit_rate, 2),
            "cache_size": cache_size,
            "enabled": settings.LLM_CACHE_ENABLED,
        }


async def _enforce_max_size(r: aioredis.Redis) -> None:
    """
    近似强制最大缓存条目数（best-effort）

    当缓存条目超过 LLM_CACHE_MAX_SIZE 时，
    使用 SCAN 找到 TTL 最小的键并删除，直到低于阈值。
    为避免每次写入都扫描，仅当概率性触发时执行（约 1% 概率）。
    """
    import random

    # 仅 1% 概率执行清理，避免性能开销
    if random.random() > 0.01:
        return

    try:
        cursor = 0
        all_keys = []
        while True:
            cursor, keys = await r.scan(cursor=cursor, match="llm_cache:*", count=200)
            all_keys.extend(keys)
            if cursor == 0:
                break

        if len(all_keys) <= settings.LLM_CACHE_MAX_SIZE:
            return

        # 超出上限，删除最旧（TTL 最小）的键
        excess = len(all_keys) - settings.LLM_CACHE_MAX_SIZE
        # 获取所有键的 TTL，按 TTL 升序排序（TTL 小 = 即将过期 = 最旧）
        keys_with_ttl = []
        for key in all_keys:
            ttl = await r.ttl(key)
            if ttl >= 0:
                keys_with_ttl.append((ttl, key))

        keys_with_ttl.sort(key=lambda x: x[0])
        to_delete = [k for _, k in keys_with_ttl[:excess]]
        if to_delete:
            await r.delete(*to_delete)
            logger.info(f"LLM 缓存清理: 删除 {len(to_delete)} 条过期/旧缓存")

    except Exception as e:
        logger.warning(f"LLM 缓存大小维护异常（静默降级）: {e}")


async def close_cache_redis() -> None:
    """关闭缓存 Redis 连接（在 FastAPI shutdown 中调用）"""
    global _redis_client
    if _redis_client is not None:
        await _redis_client.close()
        _redis_client = None
        logger.info("LLM 缓存 Redis 连接已关闭")
