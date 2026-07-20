# -*- coding: utf-8 -*-
"""Retrieval Bundle Cache — 基于 Redis 的检索结果缓存

缓存 tiered_retrieve 的 RetrievalBundle 结果，
相同查询在 TTL 内直接返回缓存结果，避免重复检索和 LLM 调用。

设计要点：
- 使用 Redis db=3，与 LLM 缓存 (db=2) 和 Checkpointer (db=1) 隔离
- 缓存键包含 index_version，索引重建后自动失效
- 超过 RETRIEVAL_CACHE_MAX_SIZE 时概率性清理最旧条目
- 所有 Redis 操作 try/except 包裹，缓存失败不影响正常检索
"""

import hashlib
import json
import logging
import random
from typing import Optional

import redis.asyncio as aioredis

from app.core.config import settings

logger = logging.getLogger(__name__)

# ── Redis 连接（懒初始化，db=3 独立隔离）──────────────────────────────────────

_redis_client: Optional[aioredis.Redis] = None

# ── 统计指标（Redis 持久化计数器，重启不丢失）─────────────────────────────────────

# Redis Key 常量
REDIS_KEY_HITS = "rag_cache:hits"
REDIS_KEY_MISSES = "rag_cache:misses"

# 进程内错误计数器（不持久化）
_cache_errors: int = 0

CACHE_KEY_PREFIX = "retrieval_cache"


async def _incr_counter(key: str) -> None:
    """原子递增 Redis 计数器（best-effort，失败静默）"""
    try:
        r = await _get_redis()
        if r is not None:
            await r.incr(key)
    except Exception:
        pass


async def _get_redis() -> Optional[aioredis.Redis]:
    """获取 Redis 客户端实例（懒初始化，db=3，best-effort：连接失败返回 None）"""
    global _redis_client
    if _redis_client is not None:
        return _redis_client

    try:
        # 复用 REDIS_CHECKPOINT_URL 的 Redis 实例，使用 db=3 隔离
        redis_url = settings.REDIS_CHECKPOINT_URL
        # 替换 db 编号为 3（保留 db=1 给 checkpointer，db=2 给 LLM 缓存）
        if "/1" in redis_url:
            redis_url = redis_url.replace("/1", "/3")
        elif "/2" in redis_url:
            redis_url = redis_url.replace("/2", "/3")
        elif redis_url.endswith("redis://localhost:6379"):
            redis_url = redis_url + "/3"

        _redis_client = aioredis.from_url(
            redis_url,
            decode_responses=True,
            socket_connect_timeout=3.0,
            socket_timeout=3.0,
            retry_on_timeout=False,
        )
        # 验证连接
        await _redis_client.ping()
        logger.info(f"检索缓存 Redis 连接已建立: {redis_url}")
        return _redis_client
    except Exception as e:
        logger.warning(f"检索缓存 Redis 连接失败，缓存功能已禁用: {e}")
        _redis_client = None
        return None


def _build_cache_key(queries_text: str, index_version: str) -> str:
    """构建缓存键

    Args:
        queries_text: 所有查询文本的拼接
        index_version: 当前索引版本

    Returns:
        缓存键字符串
    """
    normalized = queries_text.strip().lower()
    content_hash = hashlib.sha256(normalized.encode("utf-8")).hexdigest()[:16]
    return f"{CACHE_KEY_PREFIX}:{index_version}:{content_hash}"


async def get_cached_bundle(queries_text: str, index_version: str) -> Optional[dict]:
    """从缓存获取 RetrievalBundle

    Returns:
        缓存的 bundle dict，未命中返回 None
    """
    global _cache_errors

    if not settings.RETRIEVAL_CACHE_ENABLED:
        await _incr_counter(REDIS_KEY_MISSES)
        return None

    redis = await _get_redis()
    if redis is None:
        await _incr_counter(REDIS_KEY_MISSES)
        return None

    cache_key = _build_cache_key(queries_text, index_version)

    try:
        cached = await redis.get(cache_key)
        if cached:
            await _incr_counter(REDIS_KEY_HITS)
            logger.debug(f"检索缓存命中: {cache_key}")
            return json.loads(cached)
        await _incr_counter(REDIS_KEY_MISSES)
        logger.debug(f"检索缓存未命中: {cache_key}")
        return None
    except Exception as e:
        _cache_errors += 1
        logger.warning(f"检索缓存读取异常（静默降级）: {e}")
        return None


async def set_cached_bundle(
    queries_text: str,
    index_version: str,
    bundle_dict: dict,
) -> None:
    """将 RetrievalBundle 写入缓存

    Args:
        queries_text: 所有查询文本的拼接
        index_version: 当前索引版本
        bundle_dict: 序列化的 RetrievalBundle
    """
    global _cache_errors

    if not settings.RETRIEVAL_CACHE_ENABLED:
        return

    redis = await _get_redis()
    if redis is None:
        return

    cache_key = _build_cache_key(queries_text, index_version)

    try:
        serialized = json.dumps(bundle_dict, ensure_ascii=False)
        await redis.set(
            cache_key,
            serialized,
            ex=settings.RETRIEVAL_CACHE_TTL,
        )
        logger.debug(f"检索缓存已写入: {cache_key}, TTL={settings.RETRIEVAL_CACHE_TTL}s")

        # 维护缓存大小上限
        await _enforce_max_size(redis)
    except Exception as e:
        _cache_errors += 1
        logger.warning(f"检索缓存写入异常（静默降级）: {e}")


async def clear_retrieval_cache() -> int:
    """清空所有检索缓存

    Returns:
        清除的缓存条目数
    """
    global _cache_errors
    redis = await _get_redis()
    if redis is None:
        return 0

    try:
        cursor = 0
        deleted = 0
        while True:
            cursor, keys = await redis.scan(
                cursor=cursor, match=f"{CACHE_KEY_PREFIX}:*", count=100
            )
            if keys:
                await redis.delete(*keys)
                deleted += len(keys)
            if cursor == 0:
                break
        logger.info(f"检索缓存已全部清除，共 {deleted} 条")
        return deleted
    except Exception as e:
        _cache_errors += 1
        logger.warning(f"检索缓存清除异常（静默降级）: {e}")
        return 0


async def get_retrieval_cache_stats() -> dict:
    """返回缓存统计信息（供监控使用）"""
    # 从 Redis 读取持久化计数器
    cache_hits = 0
    cache_misses = 0
    try:
        r = await _get_redis()
        if r is not None:
            cache_hits = int(await r.get(REDIS_KEY_HITS) or 0)
            cache_misses = int(await r.get(REDIS_KEY_MISSES) or 0)
    except Exception:
        pass

    total = cache_hits + cache_misses
    hit_rate = (cache_hits / total * 100) if total > 0 else 0.0

    # 异步获取当前缓存条目数
    cache_size = 0
    try:
        r = await _get_redis()
        if r is not None:
            cursor = 0
            while True:
                cursor, keys = await r.scan(
                    cursor=cursor, match=f"{CACHE_KEY_PREFIX}:*", count=100
                )
                cache_size += len(keys)
                if cursor == 0:
                    break
    except Exception:
        pass

    return {
        "enabled": settings.RETRIEVAL_CACHE_ENABLED,
        "ttl": settings.RETRIEVAL_CACHE_TTL,
        "max_size": settings.RETRIEVAL_CACHE_MAX_SIZE,
        "cache_hits": cache_hits,
        "cache_misses": cache_misses,
        "cache_errors": _cache_errors,
        "hit_rate": round(hit_rate, 2),
        "cache_size": cache_size,
    }


async def _enforce_max_size(r: aioredis.Redis) -> None:
    """近似强制最大缓存条目数（best-effort）

    当缓存条目超过 RETRIEVAL_CACHE_MAX_SIZE 时，
    使用 SCAN 找到 TTL 最小的键并删除，直到低于阈值。
    仅 1% 概率触发，避免每次写入都扫描。
    """
    # 仅 1% 概率执行清理，避免性能开销
    if random.random() > 0.01:
        return

    try:
        cursor = 0
        all_keys = []
        while True:
            cursor, keys = await r.scan(
                cursor=cursor, match=f"{CACHE_KEY_PREFIX}:*", count=200
            )
            all_keys.extend(keys)
            if cursor == 0:
                break

        if len(all_keys) <= settings.RETRIEVAL_CACHE_MAX_SIZE:
            return

        # 超出上限，删除最旧（TTL 最小）的键
        excess = len(all_keys) - settings.RETRIEVAL_CACHE_MAX_SIZE
        keys_with_ttl = []
        for key in all_keys:
            ttl = await r.ttl(key)
            if ttl >= 0:
                keys_with_ttl.append((ttl, key))

        keys_with_ttl.sort(key=lambda x: x[0])
        to_delete = [k for _, k in keys_with_ttl[:excess]]
        if to_delete:
            await r.delete(*to_delete)
            logger.info(f"检索缓存清理: 删除 {len(to_delete)} 条旧缓存")

    except Exception as e:
        logger.warning(f"检索缓存大小维护异常（静默降级）: {e}")


async def close_retrieval_cache_redis() -> None:
    """关闭检索缓存 Redis 连接（在 FastAPI shutdown 中调用）"""
    global _redis_client
    if _redis_client is not None:
        await _redis_client.close()
        _redis_client = None
        logger.info("检索缓存 Redis 连接已关闭")
