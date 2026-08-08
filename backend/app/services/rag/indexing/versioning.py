# -*- coding: utf-8 -*-
"""索引版本管理 — 版本切换、健康检查与自动回滚"""

import asyncio
import hashlib
import json
import logging
import threading
from typing import Any, Optional, cast

import redis
import redis.asyncio as aioredis

from app.core.config import settings
from app.services.rag import bm25_search, sparse_search
from app.services.rag.bm25_search import install_bm25_index
from app.services.rag.indexing.manifest import (
    IndexGenerationMismatch,
    RAGIndexManifest,
    load_rag_index_manifest,
    manifest_path,
    validate_candidate_manifest,
)
from app.services.rag.lexical.artifacts import load_bm25_artifact
from app.services.rag.medical_store import get_medical_store
from app.services.rag.sparse_search import load_sparse_artifact

logger = logging.getLogger(__name__)

ACTIVE_GENERATION_KEY = "rag:active_generation"
INDEX_SWITCHED_CHANNEL = "rag:index-switched"
_CAS_ACTIVE_GENERATION = """
local current = redis.call('GET', KEYS[1])
if current == ARGV[1] then
    redis.call('SET', KEYS[1], ARGV[2])
    return 1
end
return 0
"""
_generation_redis: Optional[aioredis.Redis] = None
_generation_redis_loop: Optional[asyncio.AbstractEventLoop] = None
_worker_reload_lock = bm25_search._registry_lock
_worker_listener_stop: Optional[threading.Event] = None
_worker_listener_thread: Optional[threading.Thread] = None


class ActiveGenerationConflict(RuntimeError):
    """Raised when another publisher wins the active pointer race."""


def publish_index_switched(
    generation: str,
    previous: Optional[str],
    manifest_sha256: str,
    *,
    redis: Any = None,
) -> bool:
    """Publish a validated generation switch for every application Worker."""
    client = redis or _get_sync_redis()
    payload = json.dumps(
        {
            "generation": generation,
            "previous": previous,
            "manifest_sha256": manifest_sha256,
        },
        ensure_ascii=False,
        separators=(",", ":"),
    )
    client.publish(INDEX_SWITCHED_CHANNEL, payload)
    return True


def _get_sync_redis() -> Any:
    return redis.Redis.from_url(
        settings.REDIS_CHECKPOINT_URL,
        decode_responses=True,
        socket_connect_timeout=3.0,
        socket_timeout=3.0,
    )


def load_generation_for_worker(
    generation: str,
    *,
    manifest_sha256: Optional[str] = None,
    artifact_root: Any = None,
) -> str:
    """Validate and atomically install one generation in this Worker.

    Every potentially failing load occurs before the local references are
    changed.  A failed event therefore leaves the old BM25 and Chroma objects
    serving requests.
    """
    manifest = load_rag_index_manifest(generation, artifact_root)
    validate_candidate_manifest(manifest)
    if manifest_sha256 is not None:
        actual_digest = hashlib.sha256(
            manifest_path(generation, artifact_root).read_bytes()
        ).hexdigest()
        if actual_digest != manifest_sha256:
            raise IndexGenerationMismatch("RAG manifest SHA-256 does not match switch event")

    store = get_medical_store()
    collection = store.get_collection_for_generation(generation)
    if collection.count() != manifest.chunk_count:
        raise IndexGenerationMismatch("chroma document count does not match manifest")

    # These are local candidates.  Do not install them until all validation is
    # complete, especially because bm25s mmap loading can fail on a cold Worker.
    bm25_candidate = load_bm25_artifact(generation, artifact_root, mmap=True)
    sparse_candidate = None
    if settings.BGE_M3_ENABLED:
        sparse_candidate = load_sparse_artifact(
            generation, artifact_root, install=False
        )

    with _worker_reload_lock:
        old_bm25_index = bm25_search._bm25_index
        old_bm25_generation = bm25_search._bm25_index_generation
        old_bm25_indexes = dict(bm25_search._bm25_indexes)
        old_sparse = sparse_search._sparse_search
        old_sparse_searches = dict(sparse_search._sparse_searches)
        old_collection = store.collection
        old_active_generation = settings.ACTIVE_INDEX_VERSION
        try:
            install_bm25_index(generation, bm25_candidate)
            sparse_search._sparse_search = sparse_candidate
            if sparse_candidate is not None:
                sparse_search._sparse_searches[generation] = sparse_candidate
            store.collection = collection
            # Keep legacy no-argument callers aligned with the Redis-published
            # generation. This is process-local runtime state, never .env storage.
            settings.ACTIVE_INDEX_VERSION = generation
        except Exception:
            bm25_search._bm25_index = old_bm25_index
            bm25_search._bm25_index_generation = old_bm25_generation
            bm25_search._bm25_indexes.clear()
            bm25_search._bm25_indexes.update(old_bm25_indexes)
            sparse_search._sparse_search = old_sparse
            sparse_search._sparse_searches.clear()
            sparse_search._sparse_searches.update(old_sparse_searches)
            store.collection = old_collection
            settings.ACTIVE_INDEX_VERSION = old_active_generation
            raise
    return generation


def _worker_event_loop(client: Any, stop_event: threading.Event) -> None:
    pubsub = client.pubsub(ignore_subscribe_messages=True)
    pubsub.subscribe(INDEX_SWITCHED_CHANNEL)
    try:
        while not stop_event.is_set():
            message = pubsub.get_message(timeout=1.0)
            if not message:
                continue
            try:
                payload = json.loads(message.get("data", "{}"))
                load_generation_for_worker(
                    payload["generation"],
                    manifest_sha256=payload.get("manifest_sha256"),
                )
            except Exception:
                logger.exception("failed to load RAG generation switch event")
    finally:
        pubsub.close()


def start_index_switch_listener() -> None:
    """Start the per-process Redis listener used by Celery Workers."""
    global _worker_listener_stop, _worker_listener_thread
    if _worker_listener_thread and _worker_listener_thread.is_alive():
        return
    _worker_listener_stop = threading.Event()
    client = _get_sync_redis()
    _worker_listener_thread = threading.Thread(
        target=_worker_event_loop,
        args=(client, _worker_listener_stop),
        name="rag-index-switch-listener",
        daemon=True,
    )
    _worker_listener_thread.start()


def stop_index_switch_listener() -> None:
    global _worker_listener_stop, _worker_listener_thread
    if _worker_listener_stop is not None:
        _worker_listener_stop.set()
    if _worker_listener_thread is not None:
        _worker_listener_thread.join(timeout=2.0)
    _worker_listener_stop = None
    _worker_listener_thread = None


async def _get_generation_redis() -> aioredis.Redis:
    global _generation_redis, _generation_redis_loop
    current_loop = asyncio.get_running_loop()
    if _generation_redis is None or _generation_redis_loop is not current_loop:
        if _generation_redis is not None:
            try:
                await _generation_redis.aclose()
            except Exception:
                logger.debug("closing stale generation Redis client failed", exc_info=True)
        _generation_redis = aioredis.from_url(
            settings.REDIS_CHECKPOINT_URL,
            decode_responses=True,
            socket_connect_timeout=3.0,
            socket_timeout=3.0,
        )
        await _generation_redis.ping()
        _generation_redis_loop = current_loop
    return _generation_redis


async def get_active_index_generation(redis: Any = None) -> Optional[str]:
    """Read the cluster-wide active generation pointer."""
    client: Any = redis if redis is not None else await _get_generation_redis()
    value = await client.get(ACTIVE_GENERATION_KEY)
    return str(value) if value is not None else None


async def compare_and_set_active_generation(
    *,
    expected_generation: str,
    candidate_generation: str,
    redis: Any = None,
) -> bool:
    """Atomically activate a candidate only if the expected pointer still wins."""
    client: Any = redis if redis is not None else await _get_generation_redis()
    switched = await client.eval(
        _CAS_ACTIVE_GENERATION,
        1,
        ACTIVE_GENERATION_KEY,
        expected_generation,
        candidate_generation,
    )
    if int(switched) != 1:
        actual = await client.get(ACTIVE_GENERATION_KEY)
        raise ActiveGenerationConflict(
            f"active generation changed concurrently: expected "
            f"{expected_generation!r}, found {actual!r}"
        )
    return True


async def activate_candidate_generation(
    manifest: RAGIndexManifest,
    *,
    expected_generation: str,
    redis: Any = None,
    artifact_root: Any = None,
) -> dict:
    """Validate one candidate and atomically publish its shared pointer."""
    validate_candidate_manifest(manifest)
    store = get_medical_store()
    collection = store.get_collection_for_generation(manifest.index_generation)
    if collection.count() != manifest.chunk_count:
        raise IndexGenerationMismatch(
            "chroma document count does not match candidate manifest"
        )
    load_bm25_artifact(manifest.index_generation, artifact_root)
    if settings.BGE_M3_ENABLED:
        load_sparse_artifact(
            manifest.index_generation,
            artifact_root,
            install=False,
        )
    await compare_and_set_active_generation(
        expected_generation=expected_generation,
        candidate_generation=manifest.index_generation,
        redis=redis,
    )
    return {
        "previous": expected_generation,
        "current": manifest.index_generation,
        "doc_count": manifest.chunk_count,
    }


async def switch_index_version(new_version: str, *, auto_rollback: bool = True) -> dict:
    """切换活跃索引版本，带健康检查和自动回滚

    Args:
        new_version: 新版本标识（如 "rag-v2"）
        auto_rollback: 切换后健康检查失败时是否自动回滚

    Returns:
        {"previous": str, "current": str, "doc_count": int}
        或 {"error": str} 如果切换失败
    """
    from app.services.rag.medical_store import _reset_collection_cache

    previous = getattr(settings, 'ACTIVE_INDEX_VERSION', 'rag-v1')

    # 验证新版本 collection 存在
    store = get_medical_store()
    client = store._ensure_client()

    new_collection_name = f"medical_guidelines_{new_version}"
    try:
        col = client.get_collection(new_collection_name)
        doc_count = col.count()
    except Exception:
        return {"error": f"Collection '{new_collection_name}' 不存在"}

    # 保存原始状态用于回滚
    original_version = previous
    original_collection = store.collection

    # 执行切换
    try:
        # 更新配置（运行时更新）
        settings.ACTIVE_INDEX_VERSION = new_version
        _reset_collection_cache()  # 清除缓存，确保新版本生效

        # 更新 medical_store 的 collection 引用
        store.collection = col

        # 重建 BM25 索引
        try:
            from app.services.rag.bm25_search import rebuild_bm25_index
            await asyncio.to_thread(rebuild_bm25_index)
            logger.info(f"BM25 索引已重建（版本: {new_version}）")
        except Exception as e:
            logger.warning(f"BM25 索引重建失败（非致命）: {e}")

        # 重建 Sparse 索引（当 BGE-M3 启用时）
        try:
            from app.services.rag.sparse_search import rebuild_sparse_index
            sparse_ok = await asyncio.to_thread(rebuild_sparse_index)
            if sparse_ok:
                logger.info(f"Sparse 索引已重建（版本: {new_version}）")
            else:
                logger.info("Sparse 索引跳过（BGE-M3 未启用或不可用）")
        except Exception as e:
            logger.warning(f"Sparse 索引重建失败（非致命）: {e}")

        # 健康检查
        if auto_rollback:
            healthy = await _health_check_index(timeout=10.0)
            if not healthy:
                logger.warning(
                    f"Health check failed after switching to {new_version}, "
                    f"rolling back to {original_version}"
                )
                # 回滚
                settings.ACTIVE_INDEX_VERSION = original_version
                store.collection = original_collection
                _reset_collection_cache()
                # 重建 BM25 回滚版本
                try:
                    from app.services.rag.bm25_search import rebuild_bm25_index
                    await asyncio.to_thread(rebuild_bm25_index)
                    from app.services.rag.sparse_search import rebuild_sparse_index
                    await asyncio.to_thread(rebuild_sparse_index)
                except Exception:
                    pass
                return {"error": f"健康检查失败，已回滚到 {original_version}"}

    except Exception as e:
        logger.error(f"Switch failed: {e}")
        # 异常时尝试回滚
        if auto_rollback:
            settings.ACTIVE_INDEX_VERSION = original_version
            store.collection = original_collection
            _reset_collection_cache()
        return {"error": f"切换失败: {e}"}

    logger.info(f"索引版本切换: {previous} → {new_version} (文档数: {doc_count})")

    return {
        "previous": previous,
        "current": new_version,
        "doc_count": doc_count,
    }


async def _health_check_index(timeout: float = 10.0) -> bool:
    """索引健康检查：执行标准查询验证索引可用性和响应时间

    使用简单的向量查询验证索引可用性，不调用 LLM。

    Args:
        timeout: 单次查询超时阈值（秒）

    Returns:
        True 健康，False 不健康
    """
    import time

    from app.services.rag.medical_store import get_medical_store

    try:
        store = get_medical_store()
        if store.collection is None:
            logger.warning("Health check: collection is None")
            return False

        # 使用标准测试查询验证索引
        test_queries = ["高血压 诊疗指南", "糖尿病 治疗方案"]

        # 获取 embedding 函数（get_embedding 为 async，内部走 LRU 缓存）
        from app.services.rag.embeddings import get_embedding

        for query in test_queries:
            start = time.time()

            # 生成查询向量
            query_embedding = await get_embedding(query)

            # 执行向量查询
            results = store.collection.query(
                query_embeddings=cast(Any, [query_embedding]),
                n_results=3,
            )

            elapsed = time.time() - start

            if elapsed > timeout:
                logger.warning(f"Health check query took {elapsed:.2f}s (timeout={timeout}s)")
                return False

            # 检查结果
            if not results or not results.get('ids') or not results['ids'][0]:
                logger.warning(f"Health check query returned no results: {query}")
                return False

        logger.info("Health check passed")
        return True

    except Exception as e:
        logger.error(f"Health check failed: {e}")
        return False
