"""Worker resource lifecycle — ordered shutdown of all async resources.

Shutdown order (in close_worker_resources):
1. Graph (clear compiled cache)
2. Checkpointer (close Redis connection)
3. HTTP clients (Qwen AsyncOpenAI, embedding client)
4. Redis clients (LLM cache, retrieval cache, evaluation control)
5. DB engine (SQLAlchemy async engine)
"""

from __future__ import annotations

import logging

logger = logging.getLogger(__name__)


async def close_worker_resources() -> None:
    """Close all async resources in the correct order.

    Called by WorkerAsyncRuntime.stop() inside the worker's own event loop.
    Each close function is idempotent — safe to call even if the resource
    was never initialized or already closed.
    """
    # 1. Graph (clear compiled cache — no async close needed)
    try:
        from app.orchestration.graph import close_graph
        await close_graph()
    except Exception:
        logger.debug("close_graph skipped or failed", exc_info=True)

    # 2. Checkpointer (close Redis connection via AsyncExitStack)
    try:
        from app.orchestration.checkpointer import close_checkpointer
        await close_checkpointer()
    except Exception:
        logger.debug("close_checkpointer skipped or failed", exc_info=True)

    # 3. HTTP clients — Qwen AsyncOpenAI
    try:
        from app.services.qwen_client import client as qwen_client
        if qwen_client is not None:
            await qwen_client.close()
            from app.services import qwen_client as qc_mod
            qc_mod.client = None
    except Exception:
        logger.debug("close qwen_client skipped or failed", exc_info=True)

    # 4. HTTP clients — embedding client
    try:
        from app.services.rag.embeddings import _embed_client, _http_client
        await _embed_client.close()
        await _http_client.aclose()
    except Exception:
        logger.debug("close embedding clients skipped or failed", exc_info=True)

    # 5. Redis clients — LLM cache
    try:
        from app.services.llm_cache import close_cache_redis
        await close_cache_redis()
    except Exception:
        logger.debug("close_cache_redis skipped or failed", exc_info=True)

    # 6. Redis clients — retrieval cache
    try:
        from app.services.rag.retrieval_cache import close_retrieval_cache_redis
        await close_retrieval_cache_redis()
    except Exception:
        logger.debug("close_retrieval_cache_redis skipped or failed", exc_info=True)

    # 7. Redis clients — evaluation control
    try:
        from app.services.evaluation_cancel import close_control_redis
        await close_control_redis()
    except Exception:
        logger.debug("close_control_redis skipped or failed", exc_info=True)

    # 8. DB engine (SQLAlchemy async engine)
    try:
        from app.db.session import dispose_engine
        await dispose_engine()
    except Exception:
        logger.debug("dispose_engine skipped or failed", exc_info=True)

    logger.info("All worker async resources closed")
