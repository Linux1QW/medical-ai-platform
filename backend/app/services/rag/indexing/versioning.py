# -*- coding: utf-8 -*-
"""索引版本管理 — 版本切换、健康检查与自动回滚"""

import asyncio
import logging
import re
from pathlib import Path

from app.services.rag.medical_store import get_medical_store

logger = logging.getLogger(__name__)


async def switch_index_version(new_version: str, *, auto_rollback: bool = True) -> dict:
    """切换活跃索引版本，带健康检查和自动回滚

    Args:
        new_version: 新版本标识（如 "rag-v2"）
        auto_rollback: 切换后健康检查失败时是否自动回滚

    Returns:
        {"previous": str, "current": str, "doc_count": int}
        或 {"error": str} 如果切换失败
    """
    from app.core.config import settings
    from app.services.rag.medical_store import _reset_collection_cache

    previous = getattr(settings, 'ACTIVE_INDEX_VERSION', 'rag-v1')

    # 验证新版本 collection 存在
    store = get_medical_store()
    if store.client is None:
        store._init_client()

    new_collection_name = f"medical_guidelines_{new_version}"
    try:
        col = store.client.get_collection(new_collection_name)
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

        # 持久化到 .env 文件（健康检查通过后）
        # 从 indexing/versioning.py 出发，向上 5 级到 backend/ 目录
        env_path = Path(__file__).resolve().parent.parent.parent.parent.parent / ".env"
        if env_path.exists():
            content = env_path.read_text(encoding="utf-8")
            if "ACTIVE_INDEX_VERSION" in content:
                content = re.sub(
                    r'ACTIVE_INDEX_VERSION\s*=\s*.*',
                    f'ACTIVE_INDEX_VERSION={new_version}',
                    content,
                )
            else:
                content += f"\nACTIVE_INDEX_VERSION={new_version}\n"
            env_path.write_text(content, encoding="utf-8")
            logger.info(f"已将 ACTIVE_INDEX_VERSION={new_version} 持久化到 .env")
        else:
            # .env 不存在时创建新文件
            env_path.write_text(f"ACTIVE_INDEX_VERSION={new_version}\n", encoding="utf-8")
            logger.info(f"已创建 .env 文件并写入 ACTIVE_INDEX_VERSION={new_version}")

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
                query_embeddings=[query_embedding],
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
