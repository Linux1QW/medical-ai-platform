# -*- coding: utf-8 -*-
"""基于 BGE-M3 Learned Sparse 表示的检索

BGE-M3 在训练时联合优化了 learned sparse 表示（类似 SPLADE），
每个 token 都被赋予一个权重，形成词表级的稀疏向量。
相比传统 BM25（基于统计频率），learned sparse 能捕捉语义相关性。

本模块提供：
- LearnedSparseSearch：基于 learned sparse 的检索器
- 与现有 BM25 + Dense 两路检索互补，为三路融合检索提供基础

降级策略：
- 当 DualEncoder 不可用时（BGE_M3_ENABLED=False 或模型加载失败），
  search() 返回空列表，系统自动降级为现有 BM25 + Dense 方案。
"""

import logging
from typing import Optional

import numpy as np

from app.services.rag.dual_encoder import DualEncoder, get_dual_encoder

logger = logging.getLogger(__name__)


class LearnedSparseSearch:
    """
    基于 BGE-M3 learned sparse 表示的检索器

    类似 SPLADE，但由 BGE-M3 统一提供 dense + sparse 双表示。
    语料和查询均编码为 {token_id: weight} 稀疏向量，
    通过点积计算相似度。

    降级行为：
    - 未设置编码器时，search() 返回空列表
    - 索引未构建时，search() 返回空列表
    """

    def __init__(self):
        self._encoder: Optional[DualEncoder] = None
        self._corpus_sparse: list = []   # list of {token_id: weight}
        self._corpus_dense: Optional[np.ndarray] = None  # (N, 1024)

    def set_encoder(self, encoder: DualEncoder) -> None:
        """设置双表示编码器

        Args:
            encoder: DualEncoder 实例（来自 get_dual_encoder()）
        """
        self._encoder = encoder
        logger.info("LearnedSparseSearch: 编码器已设置")

    def build_index(self, texts: list) -> None:
        """构建稀疏索引（编码所有文档）

        Args:
            texts: 语料文本列表

        Raises:
            RuntimeError: 编码器未设置时
        """
        if not self._encoder:
            logger.warning("LearnedSparseSearch.build_index: 编码器未设置，跳过索引构建")
            return

        try:
            result = self._encoder.encode_corpus(texts)
            self._corpus_dense = result["dense"]
            self._corpus_sparse = result["sparse"]
            logger.info(
                f"LearnedSparseSearch 索引构建完成：{len(texts)} 条文档，"
                f"dense shape={self._corpus_dense.shape}"
            )
        except Exception as e:
            logger.error(f"LearnedSparseSearch 索引构建失败: {e}")
            self._corpus_sparse = []
            self._corpus_dense = None

    def search(self, query: str, top_k: int = 30) -> list:
        """检索：使用 learned sparse 表示计算相似度

        Args:
            query: 查询文本
            top_k: 返回条数

        Returns:
            [(doc_index, score), ...] 按分数降序排列
            当编码器不可用或索引未构建时返回空列表（降级）
        """
        if not self._encoder:
            logger.debug("LearnedSparseSearch.search: 编码器未设置，返回空结果（降级）")
            return []

        if not self._corpus_sparse:
            logger.debug("LearnedSparseSearch.search: 索引为空，返回空结果")
            return []

        try:
            query_enc = self._encoder.encode_query(query)
            query_sparse = query_enc["sparse"]

            # 计算查询与所有文档的 sparse 点积相似度
            scores = []
            for i, doc_sparse in enumerate(self._corpus_sparse):
                score = self._sparse_dot(query_sparse, doc_sparse)
                scores.append((i, score))

            # 按分数降序排列，取 top_k
            scores.sort(key=lambda x: x[1], reverse=True)
            result = scores[:top_k]

            logger.debug(
                f"LearnedSparseSearch.search: query='{query[:40]}...' "
                f"→ {len(result)} 条结果，top score={result[0][1]:.4f}" if result else "→ 0 条结果"
            )
            return result

        except Exception as e:
            logger.warning(f"LearnedSparseSearch.search 失败，降级返回空结果: {e}")
            return []

    @staticmethod
    def _sparse_dot(vec_a: dict, vec_b: dict) -> float:
        """计算两个稀疏向量的点积

        优化：遍历较小的向量，在较大向量中查找匹配项。

        Args:
            vec_a: {token_id: weight}
            vec_b: {token_id: weight}

        Returns:
            点积分数
        """
        # 遍历较小的向量以提升效率
        if len(vec_a) > len(vec_b):
            vec_a, vec_b = vec_b, vec_a

        score = 0.0
        for token_id, weight_a in vec_a.items():
            if token_id in vec_b:
                score += weight_a * vec_b[token_id]
        return score

    @property
    def is_indexed(self) -> bool:
        """索引是否已构建"""
        return bool(self._corpus_sparse)


# ── 模块级单例 ─────────────────────────────────────────────────────────────────

_sparse_search: Optional[LearnedSparseSearch] = None


def get_sparse_search() -> Optional[LearnedSparseSearch]:
    """获取全局 LearnedSparseSearch 实例

    当 BGE_M3_ENABLED=False 或 DualEncoder 不可用时返回 None，
    调用方应降级为现有 BM25 + Dense 两路检索。
    """
    global _sparse_search

    if not _sparse_search:
        encoder = get_dual_encoder()
        if encoder is None:
            return None

        _sparse_search = LearnedSparseSearch()
        _sparse_search.set_encoder(encoder)

    return _sparse_search


def rebuild_sparse_index() -> bool:
    """重建 Sparse 索引（在索引版本切换后调用）

    从 ChromaDB 加载文档并构建 learned sparse 索引。
    当 BGE_M3_ENABLED=False 或编码器不可用时跳过。

    Returns:
        True 表示索引构建成功，False 表示失败或跳过
    """
    global _sparse_search

    encoder = get_dual_encoder()
    if encoder is None:
        logger.info("Sparse 索引重建: BGE-M3 未启用，跳过")
        return False

    try:
        from app.services.rag.medical_store import _get_collection_name, get_medical_store

        store = get_medical_store()
        if store.client is None:
            store._init_client()

        collection_name = _get_collection_name()
        try:
            collection = store.client.get_collection(collection_name)
        except Exception:
            logger.warning(f"Sparse 索引: collection '{collection_name}' 不存在")
            return False

        if collection.count() == 0:
            logger.warning(f"Sparse 索引: collection '{collection_name}' 为空")
            return False

        # 从 collection 获取所有文档文本
        count = collection.count()
        all_texts = []
        batch_size = 1000
        for offset in range(0, count, batch_size):
            result = collection.get(
                limit=batch_size,
                offset=offset,
                include=["documents"],
            )
            if result["documents"]:
                all_texts.extend(result["documents"])

        if not all_texts:
            logger.warning("Sparse 索引: 无文档可索引")
            return False

        # 重建实例
        _sparse_search = LearnedSparseSearch()
        _sparse_search.set_encoder(encoder)
        _sparse_search.build_index(all_texts)

        logger.info(f"Sparse 索引已重建: {len(all_texts)} 条文档")
        return True

    except Exception as e:
        logger.error(f"Sparse 索引重建失败: {e}")
        return False
