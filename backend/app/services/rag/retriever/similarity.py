# -*- coding: utf-8 -*-
"""向量相似度工具 — MQE 语义漂移防护"""

import logging
import math
from typing import List

from app.services.rag.embeddings import get_embeddings

logger = logging.getLogger(__name__)

# ── MQE 语义漂移防护阈值 ──
MQE_SIMILARITY_THRESHOLD = 0.7  # 扩展查询与原始查询的最低 embedding 余弦相似度


def _cosine_similarity(v1: List[float], v2: List[float]) -> float:
    """计算两个向量的余弦相似度"""
    dot = sum(a * b for a, b in zip(v1, v2, strict=False))
    norm1 = math.sqrt(sum(a * a for a in v1))
    norm2 = math.sqrt(sum(b * b for b in v2))
    if norm1 == 0.0 or norm2 == 0.0:
        return 0.0
    return dot / (norm1 * norm2)


async def _filter_by_embedding_similarity(
    original_query: str,
    expanded_queries: List[str],
    threshold: float = MQE_SIMILARITY_THRESHOLD,
) -> List[str]:
    """通过 embedding 余弦相似度校验，过滤与原始查询语义偏离过大的扩展查询

    防止 MQE 扩展后引入语义漂移：计算每条扩展 query 与原始 query 的向量相似度，
    低于阈值（默认 0.7）的扩展查询视为语义漂移被丢弃。

    Args:
        original_query: 原始医学查询文本
        expanded_queries: LLM 生成的扩展查询列表
        threshold: 相似度阈值（0-1），默认 0.7

    Returns:
        通过相似度校验的扩展查询列表
    """
    if not expanded_queries:
        return []

    try:
        # 并行获取原始查询和所有扩展查询的 embedding
        all_texts = [original_query] + expanded_queries
        all_embeddings = await get_embeddings(all_texts)
        orig_emb = all_embeddings[0]

        filtered = []
        for query, emb in zip(expanded_queries, all_embeddings[1:], strict=False):
            sim = _cosine_similarity(orig_emb, emb)
            if sim >= threshold:
                filtered.append(query)
                logger.debug(f"MQE 扩展查询通过语义校验: '{query[:30]}' (sim={sim:.3f})")
            else:
                logger.info(
                    f"MQE 扩展查询因语义漂移被过滤: '{query[:30]}' "
                    f"(sim={sim:.3f} < {threshold})"
                )

        logger.info(
            f"MQE 相似度校验：{len(expanded_queries)} 条扩展查询 "
            f"→ {len(filtered)} 条通过（阈值={threshold}）"
        )
        return filtered

    except Exception as e:
        logger.warning(f"MQE 相似度校验获取 embedding 失败，跳过过滤: {e}")
        return expanded_queries  # 降级：失败时保留全部扩展查询
