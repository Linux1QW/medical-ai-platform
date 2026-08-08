# -*- coding: utf-8 -*-
"""混合检索融合策略 — 加权 RRF + 分数归一化

提供多路检索结果的融合方法：
1. weighted_rrf: 加权 Reciprocal Rank Fusion，医学场景默认权重
   - BM25: 0.30（精确术语匹配补充）
   - Dense: 0.45（语义向量主导）
   - Sparse: 0.25（稀疏特征辅助）
2. simple_weighted_sum: 简单加权分数融合（备选方案）
3. normalize_scores: Min-Max 分数归一化
"""

from typing import Optional

from app.core.config import settings


def normalize_scores(scores: list[float]) -> list[float]:
    """Min-Max 归一化到 [0, 1]

    Args:
        scores: 原始分数列表

    Returns:
        归一化后的分数列表；若所有分数相同则返回全 1.0
    """
    if not scores:
        return []
    min_s, max_s = min(scores), max(scores)
    range_s = max_s - min_s
    if range_s < 1e-8:
        return [1.0] * len(scores)
    return [(s - min_s) / range_s for s in scores]


def weighted_rrf(
    rankings: list[list[tuple[str, float]]],
    weights: Optional[list[float]] = None,
    k: Optional[int] = None,
) -> list[tuple[str, float]]:
    """加权 RRF (Reciprocal Rank Fusion) 融合

    公式：score(d) = Σ weight_i / (k + rank_i(d))
    其中 rank 从 0 开始（即 rank+1 等效于从 1 开始的排名）

    Args:
        rankings: 每个检索源的结果 [(doc_id, score), ...]
        weights: 每个检索源的权重
                 默认 [0.30, 0.45, 0.25] 对应 [BM25, Dense, Sparse]
        k: RRF 常数；未提供时读取统一配置

    Returns:
        融合后的 [(doc_id, fused_score), ...] 按分数降序
    """
    if not rankings:
        return []

    if k is None:
        k = settings.RRF_K

    n_sources = len(rankings)
    if weights is None:
        # 医学场景默认权重：向量主导，BM25 补充精确匹配
        if n_sources == 3:
            weights = [
                settings.RRF_WEIGHT_BM25,
                settings.RRF_WEIGHT_DENSE,
                settings.RRF_WEIGHT_SPARSE,
            ]
        elif n_sources == 2:
            total = settings.RRF_WEIGHT_BM25 + settings.RRF_WEIGHT_DENSE
            weights = (
                [
                    settings.RRF_WEIGHT_BM25 / total,
                    settings.RRF_WEIGHT_DENSE / total,
                ]
                if total > 0
                else [0.40, 0.60]
            )
        else:
            weights = [1.0 / n_sources] * n_sources

    # 加权 RRF 计算（RRF 仅依赖排名，不依赖原始分数）
    fused_scores: dict[str, float] = {}
    for ranking, weight in zip(rankings, weights, strict=False):
        for rank, (doc_id, _score) in enumerate(ranking):
            rrf_score = weight / (k + rank + 1)
            document_id = str(doc_id)
            fused_scores[document_id] = fused_scores.get(document_id, 0.0) + rrf_score

    # 按融合分数降序排列
    return sorted(fused_scores.items(), key=lambda x: x[1], reverse=True)


def simple_weighted_sum(
    rankings: list[list[tuple[str, float]]],
    weights: Optional[list[float]] = None,
) -> list[tuple[str, float]]:
    """简单加权分数融合（备选方案）

    适用于各检索源分数已归一化到同一量纲的场景。
    公式：score(d) = Σ weight_i * score_i(d)

    Args:
        rankings: 每个检索源的结果 [(doc_id, score), ...]
        weights: 每个检索源的权重，默认等权

    Returns:
        融合后的 [(doc_id, fused_score), ...] 按分数降序
    """
    if not rankings:
        return []

    n_sources = len(rankings)
    if weights is None:
        weights = [1.0 / n_sources] * n_sources

    fused: dict[str, float] = {}
    for ranking, weight in zip(rankings, weights, strict=False):
        for doc_id, score in ranking:
            document_id = str(doc_id)
            fused[document_id] = fused.get(document_id, 0.0) + weight * score

    return sorted(fused.items(), key=lambda x: x[1], reverse=True)
