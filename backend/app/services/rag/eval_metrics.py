# -*- coding: utf-8 -*-
"""RAG 检索质量指标 — 基于 ground truth 的排序评估纯函数

所有函数均为无副作用纯函数，输入 (ranked_ids, relevant_ids)，可独立单测。
约定：
- ranked_ids: 检索返回的文档标识列表，按相关性从高到低排列（rank 从 1 开始）
- relevant_ids: 该查询的相关文档标识集合（ground truth）

标识可以是 doc_id，也可以是"期望来源组"的 id（见 evaluate_retrieval 的用法）。
"""

from __future__ import annotations

import math
from typing import Dict, Iterable, List, Sequence, Set


def _as_set(relevant: Iterable[str]) -> Set[str]:
    return relevant if isinstance(relevant, set) else set(relevant)


def hit_at_k(ranked_ids: Sequence[str], relevant_ids: Iterable[str], k: int) -> float:
    """top-k 内是否至少命中一个相关文档（1.0 / 0.0）"""
    rel = _as_set(relevant_ids)
    if not rel:
        return 0.0
    return 1.0 if any(doc in rel for doc in ranked_ids[:k]) else 0.0


def recall_at_k(ranked_ids: Sequence[str], relevant_ids: Iterable[str], k: int) -> float:
    """recall@k = |命中的相关文档| / |全部相关文档|"""
    rel = _as_set(relevant_ids)
    if not rel:
        return 0.0
    hit = sum(1 for doc in set(ranked_ids[:k]) if doc in rel)
    return hit / len(rel)


def precision_at_k(ranked_ids: Sequence[str], relevant_ids: Iterable[str], k: int) -> float:
    """precision@k = |top-k 内的相关文档| / k"""
    if k <= 0:
        return 0.0
    rel = _as_set(relevant_ids)
    if not rel:
        return 0.0
    hit = sum(1 for doc in ranked_ids[:k] if doc in rel)
    return hit / k


def reciprocal_rank(ranked_ids: Sequence[str], relevant_ids: Iterable[str]) -> float:
    """首个相关文档的倒数排名（Reciprocal Rank）；未命中返回 0"""
    rel = _as_set(relevant_ids)
    if not rel:
        return 0.0
    for idx, doc in enumerate(ranked_ids, start=1):
        if doc in rel:
            return 1.0 / idx
    return 0.0


def dcg_at_k(ranked_ids: Sequence[str], relevant_ids: Iterable[str], k: int) -> float:
    """折损累计增益（二值相关性，gain=1）：sum(1 / log2(rank + 1))"""
    rel = _as_set(relevant_ids)
    if not rel:
        return 0.0
    dcg = 0.0
    for idx, doc in enumerate(ranked_ids[:k], start=1):
        if doc in rel:
            dcg += 1.0 / math.log2(idx + 1)
    return dcg


def ndcg_at_k(ranked_ids: Sequence[str], relevant_ids: Iterable[str], k: int) -> float:
    """归一化 DCG@k = DCG@k / IDCG@k（二值相关性）"""
    rel = _as_set(relevant_ids)
    if not rel:
        return 0.0
    dcg = dcg_at_k(ranked_ids, rel, k)
    # 理想排序：min(相关文档数, k) 个相关文档排在最前
    ideal_hits = min(len(rel), k)
    idcg = sum(1.0 / math.log2(idx + 1) for idx in range(1, ideal_hits + 1))
    if idcg == 0.0:
        return 0.0
    return dcg / idcg


def evaluate_case(
    ranked_ids: Sequence[str],
    relevant_ids: Iterable[str],
    k_values: Sequence[int] = (1, 3, 5, 10),
) -> Dict[str, float]:
    """对单个查询计算全部指标，返回扁平化 metric 字典

    输出键示例：hit@1, recall@5, precision@3, ndcg@10, mrr
    """
    rel = _as_set(relevant_ids)
    metrics: Dict[str, float] = {"mrr": round(reciprocal_rank(ranked_ids, rel), 6)}
    for k in k_values:
        metrics[f"hit@{k}"] = round(hit_at_k(ranked_ids, rel, k), 6)
        metrics[f"recall@{k}"] = round(recall_at_k(ranked_ids, rel, k), 6)
        metrics[f"precision@{k}"] = round(precision_at_k(ranked_ids, rel, k), 6)
        metrics[f"ndcg@{k}"] = round(ndcg_at_k(ranked_ids, rel, k), 6)
    return metrics


def aggregate(case_metrics: List[Dict[str, float]]) -> Dict[str, float]:
    """对多个查询的 metric 字典按键求宏平均（macro-average）"""
    if not case_metrics:
        return {}
    keys = set()
    for m in case_metrics:
        keys.update(m.keys())
    agg: Dict[str, float] = {}
    for key in keys:
        vals = [m[key] for m in case_metrics if key in m]
        agg[key] = round(sum(vals) / len(vals), 6) if vals else 0.0
    return agg
