# -*- coding: utf-8 -*-
"""RRF 融合与三路混合召回 — BM25 + Dense + (可选) Sparse"""

import asyncio
import logging
from typing import Dict, List

from app.core.config import settings
from app.services.rag.bm25_search import get_bm25_index
from app.services.rag.hybrid_fusion import weighted_rrf
from app.services.rag.retriever.base import retrieve_medical_evidence
from app.services.rag.sparse_search import get_sparse_search

logger = logging.getLogger(__name__)

# ── RRF 融合参数 ──
RRF_K = 60  # Reciprocal Rank Fusion 常数，控制排名权重衰减速度


def reciprocal_rank_fusion(
    vector_results: List[Dict],
    bm25_results: List[Dict],
    top_k: int = 10,
    k: int = RRF_K,
) -> List[Dict]:
    """Reciprocal Rank Fusion（RRF）— 融合向量检索和 BM25 检索结果

    RRF 公式：score(d) = Σ 1 / (k + rank_i(d))
    其中 rank_i(d) 为文档 d 在第 i 个检索列表中的排名（从 1 开始）

    Args:
        vector_results: 向量检索结果列表
        bm25_results: BM25 关键词检索结果列表
        top_k: 返回条数
        k: RRF 常数（默认 60，值越大排名差异的影响越小）

    Returns:
        融合后按 RRF 分数降序排列的文档列表
    """
    # 优先使用稳定的 doc_id 作为去重键；降级到 text[:100]（小概率出现前缀相同但内容不同的情况）
    doc_scores = {}   # dedup_key -> {"doc": Dict, "rrf_score": float}

    # 处理向量检索结果
    for rank, doc in enumerate(vector_results, 1):
        text = doc.get("text", "")
        dedup_key = doc.get("doc_id") or (text[:100] if len(text) > 100 else text)
        if not dedup_key:
            continue

        rrf_score = 1.0 / (k + rank)
        if dedup_key in doc_scores:
            doc_scores[dedup_key]["rrf_score"] += rrf_score
        else:
            doc_scores[dedup_key] = {"doc": doc, "rrf_score": rrf_score}

    # 处理 BM25 检索结果
    for rank, doc in enumerate(bm25_results, 1):
        text = doc.get("text", "")
        dedup_key = doc.get("doc_id") or (text[:100] if len(text) > 100 else text)
        if not dedup_key:
            continue

        rrf_score = 1.0 / (k + rank)
        if dedup_key in doc_scores:
            doc_scores[dedup_key]["rrf_score"] += rrf_score
        else:
            doc_scores[dedup_key] = {"doc": doc, "rrf_score": rrf_score}

    # 按 RRF 分数降序排列
    sorted_results = sorted(
        doc_scores.values(), key=lambda x: x["rrf_score"], reverse=True
    )

    # 返回 top_k，附带 RRF 分数
    final_results = []
    for item in sorted_results[:top_k]:
        doc_copy = dict(item["doc"])
        doc_copy["rrf_score"] = round(item["rrf_score"], 6)
        # 保留原始向量相似度分数（来自向量检索通道的 score）
        if "score" in doc_copy and "bm25_score" not in doc_copy:
            doc_copy["vector_score"] = doc_copy["score"]
        # 用 rrf_score 作为统一的 score 字段（用于后续排序）
        doc_copy["score"] = doc_copy["rrf_score"]
        final_results.append(doc_copy)

    return final_results


async def hybrid_recall(  # noqa: C901
    query: str,
    top_k: int = 10,
) -> tuple:
    """三路混合检索融合：BM25 + Dense + (可选) Sparse via weighted_rrf

    当 BGE_M3_ENABLED=True 时，并行执行三路检索并通过 weighted_rrf 融合；
    否则降级为 BM25 + Dense 两路融合。

    Args:
        query: 查询文本
        top_k: 每路召回条数

    Returns:
        (fused_results, fusion_meta)
        - fused_results: List[Dict] 融合后的文档列表（含 rrf_score / score 字段）
        - fusion_meta: dict 融合元信息（sources 各路人马数量、weights、fused_count）
    """
    loop = asyncio.get_event_loop()
    recall_k = top_k * 3  # 粗召回数量（给融合留足候选）

    # ── 定义三路检索函数 ──

    async def vector_search() -> List[Dict]:
        try:
            return await retrieve_medical_evidence(query, top_k=recall_k)
        except Exception as e:
            logger.warning(f"hybrid_recall-Dense通道失败: {e}")
            return []

    def bm25_search_sync() -> List[Dict]:
        try:
            index = get_bm25_index()
            return index.search(query, top_k=recall_k)
        except Exception as e:
            logger.warning(f"hybrid_recall-BM25通道失败: {e}")
            return []

    def sparse_search_sync() -> List[Dict]:
        """BGE-M3 Learned Sparse 检索通道（可选降级）"""
        try:
            ss = get_sparse_search()
            if ss is None or not ss.is_indexed:
                return []
            results = ss.search(query, top_k=recall_k)
            return [{"_sparse_idx": idx, "sparse_score": score} for idx, score in results]
        except Exception as e:
            logger.warning(f"hybrid_recall-Sparse通道失败（降级）: {e}")
            return []

    # ── 并行执行三路检索 ──
    if settings.BGE_M3_ENABLED:
        vector_results, bm25_results, sparse_results = await asyncio.gather(
            vector_search(),
            loop.run_in_executor(None, bm25_search_sync),
            loop.run_in_executor(None, sparse_search_sync),
        )
    else:
        vector_results, bm25_results = await asyncio.gather(
            vector_search(),
            loop.run_in_executor(None, bm25_search_sync),
        )
        sparse_results = []

    logger.info(
        f"hybrid_recall 粗召回: BM25={len(bm25_results)}, "
        f"Dense={len(vector_results)}, Sparse={len(sparse_results)}"
    )

    # ── 构建 ranking 格式用于 weighted_rrf ──
    # BM25 ranking: (doc_id, bm25_score)
    bm25_ranking = [
        (doc["doc_id"] if "doc_id" in doc else doc.get("id", f"bm25_{i}"),
         doc.get("bm25_score", 0.0))
        for i, doc in enumerate(bm25_results)
    ]
    # Dense ranking: (doc_id, vector_score)
    dense_ranking = [
        (doc.get("doc_id", f"dense_{i}"),
         doc.get("score", doc.get("vector_score", 0.0)))
        for i, doc in enumerate(vector_results)
    ]
    # Sparse ranking: (_sparse_idx, sparse_score)
    sparse_ranking = [
        (d["_sparse_idx"], d["sparse_score"])
        for d in sparse_results
    ]

    # ── 加权 RRF 融合 ──
    # 构建 doc_id -> 原始 dict 的映射（用于融合后还原完整文档）
    doc_map: Dict = {}
    for doc in bm25_results:
        key = doc.get("doc_id") or doc.get("id", "")
        if key:
            doc_map[key] = doc
    for doc in vector_results:
        key = doc.get("doc_id", "")
        if key and key not in doc_map:
            doc_map[key] = doc
        elif key in doc_map:
            # 合并 vector_score 到已有记录
            doc_map[key]["vector_score"] = doc.get("score", doc.get("vector_score", 0.0))

    if sparse_ranking:
        # 三路融合（权重来自 config，便于基于评估集调参）
        fusion_weights = [
            settings.RRF_WEIGHT_BM25,
            settings.RRF_WEIGHT_DENSE,
            settings.RRF_WEIGHT_SPARSE,
        ]
        fused = weighted_rrf(
            rankings=[bm25_ranking, dense_ranking, sparse_ranking],
            weights=fusion_weights,
            k=35,
        )
    else:
        # 降级为两路融合：取 BM25/Dense 权重归一化
        _bm25, _dense = settings.RRF_WEIGHT_BM25, settings.RRF_WEIGHT_DENSE
        _total = _bm25 + _dense
        if _total <= 0:
            fusion_weights = [0.40, 0.60]
        else:
            fusion_weights = [_bm25 / _total, _dense / _total]
        fused = weighted_rrf(
            rankings=[bm25_ranking, dense_ranking],
            weights=fusion_weights,
            k=35,
        )

    # ── 将融合后的 (doc_id, score) 还原为完整 dict ──
    fused_results: List[Dict] = []
    for doc_id, rrf_score in fused:
        doc = doc_map.get(doc_id)
        if doc is None:
            # sparse-only 文档（无 BM25/Dense 匹配），尝试从 sparse 结果还原
            continue
        doc_copy = dict(doc)
        doc_copy["rrf_score"] = round(rrf_score, 6)
        doc_copy["score"] = doc_copy["rrf_score"]
        fused_results.append(doc_copy)

    fusion_meta = {
        "method": "weighted_rrf",
        "k": 35,
        "sources": {
            "bm25": len(bm25_ranking),
            "dense": len(dense_ranking),
            "sparse": len(sparse_ranking),
        },
        "weights": fusion_weights,
        "fused_count": len(fused_results),
    }

    return fused_results, fusion_meta
