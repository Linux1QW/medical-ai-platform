# -*- coding: utf-8 -*-
"""分级检索 — Level 1 (BM25+向量+RRF) → Level 2 (MQE) → Level 3 (HyDE)"""

import logging
import time
from typing import Any, Literal

from app.core.config import settings
from app.services.observability.langfuse_client import get_tracer
from app.services.observability.metrics import (
    RAG_RETRIEVAL_DURATION,
    record_rag_observability,
)
from app.services.rag.indexing.manifest import IndexGenerationMismatch
from app.services.rag.indexing.versioning import get_active_index_generation
from app.services.rag.retrieval_cache import get_cached_bundle, set_cached_bundle
from app.services.rag.retriever.fusion import hybrid_recall
from app.services.rag.retriever.hyde import hyde_retrieve
from app.services.rag.retriever.mqe import expand_queries
from app.services.rag.retriever.similarity import _filter_by_embedding_similarity
from app.services.rag.types import (
    MAX_MQE_EXPANSIONS,
    MAX_RAG_CANDIDATES,
    EvidenceItem,
    RetrievalBundle,
    RetrievalConfidence,
)

logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────────────────────
# 分级检索：辅助函数
# ─────────────────────────────────────────────────────────────────────────────

def _dict_to_evidence(
    dicts: list,
    query_type: str,
    retrieved_via: str = "base",
    *,
    generation: str | None = None,
) -> list:
    """将 dict 格式的检索结果转为 EvidenceItem"""
    items = []
    for d in dicts:
        item_generation = d.get("generation") or generation
        if generation is not None and item_generation != generation:
            raise IndexGenerationMismatch(
                f"candidate generation {item_generation!r} does not match "
                f"runtime generation {generation!r}"
            )
        vector_score = d.get("vector_score")
        if vector_score is None and d.get("rrf_score") is None:
            vector_score = d.get("score")
        item = EvidenceItem(
            doc_id=str(d.get("doc_id", d.get("id", ""))),
            generation=item_generation,
            text=d.get("text", ""),
            source=d.get("source", "未知"),
            page=d.get("page"),
            heading_path=d.get("heading_path", ""),
            chunk_seq=d.get("chunk_seq") if isinstance(d.get("chunk_seq"), int) and d.get("chunk_seq", -1) >= 0 else None,
            query_types=[query_type],
            vector_score=vector_score,
            bm25_score=d.get("bm25_score"),
            sparse_score=d.get("sparse_score"),
            rrf_score=d.get("rrf_score"),
            # 从 metadata 提取增强字段（如果存在）
            organization=d.get("organization"),
            year=d.get("year") if isinstance(d.get("year"), int) and d.get("year", 0) > 0 else None,
            version=d.get("version"),
            document_type=d.get("document_type"),
            departments=d.get("departments"),
            disease_tags=d.get("disease_tags"),
            population=d.get("population"),
            content_type=d.get("content_type"),
            recommendation_level=d.get("recommendation_level"),
            evidence_level=d.get("evidence_level"),
            retrieved_via=retrieved_via,
        )
        items.append(item)
    return items


def _assess_confidence(
    candidates: list,
    query_types_covered: int,
    source_count: int,
    max_vector_score: float,
    max_rrf_score: float,
) -> RetrievalConfidence:
    """评估检索置信度

    Returns:
        RetrievalConfidence.HIGH: 多来源高分，直接使用
        RetrievalConfidence.MEDIUM: 部分满足，尝试增强
        RetrievalConfidence.LOW: 严重不足，准备拒答
    """
    if not candidates:
        return RetrievalConfidence.LOW

    # HIGH: 充分证据
    if (
        len(candidates) >= 5
        and source_count >= 3
        and max_vector_score >= 0.7
        and query_types_covered >= 2
    ):
        return RetrievalConfidence.HIGH

    # MEDIUM: 有一定证据但不充分
    if (
        len(candidates) >= 3
        and source_count >= 2
        and (max_vector_score >= 0.5 or max_rrf_score >= 0.015)
    ):
        return RetrievalConfidence.MEDIUM

    # LOW: 证据严重不足
    return RetrievalConfidence.LOW


def _assess_retrieval(
    candidates: list,
    query_types_with_hits: set,
    all_query_types: set,
) -> str:
    """兼容包装，返回 'sufficient' 或 'insufficient'"""
    if not candidates:
        return "unavailable"

    sources = set(c.source for c in candidates)
    max_vector = max((c.vector_score or 0) for c in candidates)
    max_rrf = max((c.rrf_score or 0) for c in candidates)

    confidence = _assess_confidence(
        candidates=candidates,
        query_types_covered=len(query_types_with_hits),
        source_count=len(sources),
        max_vector_score=max_vector,
        max_rrf_score=max_rrf,
    )
    return "sufficient" if confidence in (
        RetrievalConfidence.HIGH, RetrievalConfidence.MEDIUM
    ) else "insufficient"


def _merge_evidence(
    *result_lists: list,
) -> list:
    """合并多路证据并去重（按 doc_id），保留最高分数"""
    seen: dict = {}

    for result_list in result_lists:
        for rank, item in enumerate(result_list, 1):
            candidate = item.model_copy()
            if candidate.rrf_score is None:
                candidate.rrf_score = round(1.0 / (settings.RRF_K + rank), 6)

            key = candidate.doc_id
            if key in seen:
                existing = seen[key]
                # 合并 query_types
                existing.query_types = list(
                    dict.fromkeys(existing.query_types + candidate.query_types)
                )
                # 保留各阶段最高分
                if (candidate.vector_score or 0) > (existing.vector_score or 0):
                    existing.vector_score = candidate.vector_score
                if (candidate.bm25_score or 0) > (existing.bm25_score or 0):
                    existing.bm25_score = candidate.bm25_score
                if (candidate.sparse_score or 0) > (existing.sparse_score or 0):
                    existing.sparse_score = candidate.sparse_score
                if (candidate.rrf_score or 0) > (existing.rrf_score or 0):
                    existing.rrf_score = candidate.rrf_score
                if existing.generation is None:
                    existing.generation = candidate.generation
            else:
                seen[key] = candidate

    results = list(seen.values())
    results.sort(key=lambda item: item.rrf_score or 0.0, reverse=True)
    return results


# ─────────────────────────────────────────────────────────────────────────────
# 分级检索主入口
# ─────────────────────────────────────────────────────────────────────────────

async def tiered_retrieve(  # noqa: C901
    queries: list,
    top_k_per_query: int = 10,
    candidate_limit: int = MAX_RAG_CANDIDATES,
) -> RetrievalBundle:
    """分级检索：Level 1 (BM25+向量+RRF) → Level 2 (MQE) → Level 3 (HyDE)

    每个级别判断召回是否充分，足够则提前返回。

    Args:
        queries: 检索查询列表（通常包含 case/diagnosis/treatment 三类）
        top_k_per_query: 每个查询的召回条数
        candidate_limit: 最终候选上限

    Returns:
        RetrievalBundle 包含状态、级别、查询和候选证据
    """
    # 空查询防御
    if not queries:
        logger.warning("tiered_retrieve called with empty queries list")
        return RetrievalBundle(
            status="unavailable",
            level_used="base",
            queries=[],
            candidates=[],
            trace={"error": "empty_queries"},
        )

    # ── 缓存检查 ──
    try:
        index_generation = await get_active_index_generation()
    except Exception as exc:
        logger.error("active RAG generation unavailable: %s", exc)
        index_generation = None
    if not index_generation:
        return RetrievalBundle(
            status="unavailable",
            level_used="base",
            queries=queries,
            candidates=[],
            trace={"error": "active_generation_unavailable"},
        )
    cached = await get_cached_bundle(
        queries,
        index_generation,
        top_k=top_k_per_query,
    )
    if cached is not None:
        logger.info(f"tiered_retrieve 缓存命中: {index_generation}")
        cached_result = RetrievalBundle.model_validate(cached)
        cached_result.trace = record_rag_observability(
            {
                **cached_result.trace,
                "index_generation": index_generation,
                "cache_hit": True,
            }
        )
        return cached_result

    start = time.monotonic()
    trace: dict[str, Any] = {
        "index_generation": index_generation,
        "cache_hit": False,
        "queries": [{"type": q.query_type, "text": q.text[:100], "source": q.source} for q in queries],
        "levels_attempted": [],
        "retrieval_level": "base",
        "candidate_count": 0,
        "rerank_input_count": 0,
        "llm_rerank_count": 0,
        "retrieval_status": "candidate",
        "timing": {
            "embedding_ms": 0,
            "retrieval_ms": 0,
            "rerank_ms": 0,
            "llm_ms": 0,
        },
        "estimated_cost": 0.0,
        "degraded": False,
        "retrieval": {"fusion": None},
    }

    set(q.query_type for q in queries)
    query_types_with_hits: set = set()
    all_candidates: list = []
    level_used: Literal["base", "mqe", "hyde"] = "base"
    decisions: list = []  # 每级决策记录

    def _compute_confidence(cands: list, qtypes_hits: set) -> RetrievalConfidence:
        """从候选列表计算置信度"""
        if not cands:
            return RetrievalConfidence.LOW
        sources = set(c.source for c in cands)
        max_vec = max((c.vector_score or 0) for c in cands)
        max_rrf = max((c.rrf_score or 0) for c in cands)
        return _assess_confidence(
            candidates=cands,
            query_types_covered=len(qtypes_hits),
            source_count=len(sources),
            max_vector_score=max_vec,
            max_rrf_score=max_rrf,
        )

    def _build_confidence_trace(conf: RetrievalConfidence) -> dict:
        """构建标准化置信度 trace 片段"""
        sources = set(c.source for c in all_candidates) if all_candidates else set()
        max_vec = max((c.vector_score or 0) for c in all_candidates) if all_candidates else 0
        max_rrf = max((c.rrf_score or 0) for c in all_candidates) if all_candidates else 0
        return {
            "confidence": conf.value,
            "scores": {
                "vector": round(max_vec, 4),
                "rrf": round(max_rrf, 4),
                "source_count": len(sources),
                "candidate_count": len(all_candidates),
                "query_types_covered": len(query_types_with_hits),
            },
            "thresholds": {
                "min_vector": 0.5,
                "min_rrf": 0.015,
                "min_candidates": 3,
                "min_sources": 2,
            },
        }

    # ── Level 1: 基础混合召回（三路融合） ──
    trace["levels_attempted"].append("base")
    fusion_info = None  # 记录融合元信息
    for query in queries:
        fused, fusion_meta = await hybrid_recall(
            query.text,
            top_k=top_k_per_query,
            generation=index_generation,
        )
        fusion_info = fusion_meta

        if not fused:
            continue

        if fused:
            query_types_with_hits.add(query.query_type)

        # 转换并合并
        evidence_items = _dict_to_evidence(
            fused,
            query.query_type,
            retrieved_via="base",
            generation=index_generation,
        )
        # 为融合结果设置 rrf_score
        for item in evidence_items:
            if item.rrf_score is None:
                raw = next((f for f in fused if f.get("doc_id") == item.doc_id), None)
                if raw:
                    item.rrf_score = raw.get("rrf_score", raw.get("score", 0))

        all_candidates = _merge_evidence(all_candidates, evidence_items)

    confidence = _compute_confidence(all_candidates, query_types_with_hits)
    decisions.append({"level": "base", "confidence": confidence.value, "candidates": len(all_candidates)})
    if confidence == RetrievalConfidence.HIGH:
        elapsed = time.monotonic() - start
        trace["total_ms"] = round(elapsed * 1000, 1)
        trace["timing"]["retrieval_ms"] = trace["total_ms"]
        trace["retrieval_level"] = "base"
        trace["candidate_count"] = len(all_candidates[:candidate_limit])
        trace["retrieval_status"] = "candidate"
        trace.update(_build_confidence_trace(confidence))
        trace["decisions"] = decisions
        trace["retrieval"]["fusion"] = fusion_info
        result = RetrievalBundle(
            status="candidate",
            level_used="base",
            queries=queries,
            candidates=all_candidates[:candidate_limit],
            confidence=confidence.value,
            trace=trace,
        )
        result.trace = record_rag_observability(result.trace)
        await set_cached_bundle(
            queries,
            index_generation,
            result.model_dump(),
            top_k=top_k_per_query,
        )
        return result
    # MEDIUM / LOW → 继续 L2 MQE

    # ── Level 2: MQE ──
    trace["levels_attempted"].append("mqe")
    level_used = "mqe"
    mqe_expansion_count = 0

    for query in queries:
        if mqe_expansion_count >= MAX_MQE_EXPANSIONS:
            break

        expanded = await expand_queries(query.text, n=2)
        if not expanded:
            continue

        # 语义漂移过滤
        expanded = await _filter_by_embedding_similarity(query.text, expanded)

        for eq in expanded:
            mqe_expansion_count += 1
            fused, fusion_meta = await hybrid_recall(
                eq,
                top_k=top_k_per_query,
                generation=index_generation,
            )
            fusion_info = fusion_meta

            if not fused:
                continue

            if fused:
                query_types_with_hits.add(query.query_type)

            evidence_items = _dict_to_evidence(
                fused,
                query.query_type,
                retrieved_via="mqe",
                generation=index_generation,
            )
            all_candidates = _merge_evidence(all_candidates, evidence_items)

            if len(all_candidates) >= candidate_limit:
                break

    confidence = _compute_confidence(all_candidates, query_types_with_hits)
    decisions.append({"level": "mqe", "confidence": confidence.value, "candidates": len(all_candidates)})
    if confidence in (RetrievalConfidence.HIGH, RetrievalConfidence.MEDIUM):
        elapsed = time.monotonic() - start
        trace["total_ms"] = round(elapsed * 1000, 1)
        trace["timing"]["retrieval_ms"] = trace["total_ms"]
        trace["mqe_expansions"] = mqe_expansion_count
        trace["retrieval_level"] = "mqe"
        trace["candidate_count"] = len(all_candidates[:candidate_limit])
        trace["retrieval_status"] = "candidate"
        trace.update(_build_confidence_trace(confidence))
        trace["decisions"] = decisions
        trace["retrieval"]["fusion"] = fusion_info
        result = RetrievalBundle(
            status="candidate",
            level_used="mqe",
            queries=queries,
            candidates=all_candidates[:candidate_limit],
            confidence=confidence.value,
            trace=trace,
        )
        result.trace = record_rag_observability(result.trace)
        await set_cached_bundle(
            queries,
            index_generation,
            result.model_dump(),
            top_k=top_k_per_query,
        )
        return result
    # LOW → 继续 L3 HyDE

    # ── Level 3: HyDE（每次评估最多 1 次）──
    trace["levels_attempted"].append("hyde")
    level_used = "hyde"

    # 选择最有价值的查询做 HyDE（优先 case 类型）
    hyde_query = next((q for q in queries if q.query_type == "case"), queries[0])
    hyde_success = False
    try:
        hyde_results = await hyde_retrieve(
            hyde_query.text,
            top_k=top_k_per_query,
            generation=index_generation,
        )
        if hyde_results:
            query_types_with_hits.add(hyde_query.query_type)
            evidence_items = _dict_to_evidence(
                hyde_results,
                hyde_query.query_type,
                retrieved_via="hyde",
                generation=index_generation,
            )
            all_candidates = _merge_evidence(all_candidates, evidence_items)
        hyde_success = True
    except Exception as e:
        logger.warning(f"HyDE 检索失败: {e}")

    # 最终评估
    confidence = _compute_confidence(all_candidates, query_types_with_hits)
    decisions.append({"level": "hyde", "confidence": confidence.value, "candidates": len(all_candidates)})
    elapsed = time.monotonic() - start
    trace["total_ms"] = round(elapsed * 1000, 1)
    trace["timing"]["retrieval_ms"] = trace["total_ms"]
    trace["mqe_expansions"] = mqe_expansion_count
    trace["hyde_calls"] = 1 if hyde_success else 0
    trace["retrieval_level"] = "hyde"
    trace["candidate_count"] = len(all_candidates[:candidate_limit])
    final_status: Literal["candidate", "insufficient"] = (
        "candidate" if confidence != RetrievalConfidence.LOW else "insufficient"
    )
    trace["retrieval_status"] = final_status
    trace.update(_build_confidence_trace(confidence))
    trace["decisions"] = decisions
    trace["retrieval"]["fusion"] = fusion_info

    result = RetrievalBundle(
        status=final_status,
        level_used=level_used,
        queries=queries,
        candidates=all_candidates[:candidate_limit],
        confidence=confidence.value,
        trace=trace,
    )
    result.trace = record_rag_observability(result.trace)

    # ── 写入缓存 ──
    await set_cached_bundle(
        queries,
        index_generation,
        result.model_dump(),
        top_k=top_k_per_query,
    )

    # ── Langfuse trace + Prometheus 指标 ──
    _elapsed_ms = (time.monotonic() - start) * 1000
    try:
        _query_text = queries[0].text if queries else ""
        get_tracer().trace_rag_retrieval(
            trace_name="rag_tiered_retrieve",
            query=_query_text,
            results=[{"score": c.rrf_score or 0} for c in result.candidates[:5]],
            latency_ms=_elapsed_ms,
        )
    except Exception as e:
        logger.debug(f"Langfuse RAG trace 异常: {e}")
    RAG_RETRIEVAL_DURATION.observe(_elapsed_ms / 1000)

    return result
