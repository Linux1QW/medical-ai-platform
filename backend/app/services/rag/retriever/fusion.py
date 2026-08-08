# -*- coding: utf-8 -*-
"""Unified BM25, dense, and learned-sparse retrieval fusion."""

import asyncio
import logging
from typing import Any

from app.core.config import settings
from app.services.rag.bm25_search import get_bm25_index
from app.services.rag.hybrid_fusion import weighted_rrf
from app.services.rag.retriever.base import retrieve_medical_evidence
from app.services.rag.sparse_search import get_sparse_search

logger = logging.getLogger(__name__)


class FusionResult(tuple):
    """Tuple result that remains await-compatible with the Task 5 test shape."""

    def __new__(
        cls,
        fused_results: list[dict[str, Any]],
        meta: dict[str, Any],
    ) -> "FusionResult":
        return super().__new__(cls, (fused_results, meta))

    def __await__(self):
        async def _identity() -> tuple[list[dict[str, Any]], dict[str, Any]]:
            return tuple(self)

        return _identity().__await__()


def _document_id(document: dict[str, Any]) -> str:
    """Return the stable document identity used by every retrieval channel."""
    document_id = document.get("doc_id", document.get("id"))
    if document_id is None or document_id == "":
        return ""
    return str(document_id)


def _normalized_document(document: dict[str, Any], channel: str) -> dict[str, Any]:
    """Keep channel scores separate while normalizing document identity."""
    normalized = dict(document)
    normalized["doc_id"] = _document_id(document)
    normalized.setdefault("generation", str(settings.ACTIVE_INDEX_VERSION))
    if channel == "bm25":
        normalized["bm25_score"] = float(document.get("bm25_score", 0.0))
    elif channel == "dense":
        normalized["vector_score"] = float(
            document.get("vector_score", document.get("score", 0.0))
        )
    elif channel == "sparse":
        normalized["sparse_score"] = float(document.get("sparse_score", 0.0))
    return normalized


def _channel_entry_to_document(
    entry: Any,
    channel: str,
    score_field: str,
) -> dict[str, Any]:
    """Normalize either a document dict or (doc_id, score, document) channel entry."""
    if isinstance(entry, (tuple, list)) and len(entry) == 3:
        doc_id, score, document = entry
        normalized = dict(document) if isinstance(document, dict) else {}
        normalized["doc_id"] = "" if doc_id is None else str(doc_id)
        normalized[score_field] = float(score)
        return _normalized_document(normalized, channel)

    if isinstance(entry, dict):
        return _normalized_document(entry, channel)

    return {}


def run_three_way_fusion(
    bm25: list[Any],
    dense: list[Any],
    sparse: list[Any],
    top_k: int | None = None,
) -> FusionResult:
    """Fuse ranked channel results by stable string document identity."""
    channels = (
        ("bm25", bm25, "bm25_score"),
        ("dense", dense, "vector_score"),
        ("sparse", sparse, "sparse_score"),
    )
    documents: dict[str, dict[str, Any]] = {}
    rankings: list[list[tuple[str, float]]] = []

    for channel, raw_documents, score_field in channels:
        normalized = [
            document
            for document in (
                _channel_entry_to_document(entry, channel, score_field)
                for entry in raw_documents
            )
            if _document_id(document)
        ]
        normalized.sort(key=lambda document: document.get(score_field, 0.0), reverse=True)
        rankings.append(
            [(document["doc_id"], float(document.get(score_field, 0.0))) for document in normalized]
        )
        for document in normalized:
            existing = documents.get(document["doc_id"])
            if existing is None:
                documents[document["doc_id"]] = document
            else:
                existing.update(
                    {
                        key: value
                        for key, value in document.items()
                        if key.endswith("_score") or not existing.get(key)
                    }
                )

    weights = [
        settings.RRF_WEIGHT_BM25,
        settings.RRF_WEIGHT_DENSE,
        settings.RRF_WEIGHT_SPARSE,
    ]
    if not sparse:
        two_way_total = settings.RRF_WEIGHT_BM25 + settings.RRF_WEIGHT_DENSE
        weights = (
            [settings.RRF_WEIGHT_BM25 / two_way_total, settings.RRF_WEIGHT_DENSE / two_way_total]
            if two_way_total > 0
            else [0.40, 0.60]
        )
        rankings = rankings[:2]

    fused = weighted_rrf(rankings=rankings, weights=weights, k=settings.RRF_K)
    if top_k is not None:
        fused = fused[:top_k]

    fused_results = []
    for document_id, rrf_score in fused:
        document = dict(documents[document_id])
        document["doc_id"] = str(document_id)
        document["rrf_score"] = round(rrf_score, 6)
        document["score"] = document["rrf_score"]
        fused_results.append(document)

    return FusionResult(
        fused_results,
        {
            "method": "weighted_rrf",
            "k": settings.RRF_K,
            "sources": {"bm25": len(bm25), "dense": len(dense), "sparse": len(sparse)},
            "weights": weights,
            "fused_count": len(fused_results),
        },
    )


def reciprocal_rank_fusion(
    vector_results: list[dict[str, Any]],
    bm25_results: list[dict[str, Any]],
    top_k: int = 10,
) -> list[dict[str, Any]]:
    """Compatibility wrapper for callers that only provide dense and BM25 results."""
    fused, _ = run_three_way_fusion(
        bm25=bm25_results,
        dense=vector_results,
        sparse=[],
        top_k=top_k,
    )
    return fused


async def hybrid_recall(query: str, top_k: int = 10) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Run all enabled retrieval channels and fuse them with one RRF implementation."""
    recall_k = top_k * 3
    loop = asyncio.get_running_loop()

    async def dense_search() -> list[dict[str, Any]]:
        try:
            return await retrieve_medical_evidence(query, top_k=recall_k)
        except Exception as error:
            logger.warning("hybrid_recall dense channel failed: %s", error)
            return []

    def bm25_search() -> list[dict[str, Any]]:
        try:
            return get_bm25_index().search(query, top_k=recall_k)
        except Exception as error:
            logger.warning("hybrid_recall BM25 channel failed: %s", error)
            return []

    def sparse_search() -> list[dict[str, Any]]:
        try:
            search = get_sparse_search()
            return search.search(query, top_k=recall_k) if search and search.is_indexed else []
        except Exception as error:
            logger.warning("hybrid_recall sparse channel failed: %s", error)
            return []

    if settings.BGE_M3_ENABLED:
        dense, bm25, sparse = await asyncio.gather(
            dense_search(),
            loop.run_in_executor(None, bm25_search),
            loop.run_in_executor(None, sparse_search),
        )
    else:
        dense, bm25 = await asyncio.gather(
            dense_search(), loop.run_in_executor(None, bm25_search)
        )
        sparse = []

    return run_three_way_fusion(bm25=bm25, dense=dense, sparse=sparse, top_k=top_k)
