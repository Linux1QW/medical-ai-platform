# -*- coding: utf-8 -*-
"""Regression tests for the single hybrid-recall path."""

from unittest.mock import AsyncMock, Mock

import pytest

from app.services.rag.indexing.manifest import IndexGenerationMismatch
from app.services.rag.retriever import fusion, hybrid
from app.services.rag.sparse_search import LearnedSparseSearch


def hit(doc_id: str, **scores: float) -> dict:
    """Build the smallest document payload accepted by retrieval fusion."""
    return {
        "doc_id": doc_id,
        "text": f"document {doc_id}",
        "source": "guideline.pdf",
        **scores,
    }


@pytest.mark.asyncio
async def test_sparse_only_hit_survives_fusion(monkeypatch):
    sparse = Mock()
    sparse.is_indexed = True
    sparse.search.return_value = [hit("s1", sparse_score=8.0)]

    monkeypatch.setattr(fusion, "retrieve_medical_evidence", AsyncMock(return_value=[]))
    monkeypatch.setattr(
        fusion,
        "get_active_index_generation",
        AsyncMock(return_value="rag-test"),
    )
    monkeypatch.setattr(
        fusion,
        "get_bm25_index",
        lambda _generation: Mock(search=Mock(return_value=[])),
    )
    monkeypatch.setattr(
        fusion,
        "get_sparse_search",
        lambda _generation: sparse,
    )
    monkeypatch.setattr(fusion.settings, "BGE_M3_ENABLED", True)

    fused, _ = await fusion.hybrid_recall("lung cancer", top_k=5)

    assert [item["doc_id"] for item in fused] == ["s1"]
    assert fused[0]["sparse_score"] == 8.0
    assert fused[0]["rrf_score"] > 0


@pytest.mark.asyncio
async def test_legacy_hybrid_uses_unified_recall(monkeypatch):
    recall = AsyncMock(return_value=([hit("d1", rrf_score=0.02)], {}))
    monkeypatch.setattr(hybrid, "hybrid_recall", recall)

    result = await hybrid.hybrid_retrieve("肺癌", top_k=5, enable_rerank=False)

    recall.assert_awaited_once_with("肺癌", top_k=15)
    assert result == [hit("d1", rrf_score=0.02)]


def test_sparse_search_returns_string_document_ids():
    encoder = Mock()
    encoder.encode_corpus.return_value = {
        "dense": Mock(),
        "sparse": [{1: 1.0}, {1: 2.0}],
    }
    encoder.encode_query.return_value = {"sparse": {1: 1.0}}
    search = LearnedSparseSearch()
    search.set_encoder(encoder)
    search.build_index(
        [
            {"doc_id": 101, "text": "first", "source": "a.pdf"},
            {"doc_id": "second", "text": "second", "source": "b.pdf"},
        ]
    )

    results = search.search("query", top_k=2)

    assert results == [
        {"doc_id": "second", "text": "second", "source": "b.pdf", "sparse_score": 2.0},
        {"doc_id": "101", "text": "first", "source": "a.pdf", "sparse_score": 1.0},
    ]


def test_sparse_search_empty_rebuild_clears_stale_index():
    encoder = Mock()
    encoder.encode_corpus.return_value = {
        "dense": Mock(),
        "sparse": [{1: 1.0}],
    }
    encoder.encode_query.return_value = {"sparse": {1: 1.0}}
    search = LearnedSparseSearch()
    search.set_encoder(encoder)
    search.build_index([{"doc_id": "old", "text": "old"}])

    assert search.is_indexed is True

    search.build_index([])

    assert search.is_indexed is False
    assert search.search("query", top_k=1) == []


def test_rrf_document_identity_is_string_and_uses_settings():
    fused, meta = fusion.run_three_way_fusion(
        bm25=[hit("7", bm25_score=5.0)],
        dense=[hit(7, score=0.8)],
        sparse=[],
    )

    assert [item["doc_id"] for item in fused] == ["7"]
    assert fused[0]["bm25_score"] == 5.0
    assert fused[0]["vector_score"] == 0.8
    assert meta["k"] == fusion.settings.RRF_K


@pytest.mark.asyncio
async def test_three_way_fusion_accepts_unified_tuple_entries():
    fused, meta = await fusion.run_three_way_fusion(
        bm25=[("tuple-doc", 5.0, hit("tuple-doc"))],
        dense=[],
        sparse=[],
    )

    assert [item["doc_id"] for item in fused] == ["tuple-doc"]
    assert fused[0]["bm25_score"] == 5.0
    assert meta["sources"] == {"bm25": 1, "dense": 0, "sparse": 0}


def test_three_way_fusion_preserves_channel_scores_and_generation(monkeypatch):
    fused, meta = fusion.run_three_way_fusion(
        bm25=[hit("shared", bm25_score=9.0)],
        dense=[hit("shared", score=0.75)],
        sparse=[hit("shared", sparse_score=4.0)],
        generation="rag-test",
    )

    assert fused == [
        {
            "doc_id": "shared",
            "text": "document shared",
            "source": "guideline.pdf",
            "bm25_score": 9.0,
            "generation": "rag-test",
            "vector_score": 0.75,
            "sparse_score": 4.0,
            "rrf_score": round(1.0 / (fusion.settings.RRF_K + 1), 6),
            "score": round(1.0 / (fusion.settings.RRF_K + 1), 6),
        }
    ]
    assert meta["weights"] == [0.30, 0.45, 0.25]


def test_fusion_rejects_channel_result_from_another_generation():
    with pytest.raises(IndexGenerationMismatch, match="dense"):
        fusion.run_three_way_fusion(
            bm25=[],
            dense=[
                {
                    **hit("d1", score=0.8),
                    "generation": "rag-stale",
                }
            ],
            sparse=[],
            generation="rag-active",
        )


@pytest.mark.asyncio
async def test_legacy_hybrid_optionally_reranks_unified_candidates(monkeypatch):
    candidates = [
        {
            **hit(f"d{i}", rrf_score=0.02 - i / 1000),
            "sparse_score": 4.0 - i,
            "generation": "rag-test",
        }
        for i in range(4)
    ]
    recall = AsyncMock(return_value=(candidates, {}))
    reranked = [
        {
            "doc_id": "d2",
            "text": "document d2",
            "source": "guideline.pdf",
            "rerank_score": 9.0,
        },
        {
            "doc_id": "d0",
            "text": "document d0",
            "source": "guideline.pdf",
            "rerank_score": 8.0,
        },
    ]
    rerank = AsyncMock(return_value=reranked)
    monkeypatch.setattr(hybrid, "hybrid_recall", recall)
    monkeypatch.setattr(hybrid, "rerank_documents", rerank)

    result = await hybrid.hybrid_retrieve(
        "肺癌", top_k=2, enable_rerank=True, rerank_threshold=6
    )

    recall.assert_awaited_once_with("肺癌", top_k=6)
    rerank.assert_awaited_once_with(
        query="肺癌", documents=candidates, top_k=2, threshold=6
    )
    assert [item["doc_id"] for item in result] == ["d2", "d0"]
    assert result[0]["sparse_score"] == 2.0
    assert result[0]["rrf_score"] == candidates[2]["rrf_score"]
    assert result[0]["generation"] == "rag-test"
