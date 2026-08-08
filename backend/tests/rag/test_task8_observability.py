"""Task 8 Prometheus and trace observability contracts."""

from unittest.mock import AsyncMock, Mock

import pytest

from app.services.observability import metrics
from app.services.rag.retriever import fusion, tiered
from app.services.rag.types import RetrievalBundle, RetrievalQuery


def test_task8_prometheus_metric_names_are_available():
    expected = {
        "RAG_INDEX_GENERATION": "rag_index_generation",
        "BM25_LOAD_SECONDS": "bm25_load_seconds",
        "BM25_QUERY_SECONDS": "bm25_query_seconds",
        "BM25_CANDIDATES": "bm25_candidates",
        "BM25_TOP_SCORE": "bm25_top_score",
        "LEXICAL_EXPANSION_COUNT": "lexical_expansion_count",
        "FILTER_FALLBACK": "filter_fallback",
        "CACHE_HIT": "cache_hit",
        "RETRIEVAL_LEVEL": "retrieval_level",
        "RAG_CHANNEL_CANDIDATES": "rag_channel_candidates",
    }

    for attribute, metric_name in expected.items():
        metric = getattr(metrics, attribute)
        assert metric._name == metric_name


def test_record_rag_observability_normalizes_required_trace_fields():
    trace = metrics.record_rag_observability(
        {
            "index_generation": "rag-v2",
            "bm25_load_seconds": 0.25,
            "bm25_query_seconds": 0.001,
            "bm25_candidates": 12,
            "bm25_top_score": 4.2,
            "lexical_expansion_count": 2,
            "filter_fallback": False,
            "cache_hit": True,
            "retrieval_level": "base",
            "channel_candidates": {"bm25": 12, "dense": 9, "sparse": 0},
        }
    )

    assert trace["index_generation"] == "rag-v2"
    assert trace["cache_hit"] is True
    assert trace["channel_candidates"] == {"bm25": 12, "dense": 9, "sparse": 0}


@pytest.mark.asyncio
async def test_hybrid_recall_reports_bm25_timings_and_channel_counts(monkeypatch):
    dense = AsyncMock(return_value=[])
    bm25 = Mock()
    bm25.search.return_value = [
        {"doc_id": "bm25-1", "source": "guide.pdf", "bm25_score": 4.2}
    ]
    monkeypatch.setattr(fusion, "retrieve_medical_evidence", dense)
    monkeypatch.setattr(fusion, "get_bm25_index", lambda _generation: bm25)
    monkeypatch.setattr(
        fusion, "get_active_index_generation", AsyncMock(return_value="rag-v2")
    )
    monkeypatch.setattr(fusion.settings, "BGE_M3_ENABLED", False)

    _results, meta = await fusion.hybrid_recall("query", top_k=3)

    assert meta["generation"] == "rag-v2"
    assert meta["channel_candidates"] == {"bm25": 1, "dense": 0, "sparse": 0}
    assert meta["bm25_candidates"] == 1
    assert meta["bm25_top_score"] == 4.2
    assert meta["bm25_load_seconds"] >= 0
    assert meta["bm25_query_seconds"] >= 0


@pytest.mark.asyncio
async def test_tiered_cache_hit_is_recorded_in_trace(monkeypatch):
    cached = RetrievalBundle(
        status="candidate",
        level_used="base",
        queries=[RetrievalQuery(query_type="case", text="query")],
        candidates=[],
        trace={"index_generation": "rag-v2", "retrieval_level": "base"},
    ).model_dump()
    monkeypatch.setattr(
        tiered, "get_active_index_generation", AsyncMock(return_value="rag-v2")
    )
    monkeypatch.setattr(tiered, "get_cached_bundle", AsyncMock(return_value=cached))

    result = await tiered.tiered_retrieve(
        [RetrievalQuery(query_type="case", text="query")], top_k_per_query=3
    )

    assert result.trace["cache_hit"] is True
