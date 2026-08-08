# -*- coding: utf-8 -*-
"""检索模块集成测试（需要知识库数据）"""

from unittest.mock import AsyncMock, Mock

import pytest

from app.services.rag.retriever import fusion, tiered
from app.services.rag.types import EvidenceItem, RetrievalConfidence, RetrievalQuery


class TestEvidenceConversion:
    """测试 dict 到 EvidenceItem 的转换"""

    def test_dict_structure(self):
        # 验证 EvidenceItem 可以从标准 dict 创建
        data = {
            "doc_id": "test-doc",
            "text": "测试文本内容",
            "source": "test.pdf",
            "page": 1,
            "heading_path": "第一章",
        }
        item = EvidenceItem(**data)
        assert item.doc_id == "test-doc"
        assert item.page == 1


class TestRetrievalBudget:
    """验证调用预算常量"""

    def test_budget_constants(self):
        from app.services.rag.types import (
            MAX_HYDE_CALLS,
            MAX_MQE_EXPANSIONS,
            MAX_RAG_CANDIDATES,
        )
        assert MAX_MQE_EXPANSIONS == 2
        assert MAX_HYDE_CALLS == 1
        assert MAX_RAG_CANDIDATES == 20


@pytest.mark.asyncio
async def test_dense_and_bm25_receive_exact_raw_query(monkeypatch):
    dense = AsyncMock(return_value=[])
    bm25_index = Mock()
    bm25_index.search.return_value = []
    monkeypatch.setattr(fusion, "retrieve_medical_evidence", dense)
    monkeypatch.setattr(fusion, "get_bm25_index", lambda _generation: bm25_index)
    monkeypatch.setattr(
        fusion,
        "get_active_index_generation",
        AsyncMock(return_value="rag-redis"),
    )
    monkeypatch.setattr(fusion.settings, "BGE_M3_ENABLED", False)

    await fusion.hybrid_recall("心梗治疗", top_k=5)

    dense.assert_awaited_once_with(
        "心梗治疗", top_k=15, generation="rag-redis"
    )
    bm25_index.search.assert_called_once_with("心梗治疗", top_k=15)


@pytest.mark.asyncio
async def test_tiered_retrieve_passes_original_query_text_to_hybrid_recall(monkeypatch):
    original_text = "  EGFR c.2573T>G ≥50%  "
    hybrid_recall = AsyncMock(
        return_value=(
            [
                {
                    "doc_id": "d1",
                    "text": "evidence",
                    "source": "guide.pdf",
                    "rrf_score": 0.02,
                }
            ],
            {"channels": ["bm25"]},
        )
    )
    get_cached = AsyncMock(return_value=None)
    set_cached = AsyncMock()
    monkeypatch.setattr(tiered, "get_cached_bundle", get_cached)
    monkeypatch.setattr(tiered, "set_cached_bundle", set_cached)
    monkeypatch.setattr(
        tiered,
        "get_active_index_generation",
        AsyncMock(return_value="rag-redis"),
    )
    monkeypatch.setattr(tiered, "hybrid_recall", hybrid_recall)
    monkeypatch.setattr(
        tiered,
        "_assess_confidence",
        lambda **kwargs: RetrievalConfidence.HIGH,
    )

    queries = [RetrievalQuery(query_type="case", text=original_text)]
    await tiered.tiered_retrieve(
        queries,
        top_k_per_query=5,
    )

    get_cached.assert_awaited_once_with(
        queries,
        "rag-redis",
        top_k=5,
    )
    hybrid_recall.assert_awaited_once_with(
        original_text,
        top_k=5,
        generation="rag-redis",
    )
    cached_payload = set_cached.await_args.args[2]
    assert cached_payload["candidates"][0]["generation"] == "rag-redis"
    assert set_cached.await_args.kwargs == {"top_k": 5}


@pytest.mark.asyncio
async def test_hybrid_channels_share_one_redis_generation(monkeypatch):
    dense = AsyncMock(
        return_value=[
            {
                "doc_id": "d1",
                "text": "dense",
                "source": "a.pdf",
                "score": 0.8,
                "generation": "rag-redis",
            }
        ]
    )
    bm25_index = Mock()
    bm25_index.search.return_value = []
    sparse = Mock(is_indexed=True)
    sparse.search.return_value = []
    bm25_get = Mock(return_value=bm25_index)
    sparse_get = Mock(return_value=sparse)
    monkeypatch.setattr(
        fusion,
        "get_active_index_generation",
        AsyncMock(return_value="rag-redis"),
    )
    monkeypatch.setattr(fusion, "retrieve_medical_evidence", dense)
    monkeypatch.setattr(fusion, "get_bm25_index", bm25_get)
    monkeypatch.setattr(fusion, "get_sparse_search", sparse_get)
    monkeypatch.setattr(fusion.settings, "BGE_M3_ENABLED", True)

    fused, _ = await fusion.hybrid_recall("query", top_k=3)

    dense.assert_awaited_once_with("query", top_k=9, generation="rag-redis")
    bm25_get.assert_called_once_with("rag-redis")
    sparse_get.assert_called_once_with("rag-redis")
    assert fused[0]["generation"] == "rag-redis"


def test_merge_evidence_converts_each_channel_rank_to_rrf():
    bm25_only = EvidenceItem(
        doc_id="bm25",
        text="bm25",
        source="a.pdf",
        bm25_score=100.0,
    )
    dense_only = EvidenceItem(
        doc_id="dense",
        text="dense",
        source="b.pdf",
        vector_score=0.01,
    )
    sparse_only = EvidenceItem(
        doc_id="sparse",
        text="sparse",
        source="c.pdf",
        sparse_score=8.0,
    )

    merged = tiered._merge_evidence(
        [bm25_only],
        [dense_only],
        [sparse_only],
    )

    assert [item.doc_id for item in merged] == ["bm25", "dense", "sparse"]
    assert {item.rrf_score for item in merged} == {
        round(1.0 / (tiered.settings.RRF_K + 1), 6)
    }


def test_dict_to_evidence_preserves_all_retrieval_fields():
    item = tiered._dict_to_evidence(
        [
            {
                "doc_id": 7,
                "text": "evidence",
                "source": "guide.pdf",
                "bm25_score": 4.0,
                "vector_score": 0.8,
                "sparse_score": 2.0,
                "rrf_score": 0.03,
                "generation": "rag-v9",
            }
        ],
        "case",
    )[0]

    assert item.doc_id == "7"
    assert item.bm25_score == 4.0
    assert item.vector_score == 0.8
    assert item.sparse_score == 2.0
    assert item.rrf_score == 0.03
    assert item.generation == "rag-v9"
