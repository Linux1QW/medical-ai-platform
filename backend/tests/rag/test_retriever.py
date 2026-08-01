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
    monkeypatch.setattr(fusion, "get_bm25_index", lambda: bm25_index)
    monkeypatch.setattr(fusion.settings, "BGE_M3_ENABLED", False)

    await fusion.hybrid_recall("心梗治疗", top_k=5)

    dense.assert_awaited_once_with("心梗治疗", top_k=15)
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
    monkeypatch.setattr(tiered, "get_cached_bundle", AsyncMock(return_value=None))
    monkeypatch.setattr(tiered, "set_cached_bundle", AsyncMock())
    monkeypatch.setattr(tiered, "hybrid_recall", hybrid_recall)
    monkeypatch.setattr(
        tiered,
        "_assess_confidence",
        lambda **kwargs: RetrievalConfidence.HIGH,
    )

    await tiered.tiered_retrieve(
        [RetrievalQuery(query_type="case", text=original_text)],
        top_k_per_query=5,
    )

    hybrid_recall.assert_awaited_once_with(original_text, top_k=5)
