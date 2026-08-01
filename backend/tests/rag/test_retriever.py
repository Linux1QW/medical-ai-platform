# -*- coding: utf-8 -*-
"""检索模块集成测试（需要知识库数据）"""

from unittest.mock import AsyncMock, Mock

import pytest

from app.services.rag.retriever import fusion

from app.services.rag.types import EvidenceItem


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
