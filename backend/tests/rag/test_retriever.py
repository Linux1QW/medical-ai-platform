# -*- coding: utf-8 -*-
"""检索模块集成测试（需要知识库数据）"""

import pytest
from app.services.rag.types import RetrievalQuery, EvidenceItem


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
            MAX_MQE_EXPANSIONS,
            MAX_HYDE_CALLS,
            MAX_RAG_CANDIDATES,
        )
        assert MAX_MQE_EXPANSIONS == 2
        assert MAX_HYDE_CALLS == 1
        assert MAX_RAG_CANDIDATES == 20
