# -*- coding: utf-8 -*-
"""RAG 数据契约模型测试"""

import pytest

from app.services.rag.types import (
    Citation,
    EvidenceItem,
    RetrievalBundle,
    RetrievalQuery,
)


class TestRetrievalQuery:
    def test_valid_case_query(self):
        q = RetrievalQuery(query_type="case", text="45岁男性咳嗽", source="clinical_facts")
        assert q.query_type == "case"
        assert q.source == "clinical_facts"

    def test_invalid_query_type(self):
        with pytest.raises(Exception):
            RetrievalQuery(query_type="invalid", text="test", source="clinical_facts")

    def test_invalid_source(self):
        with pytest.raises(Exception):
            RetrievalQuery(query_type="case", text="test", source="invalid_source")


class TestEvidenceItem:
    def test_minimal_evidence(self):
        e = EvidenceItem(doc_id="doc1", text="测试文本", source="test.pdf")
        assert e.page is None
        assert e.vector_score is None
        assert e.organization is None
        assert e.query_types == []

    def test_full_evidence(self):
        e = EvidenceItem(
            doc_id="doc1",
            text="测试文本",
            source="指南.pdf",
            page=42,
            heading_path="第三章 > 3.1",
            query_types=["case", "diagnosis"],
            vector_score=0.85,
            bm25_score=3.2,
            rrf_score=0.032,
            rerank_score=0.92,
            organization="CSCO",
            year=2025,
            authority_score=0.9,
            freshness_score=0.8,
        )
        assert e.organization == "CSCO"
        assert e.year == 2025
        assert len(e.query_types) == 2


class TestRetrievalBundle:
    def test_bundle_creation(self):
        bundle = RetrievalBundle(
            status="candidate",
            level_used="base",
            queries=[RetrievalQuery(query_type="case", text="test", source="clinical_facts")],
            candidates=[],
        )
        assert bundle.degraded is False
        assert bundle.trace == {}

    def test_bundle_degraded(self):
        bundle = RetrievalBundle(
            status="error",
            level_used="base",
            queries=[],
            candidates=[],
            degraded=True,
        )
        assert bundle.degraded is True


class TestCitation:
    def test_citation_creation(self):
        c = Citation(
            citation_id="rag-v2:test:p42:0",
            claim="测试声明",
            source="test.pdf",
            page=42,
            heading_path="第三章",
            text_snippet="证据文本",
            rerank_score=0.92,
        )
        assert c.citation_id == "rag-v2:test:p42:0"
