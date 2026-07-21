# -*- coding: utf-8 -*-
"""结果多样性重排单测（rerank_with_diversity）

纯函数、无 IO；验证来源配额约束与不漏召回退逻辑。
"""

from app.services.rag.reranker import rerank_with_diversity
from app.services.rag.types import EvidenceItem


def _mk(doc_id: str, source: str) -> EvidenceItem:
    return EvidenceItem(doc_id=doc_id, text=f"内容-{doc_id}", source=source)


def _sources(items):
    return [it.source for it in items]


class TestRerankWithDiversity:
    def test_cap_limits_single_source(self):
        # 4 条同源 + 2 条他源，cap=2，top_k=4 → A 最多 2 条
        docs = [
            _mk("a1", "指南A"), _mk("a2", "指南A"), _mk("a3", "指南A"),
            _mk("a4", "指南A"), _mk("b1", "指南B"), _mk("c1", "指南C"),
        ]
        out = rerank_with_diversity(docs, top_k=4, per_source_cap=2)
        srcs = _sources(out)
        assert len(out) == 4
        assert srcs.count("指南A") == 2
        assert "指南B" in srcs and "指南C" in srcs

    def test_preserves_relevance_order(self):
        # 多样性优先，但同源内部保持原相关性顺序
        docs = [_mk("a1", "A"), _mk("a2", "A"), _mk("b1", "B")]
        out = rerank_with_diversity(docs, top_k=3, per_source_cap=1)
        # cap=1：先 a1、b1 入选，a2 延后补齐
        assert [it.doc_id for it in out] == ["a1", "b1", "a2"]

    def test_backfill_when_insufficient(self):
        # 全部同源，cap=1 但 top_k=3 → 不足时补齐，保证不漏召
        docs = [_mk("a1", "A"), _mk("a2", "A"), _mk("a3", "A")]
        out = rerank_with_diversity(docs, top_k=3, per_source_cap=1)
        assert len(out) == 3
        assert [it.doc_id for it in out] == ["a1", "a2", "a3"]

    def test_empty_input(self):
        assert rerank_with_diversity([], top_k=5, per_source_cap=2) == []

    def test_zero_cap_returns_topk(self):
        docs = [_mk("a1", "A"), _mk("a2", "A")]
        out = rerank_with_diversity(docs, top_k=1, per_source_cap=0)
        assert [it.doc_id for it in out] == ["a1"]

    def test_topk_truncation(self):
        docs = [_mk("a1", "A"), _mk("b1", "B"), _mk("c1", "C")]
        out = rerank_with_diversity(docs, top_k=2, per_source_cap=5)
        assert len(out) == 2
        assert [it.doc_id for it in out] == ["a1", "b1"]

    def test_none_source_treated_as_unknown(self):
        docs = [_mk("a1", ""), _mk("a2", ""), _mk("b1", "B")]
        out = rerank_with_diversity(docs, top_k=3, per_source_cap=1)
        # 两个空来源归为"未知"，cap=1 → 仅 a1 入选，a2 延后
        assert [it.doc_id for it in out] == ["a1", "b1", "a2"]
