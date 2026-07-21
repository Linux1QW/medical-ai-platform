# -*- coding: utf-8 -*-
"""Small-to-Big 上下文扩展单测（expand_context + fetch_neighbors）

通过 monkeypatch store.fetch_neighbors 隔离 ChromaDB IO，
验证邻居拼接顺序、缺失 chunk_seq 跳过、无邻居保持原样等逻辑。
"""

import app.services.rag.retriever as retriever_mod
from app.services.rag.types import EvidenceItem


def _mk(doc_id: str, source: str, chunk_seq, text: str) -> EvidenceItem:
    return EvidenceItem(doc_id=doc_id, text=text, source=source, chunk_seq=chunk_seq)


class _FakeStore:
    """按 (source, seq) -> {seq: {"text","page"}} 预置邻居映射的桩件"""

    def __init__(self, table):
        self.table = table
        self.calls = []

    def fetch_neighbors(self, source, chunk_seq, window=1):
        self.calls.append((source, chunk_seq, window))
        return self.table.get((source, chunk_seq), {})


def _patch_store(monkeypatch, store):
    monkeypatch.setattr(retriever_mod, "get_medical_store", lambda: store)


class TestExpandContext:
    def test_prepend_and_append_in_order(self, monkeypatch):
        # 中心 seq=5，邻居 4（前）、6（后）→ 拼成 前+中+后
        store = _FakeStore({
            ("指南A", 5): {
                4: {"text": "前块", "page": 1},
                6: {"text": "后块", "page": 1},
            }
        })
        _patch_store(monkeypatch, store)
        items = [_mk("d1", "指南A", 5, "中心块")]
        out = retriever_mod.expand_context(items, window=1)
        assert out[0].text == "前块\n中心块\n后块"

    def test_only_before_neighbor(self, monkeypatch):
        store = _FakeStore({("A", 2): {1: {"text": "前", "page": 1}}})
        _patch_store(monkeypatch, store)
        items = [_mk("d1", "A", 2, "中")]
        out = retriever_mod.expand_context(items, window=1)
        assert out[0].text == "前\n中"

    def test_no_neighbors_keeps_original(self, monkeypatch):
        store = _FakeStore({})  # fetch 恒返回 {}
        _patch_store(monkeypatch, store)
        items = [_mk("d1", "A", 3, "原文")]
        out = retriever_mod.expand_context(items, window=1)
        assert out[0].text == "原文"

    def test_missing_chunk_seq_skipped(self, monkeypatch):
        store = _FakeStore({})
        _patch_store(monkeypatch, store)
        items = [_mk("d1", "A", None, "原文")]
        out = retriever_mod.expand_context(items, window=1)
        assert out[0].text == "原文"
        assert store.calls == []  # 未触发查询

    def test_negative_chunk_seq_skipped(self, monkeypatch):
        store = _FakeStore({})
        _patch_store(monkeypatch, store)
        items = [_mk("d1", "A", -1, "原文")]
        out = retriever_mod.expand_context(items, window=1)
        assert out[0].text == "原文"
        assert store.calls == []

    def test_window_zero_noop(self, monkeypatch):
        store = _FakeStore({("A", 5): {4: {"text": "前", "page": 1}}})
        _patch_store(monkeypatch, store)
        items = [_mk("d1", "A", 5, "中")]
        out = retriever_mod.expand_context(items, window=0)
        assert out[0].text == "中"
        assert store.calls == []

    def test_empty_neighbor_text_ignored(self, monkeypatch):
        # 邻居块文本为空 → 不参与拼接
        store = _FakeStore({("A", 5): {4: {"text": "", "page": 1}}})
        _patch_store(monkeypatch, store)
        items = [_mk("d1", "A", 5, "中")]
        out = retriever_mod.expand_context(items, window=1)
        assert out[0].text == "中"

    def test_fetch_failure_keeps_original(self, monkeypatch):
        class _BoomStore:
            def fetch_neighbors(self, *a, **k):
                raise RuntimeError("boom")

        _patch_store(monkeypatch, _BoomStore())
        items = [_mk("d1", "A", 5, "原文")]
        out = retriever_mod.expand_context(items, window=1)
        assert out[0].text == "原文"

    def test_empty_items(self, monkeypatch):
        store = _FakeStore({})
        _patch_store(monkeypatch, store)
        assert retriever_mod.expand_context([], window=1) == []
