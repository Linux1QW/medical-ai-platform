# -*- coding: utf-8 -*-
"""抽取式上下文压缩单测（context_compressor）

split_sentences 为纯函数；compress_text 通过 monkeypatch get_embeddings
注入确定性向量，验证 top-N 抽取、原序还原、阈值过滤与各降级路径。
"""

import pytest

import app.services.rag.context_compressor as cc
from app.services.rag.types import EvidenceItem


class TestSplitSentences:
    def test_basic_chinese(self):
        out = cc.split_sentences("第一句。第二句！第三句？")
        assert out == ["第一句。", "第二句！", "第三句？"]

    def test_tail_without_punct(self):
        out = cc.split_sentences("有标点。没标点结尾")
        assert out == ["有标点。", "没标点结尾"]

    def test_empty(self):
        assert cc.split_sentences("") == []
        assert cc.split_sentences("   ") == []


class TestCosine:
    def test_orthogonal(self):
        assert cc._cosine([1.0, 0.0], [0.0, 1.0]) == 0.0

    def test_identical(self):
        assert cc._cosine([1.0, 1.0], [1.0, 1.0]) == pytest.approx(1.0)

    def test_zero_vector(self):
        assert cc._cosine([0.0, 0.0], [1.0, 1.0]) == 0.0


def _patch_embeddings(monkeypatch, mapping):
    """mapping: text -> vector；未命中返回零向量"""
    async def fake_get_embeddings(texts):
        return [mapping.get(t, [0.0, 0.0, 0.0]) for t in texts]

    monkeypatch.setattr(cc, "get_embeddings", fake_get_embeddings)


@pytest.mark.asyncio
class TestCompressText:
    async def test_short_text_not_compressed(self, monkeypatch):
        _patch_embeddings(monkeypatch, {})
        text = "短文本。"
        out = await cc.compress_text(
            text, "查询", min_chars=400, top_sentences=3, min_score=0.1
        )
        assert out == text

    async def test_few_sentences_not_compressed(self, monkeypatch):
        _patch_embeddings(monkeypatch, {})
        text = "句一。句二。"
        out = await cc.compress_text(
            text, "查询", min_chars=1, top_sentences=3, min_score=0.0
        )
        assert out == text  # 句数 <= top_sentences

    async def test_keeps_top_and_preserves_order(self, monkeypatch):
        # 查询向量 [1,0,0]；相关句得高分，噪声句得低分
        q = "肺癌治疗"
        s1 = "肺癌一线治疗方案。"   # 高相关
        s2 = "无关内容甲。"          # 低相关
        s3 = "肺癌靶向药物推荐。"   # 高相关
        s4 = "无关内容乙。"          # 低相关
        text = s1 + s2 + s3 + s4
        _patch_embeddings(monkeypatch, {
            q: [1.0, 0.0, 0.0],
            s1: [1.0, 0.0, 0.0],
            s2: [0.0, 1.0, 0.0],
            s3: [0.9, 0.1, 0.0],
            s4: [0.0, 1.0, 0.0],
        })
        out = await cc.compress_text(
            text, q, min_chars=1, top_sentences=2, min_score=0.2
        )
        # 保留 s1、s3 且维持原文顺序
        assert out == s1 + s3

    async def test_all_below_threshold_returns_original(self, monkeypatch):
        q = "查询"
        s1 = "句一内容。"
        s2 = "句二内容。"
        s3 = "句三内容。"
        text = s1 + s2 + s3
        _patch_embeddings(monkeypatch, {
            q: [1.0, 0.0, 0.0],
            s1: [0.0, 1.0, 0.0],
            s2: [0.0, 1.0, 0.0],
            s3: [0.0, 1.0, 0.0],
        })
        out = await cc.compress_text(
            text, q, min_chars=1, top_sentences=2, min_score=0.5
        )
        assert out == text  # 全低于阈值 → 降级原文

    async def test_embedding_failure_returns_original(self, monkeypatch):
        async def boom(texts):
            raise RuntimeError("api down")

        monkeypatch.setattr(cc, "get_embeddings", boom)
        text = "句一。句二。句三。句四。"
        out = await cc.compress_text(
            text, "查询", min_chars=1, top_sentences=2, min_score=0.0
        )
        assert out == text

    async def test_empty_query_noop(self, monkeypatch):
        _patch_embeddings(monkeypatch, {})
        text = "句一。句二。句三。句四。"
        out = await cc.compress_text(
            text, "", min_chars=1, top_sentences=2, min_score=0.0
        )
        assert out == text


@pytest.mark.asyncio
class TestCompressEvidences:
    async def test_batch_compress(self, monkeypatch):
        q = "肺癌"
        s1 = "肺癌相关句。"
        s2 = "无关句。"
        s3 = "肺癌另一句。"
        _patch_embeddings(monkeypatch, {
            q: [1.0, 0.0],
            s1: [1.0, 0.0],
            s2: [0.0, 1.0],
            s3: [1.0, 0.0],
        })
        items = [EvidenceItem(doc_id="d1", text=s1 + s2 + s3, source="A")]
        out = await cc.compress_evidences(
            items, q, min_chars=1, top_sentences=2, min_score=0.2
        )
        assert out[0].text == s1 + s3

    async def test_empty_items(self, monkeypatch):
        _patch_embeddings(monkeypatch, {})
        assert await cc.compress_evidences(
            [], "q", min_chars=1, top_sentences=2, min_score=0.0
        ) == []
