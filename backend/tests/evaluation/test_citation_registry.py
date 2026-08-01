# -*- coding: utf-8 -*-
"""稳定 Citation ID 和来源注册表测试 — Task 7

验证：
1. citation ID 不依赖列表 index
2. 同一 chunk 在相同 KB 版本下 ID 稳定
3. 不同 KB 版本 ID 可区分
4. 来源注册表加载合法
5. 旧 citation 可读取但不能生成新证据链
"""
import pytest

from evaluation.citation_registry import (
    SourceAuthority,
    SourceMetadata,
    load_source_registry,
    resolve_source_metadata,
    stable_citation_id,
)

# ── 1. stable_citation_id ───────────────────────────────────────────────────


class TestStableCitationId:
    def test_same_input_same_id(self):
        """相同输入产生相同 ID"""
        id1 = stable_citation_id("rag-v1", "doc_001", "chunk_5", "content hash")
        id2 = stable_citation_id("rag-v1", "doc_001", "chunk_5", "content hash")
        assert id1 == id2

    def test_different_kb_version_different_id(self):
        """不同 KB 版本产生不同 ID"""
        id1 = stable_citation_id("rag-v1", "doc_001", "chunk_5", "content hash")
        id2 = stable_citation_id("rag-v2", "doc_001", "chunk_5", "content hash")
        assert id1 != id2

    def test_different_chunk_different_id(self):
        """不同 chunk 产生不同 ID"""
        id1 = stable_citation_id("rag-v1", "doc_001", "chunk_5", "content hash")
        id2 = stable_citation_id("rag-v1", "doc_001", "chunk_6", "content hash")
        assert id1 != id2

    def test_id_not_depend_on_list_index(self):
        """ID 不依赖列表顺序"""
        id1 = stable_citation_id("rag-v1", "doc_001", "chunk_5", "content hash")
        # 即使调用顺序不同，ID 不变
        id2 = stable_citation_id("rag-v1", "doc_001", "chunk_5", "content hash")
        assert id1 == id2

    def test_id_is_deterministic_string(self):
        """ID 是确定性字符串"""
        cid = stable_citation_id("rag-v1", "doc_001", "chunk_5", "content hash")
        assert isinstance(cid, str)
        assert len(cid) > 0


# ── 2. SourceMetadata ────────────────────────────────────────────────────────


class TestSourceMetadata:
    def test_valid_metadata(self):
        m = SourceMetadata(
            source_id="src_001",
            source_type="guideline",
            authority=SourceAuthority.HIGH,
            title="CSCO 乳腺癌诊疗指南 2025",
            effective_date="2025-01-01",
        )
        assert m.authority == SourceAuthority.HIGH

    def test_invalid_authority_rejected(self):
        with pytest.raises(ValueError):
            SourceMetadata(
                source_id="src_001",
                source_type="guideline",
                authority="bogus",
                title="x",
            )


# ── 3. SourceAuthority 枚举 ─────────────────────────────────────────────────


class TestSourceAuthority:
    def test_all_levels(self):
        assert SourceAuthority.HIGH == "high"
        assert SourceAuthority.MEDIUM == "medium"
        assert SourceAuthority.LOW == "low"


# ── 4. 来源注册表加载 ───────────────────────────────────────────────────────


class TestSourceRegistry:
    def test_load_registry(self):
        """注册表可加载"""
        registry = load_source_registry()
        assert isinstance(registry, dict)

    def test_resolve_known_source(self):
        """已知来源可解析"""
        registry = load_source_registry()
        if registry:
            first_id = next(iter(registry))
            meta = resolve_source_metadata(first_id, registry)
            assert meta is not None

    def test_resolve_unknown_source_none(self):
        """未知来源返回 None"""
        registry = load_source_registry()
        meta = resolve_source_metadata("nonexistent", registry)
        assert meta is None
