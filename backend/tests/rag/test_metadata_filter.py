# -*- coding: utf-8 -*-
"""Metadata 预过滤 where_document 构建单测（build_disease_where_document）

仅覆盖纯函数逻辑，不依赖 live 索引或 API。
"""

from app.services.rag.retriever import build_disease_where_document


def _iter_contains(where):
    """展开 where_document，逐一 yield 每个 $contains 子串"""
    if where is None:
        return
    if "$contains" in where:
        yield where["$contains"]
    elif "$or" in where:
        for clause in where["$or"]:
            yield clause["$contains"]


class TestBuildDiseaseWhereDocument:
    def test_empty_query(self):
        assert build_disease_where_document("") is None

    def test_whitespace_query(self):
        assert build_disease_where_document("   ") is None

    def test_no_disease_entity(self):
        # 纯寒暄/无医学实体 → 不过滤
        assert build_disease_where_document("今天天气不错适合散步") is None

    def test_single_disease_shape(self):
        where = build_disease_where_document("乳腺癌新辅助治疗方案")
        assert where is not None
        substrs = list(_iter_contains(where))
        assert len(substrs) >= 1
        assert all(isinstance(s, str) and s for s in substrs)
        # 期望命中乳腺癌相关实体
        assert any("乳腺癌" in s for s in substrs)

    def test_or_structure_valid(self):
        # 含多个疾病实体时应产出 $or 或 $contains，结构须合法
        where = build_disease_where_document("非小细胞肺癌 EGFR 突变靶向治疗")
        assert where is not None
        assert "$contains" in where or "$or" in where
        if "$or" in where:
            assert isinstance(where["$or"], list)
            assert all("$contains" in c for c in where["$or"])
        substrs = list(_iter_contains(where))
        assert any("肺癌" in s for s in substrs)
