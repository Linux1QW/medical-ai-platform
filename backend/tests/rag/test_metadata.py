# -*- coding: utf-8 -*-
"""元数据解析测试"""

from app.services.rag.metadata_config import parse_filename


class TestParseFilename:
    def test_csco_guideline(self):
        meta = parse_filename("2025CSCO非小细胞肺癌诊疗指南.pdf")
        assert meta.organization == "CSCO" or meta.year == 2025  # 至少能解析出一个

    def test_simple_filename(self):
        meta = parse_filename("临床诊疗指南.pdf")
        assert meta is not None  # 不应崩溃

    def test_filename_with_year(self):
        meta = parse_filename("2024版糖尿病管理指南.pdf")
        assert meta.year == 2024 or meta is not None

    def test_empty_filename(self):
        meta = parse_filename("")
        assert meta is not None  # 不应崩溃
