# -*- coding: utf-8 -*-
"""医学指南索引构建 — 兼容层

实现已拆分至 app.services.rag.indexing 包：
- chunking:   语义分块与 chunk ID 生成
- extractors: 多格式文档抽取（PDF / Word）
- enrichment: 等级提取与增强 embedding 文本
- builder:    全量构建与单文件增量索引
- versioning: 索引版本切换与健康检查

本模块保留原 import 路径与 `python -m` 执行方式：
    cd backend; python -m app.services.rag.build_medical_index
"""

from app.services.rag.indexing import (
    CHUNK_OVERLAP,
    CHUNK_SIZE,
    HEADING_LEVELS,
    HEADING_REGEX,
    PDF_DIR,
    SENTENCE_END_PUNCT,
    SUPPORTED_EXTENSIONS,
    _build_embedding_text,
    _clean_source_name,
    _extract_chunks_from_page,
    _extract_evidence_level,
    _extract_recommendation_level,
    _health_check_index,
    _table_to_text,
    build_medical_index,
    chunk_text,
    extract_document,
    extract_text_from_docx,
    extract_text_from_pdf,
    generate_doc_id,
    generate_stable_chunk_id,
    get_indexed_sources,
    index_single_pdf,
    switch_index_version,
)

__all__ = [
    "CHUNK_OVERLAP",
    "CHUNK_SIZE",
    "HEADING_LEVELS",
    "HEADING_REGEX",
    "PDF_DIR",
    "SENTENCE_END_PUNCT",
    "SUPPORTED_EXTENSIONS",
    "_build_embedding_text",
    "_clean_source_name",
    "_extract_chunks_from_page",
    "_extract_evidence_level",
    "_extract_recommendation_level",
    "_health_check_index",
    "_table_to_text",
    "build_medical_index",
    "chunk_text",
    "extract_document",
    "extract_text_from_docx",
    "extract_text_from_pdf",
    "generate_doc_id",
    "generate_stable_chunk_id",
    "get_indexed_sources",
    "index_single_pdf",
    "switch_index_version",
]


if __name__ == "__main__":
    import asyncio
    import logging

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )
    asyncio.run(build_medical_index())
