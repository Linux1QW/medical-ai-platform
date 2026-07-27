# -*- coding: utf-8 -*-
"""医学指南索引构建包 — 统一流水线：PDF/Word → Markdown → 语义切分 → 向量化存储

子模块划分：
- chunking:   语义分块（标题层级/句子边界/重叠）与 chunk ID 生成
- extractors: 多格式文档抽取（PDF / Word），含表格线性化
- enrichment: 推荐等级/证据等级提取与增强 embedding 文本构建
- builder:    全量构建与单文件增量索引
- versioning: 索引版本切换、健康检查与自动回滚
"""

from app.services.rag.indexing.builder import (
    PDF_DIR,
    build_medical_index,
    get_indexed_sources,
    index_single_pdf,
)
from app.services.rag.indexing.chunking import (
    CHUNK_OVERLAP,
    CHUNK_SIZE,
    HEADING_LEVELS,
    HEADING_REGEX,
    SENTENCE_END_PUNCT,
    _clean_source_name,
    _extract_chunks_from_page,
    chunk_text,
    generate_doc_id,
    generate_stable_chunk_id,
)
from app.services.rag.indexing.enrichment import (
    _build_embedding_text,
    _extract_evidence_level,
    _extract_recommendation_level,
)
from app.services.rag.indexing.extractors import (
    SUPPORTED_EXTENSIONS,
    _table_to_text,
    extract_document,
    extract_text_from_docx,
    extract_text_from_pdf,
)
from app.services.rag.indexing.versioning import (
    _health_check_index,
    switch_index_version,
)

__all__ = [
    # builder
    "PDF_DIR",
    "build_medical_index",
    "get_indexed_sources",
    "index_single_pdf",
    # chunking
    "CHUNK_OVERLAP",
    "CHUNK_SIZE",
    "HEADING_LEVELS",
    "HEADING_REGEX",
    "SENTENCE_END_PUNCT",
    "_clean_source_name",
    "_extract_chunks_from_page",
    "chunk_text",
    "generate_doc_id",
    "generate_stable_chunk_id",
    # enrichment
    "_build_embedding_text",
    "_extract_evidence_level",
    "_extract_recommendation_level",
    # extractors
    "SUPPORTED_EXTENSIONS",
    "_table_to_text",
    "extract_document",
    "extract_text_from_docx",
    "extract_text_from_pdf",
    # versioning
    "_health_check_index",
    "switch_index_version",
]
