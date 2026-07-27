# -*- coding: utf-8 -*-
"""元数据增强 — 推荐等级/证据等级提取与增强 embedding 文本构建"""

import logging
import re

from app.services.rag.metadata_config import DocumentMetadata

logger = logging.getLogger(__name__)

# ── 推荐等级/证据等级提取 ──────────────────────────────────────────────────────────

# 推荐等级正则
_RECOMMENDATION_LEVEL_RE = re.compile(
    r'((?:I+|IV|V?I{1,3}|[一二三四五])级推荐|[ABC]级推荐|强推荐|弱推荐|条件性推荐)',
    re.IGNORECASE
)

# 证据等级正则
_EVIDENCE_LEVEL_RE = re.compile(
    r'(?:证据等级|证据级别|证据水平)[：:\s]*([1-5][ABC]?|[ABC])',
    re.IGNORECASE
)


def _extract_recommendation_level(text: str) -> str:
    """从 chunk 文本中提取推荐等级"""
    match = _RECOMMENDATION_LEVEL_RE.search(text)
    return match.group(1) if match else ""


def _extract_evidence_level(text: str) -> str:
    """从 chunk 文本中提取证据等级"""
    match = _EVIDENCE_LEVEL_RE.search(text)
    return match.group(1) if match else ""


def _build_embedding_text(chunk_text: str, doc_meta: DocumentMetadata, heading_path: str = "",
                          recommendation_level: str = "", evidence_level: str = "") -> str:
    """构建增强 embedding 文本，将元数据注入文本前缀

    格式：
    来源机构：CSCO
    指南：非小细胞肺癌诊疗指南
    版本：2025版
    章节：第三章 > 3.1 外科治疗
    推荐等级：I级推荐
    证据等级：1A
    正文：……
    """
    parts = []
    if doc_meta.organization:
        parts.append(f"来源机构：{doc_meta.organization}")
    if doc_meta.title:
        parts.append(f"指南：{doc_meta.title}")
    if doc_meta.version:
        parts.append(f"版本：{doc_meta.version}")
    if heading_path:
        parts.append(f"章节：{heading_path}")
    if recommendation_level:
        parts.append(f"推荐等级：{recommendation_level}")
    if evidence_level:
        parts.append(f"证据等级：{evidence_level}")
    parts.append(f"正文：{chunk_text}")
    return "\n".join(parts)
