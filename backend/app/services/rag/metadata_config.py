# -*- coding: utf-8 -*-
"""医学文档元数据配置与解析

获取方式：文件名自动提取基础字段 → metadata_overrides.json 补充/覆盖 → Schema 校验
不确定的字段保持 None，禁止推测。
"""

import json
import logging
import re
from pathlib import Path
from typing import Optional
from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)

# 配置文件路径
OVERRIDES_PATH = Path(__file__).resolve().parent.parent.parent.parent / "data" / "metadata_overrides.json"


class DocumentMetadata(BaseModel):
    """文档级元数据"""
    title: str = ""
    organization: Optional[str] = None
    year: Optional[int] = None
    version: Optional[str] = None
    document_type: str = "unknown"           # clinical_guideline | textbook | reference_manual
    departments: list[str] = Field(default_factory=list)
    disease_tags: list[str] = Field(default_factory=list)
    population: list[str] = Field(default_factory=list)
    published_at: Optional[str] = None
    valid_until: Optional[str] = None
    status: str = "active"
    file_hash: Optional[str] = None
    index_version: str = "rag-v2"


# ── 文件名解析 ──

# 机构映射
ORG_PATTERNS = {
    "CSCO": "CSCO",
    "NCCN": "NCCN",
    "CACA": "CACA",
    "中华医学会": "中华医学会",
    "中国医师协会": "中国医师协会",
}

# 文档类型映射
DOC_TYPE_KEYWORDS = {
    "指南": "clinical_guideline",
    "规范": "clinical_guideline",
    "共识": "clinical_guideline",
    "教材": "textbook",
    "手册": "reference_manual",
    "分册": "textbook",
    "学": "textbook",       # 如 "内科学"、"外科学"
}


def parse_filename(filename: str) -> DocumentMetadata:
    """从文件名自动提取基础元数据

    示例：
    "2025CSCO非小细胞肺癌诊疗指南.pdf" → org=CSCO, year=2025, disease=非小细胞肺癌, type=clinical_guideline
    "内科学第10版.pdf" → type=textbook, title=内科学
    "NCCN中文华氏巨球蛋白血症淋巴浆细胞淋巴瘤.2025v2.pdf" → org=NCCN, year=2025
    """
    name = filename.replace(".pdf", "").replace(".PDF", "")
    meta = DocumentMetadata(title=name)

    # 1. 提取年份（4位数字，2000-2030范围）
    year_match = re.search(r'(20[0-2]\d|2030)', name)
    if year_match:
        meta.year = int(year_match.group())
        # 从标题中移除年份部分以获取更干净的标题
        name = name.replace(year_match.group(), "", 1).strip()

    # 2. 提取机构
    for pattern, org in ORG_PATTERNS.items():
        if pattern in name or pattern in filename:
            meta.organization = org
            # 从标题中移除机构名
            name = name.replace(pattern, "", 1).strip()
            break

    # 3. 提取文档类型
    for keyword, doc_type in DOC_TYPE_KEYWORDS.items():
        if keyword in name or keyword in filename:
            meta.document_type = doc_type
            break
    else:
        meta.document_type = "unknown"

    # 4. 提取版本号（如 "第10版"、"v2"、"V1"）
    version_match = re.search(r'(第\d+版|v\d+|V\d+)', filename)
    if version_match:
        meta.version = version_match.group()

    # 5. 清理标题
    # 移除常见后缀
    for suffix in ["诊疗指南", "临床实践指南", "中文版", "中文", "zh", ".pdf"]:
        name = name.replace(suffix, "")
    # 移除点号和多余空格
    name = re.sub(r'[.\s]+', ' ', name).strip()
    if name:
        meta.title = name

    return meta


def _load_overrides() -> dict:
    """加载 metadata_overrides.json"""
    if not OVERRIDES_PATH.exists():
        return {}
    try:
        with open(OVERRIDES_PATH, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception as e:
        logger.warning(f"加载元数据配置文件失败: {e}")
        return {}


def get_enriched_metadata(filename: str) -> DocumentMetadata:
    """获取增强元数据：文件名解析 + 配置文件覆盖 + Schema 校验

    Args:
        filename: PDF 文件名（如 "2025CSCO非小细胞肺癌诊疗指南.pdf"）

    Returns:
        DocumentMetadata 实例
    """
    # 1. 文件名自动解析
    meta = parse_filename(filename)

    # 2. 配置文件覆盖
    overrides = _load_overrides()
    if filename in overrides:
        override = overrides[filename]
        if "title" in override:
            meta.title = override["title"]
        if "organization" in override:
            meta.organization = override["organization"]
        if "year" in override:
            meta.year = override["year"]
        if "version" in override:
            meta.version = override["version"]
        if "document_type" in override:
            meta.document_type = override["document_type"]
        if "departments" in override:
            meta.departments = override["departments"]
        if "disease_tags" in override:
            meta.disease_tags = override["disease_tags"]
        if "population" in override:
            meta.population = override["population"]
        if "published_at" in override:
            meta.published_at = override["published_at"]
        if "valid_until" in override:
            meta.valid_until = override["valid_until"]
        if "status" in override:
            meta.status = override["status"]

    return meta
