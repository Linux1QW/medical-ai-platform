# -*- coding: utf-8 -*-
"""稳定 Citation ID 和来源注册表 — KB 重建后仍可定位证据。

核心语义：
- citation ID = hash(kb_version + document_id + chunk_id + content_hash)
- 来源注册表记录类型、权威等级、发布/生效日期
- 旧 citation 保存 legacy_citation_id

用法：
    from evaluation.citation_registry import stable_citation_id, load_source_registry

    cid = stable_citation_id("rag-v1", "doc_001", "chunk_5", "abc123")
    registry = load_source_registry()
"""
from __future__ import annotations

import hashlib
import json
from enum import Enum
from pathlib import Path
from typing import Optional

from pydantic import BaseModel, field_validator

# ── 来源权威等级 ─────────────────────────────────────────────────────────────


class SourceAuthority(str, Enum):
    """来源权威等级。"""

    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"


# ── SourceMetadata ───────────────────────────────────────────────────────────


class SourceMetadata(BaseModel):
    """来源元数据。"""

    source_id: str
    source_type: str  # guideline / textbook / paper / dataset
    authority: SourceAuthority
    title: str
    effective_date: Optional[str] = None
    expiry_date: Optional[str] = None
    page_offset: Optional[int] = None

    @field_validator("authority", mode="before")
    @classmethod
    def _validate_authority(cls, v: str) -> str:
        valid = {k.value for k in SourceAuthority}
        if v not in valid:
            raise ValueError(f"非法 authority: {v!r}，合法值: {sorted(valid)}")
        return v


# ── 稳定 Citation ID ─────────────────────────────────────────────────────────


def stable_citation_id(
    kb_version: str,
    document_id: str,
    chunk_id: str,
    content_hash: str,
) -> str:
    """生成稳定的 citation ID。

    使用 kb_version + document_id + chunk_id + content_hash 的 hash，
    不依赖列表 index，KB 重建后仍可定位。

    Args:
        kb_version: KB 版本。
        document_id: 文档 ID。
        chunk_id: chunk ID。
        content_hash: 内容 hash。

    Returns:
        确定性 citation ID 字符串。
    """
    raw = f"{kb_version}|{document_id}|{chunk_id}|{content_hash}"
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()[:16]


# ── 来源注册表 ───────────────────────────────────────────────────────────────

_REGISTRY_PATH = Path(__file__).parent / "source_registry.json"


def load_source_registry(path: Path | None = None) -> dict[str, dict]:
    """加载来源注册表。

    Args:
        path: 自定义路径，默认使用内置 source_registry.json。

    Returns:
        {source_id: metadata_dict} 字典。
    """
    if path is None:
        path = _REGISTRY_PATH
    if not path.exists():
        return {}
    data = json.loads(path.read_text(encoding="utf-8"))
    return data.get("sources", {})


def resolve_source_metadata(
    source_id: str,
    registry: dict[str, dict] | None = None,
) -> SourceMetadata | None:
    """解析来源元数据。

    Args:
        source_id: 来源 ID。
        registry: 注册表字典，缺省则加载内置。

    Returns:
        SourceMetadata 或 None（未找到）。
    """
    if registry is None:
        registry = load_source_registry()
    raw = registry.get(source_id)
    if raw is None:
        return None
    return SourceMetadata(**raw)
