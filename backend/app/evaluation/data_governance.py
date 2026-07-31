# -*- coding: utf-8 -*-
"""Task 15 — 数据治理和部署安全

数据分级：
- P0 (raw): 原始患者对话，最短保留期，仅 admin 可访问
- P1 (masked): 脱敏对话，可进入 Langfuse/日志
- P2 (rubric): 评分和 rubric 指标，doctor 可访问
- P3 (aggregated): 聚合监控数据，所有角色可访问

核心接口：
- redact_trace(payload, target_level) -> dict
- purge_expired_trace(traces, now) -> list
- validate_export_scope(user, scope) -> None (raises PermissionError)
"""

from __future__ import annotations

import re
import logging
from dataclasses import dataclass
from datetime import datetime, timedelta
from enum import Enum
from typing import Any, Optional

logger = logging.getLogger(__name__)


# ── DataClassification ─────────────────────────────────────────────────────


class DataClassification(Enum):
    """数据分级"""

    P0_RAW = "p0_raw"              # 原始患者对话
    P1_MASKED = "p1_masked"        # 脱敏对话
    P2_RUBRIC = "p2_rubric"        # 评分/rubric 指标
    P3_AGGREGATED = "p3_aggregated"  # 聚合监控数据

    @property
    def level(self) -> int:
        return {
            DataClassification.P0_RAW: 0,
            DataClassification.P1_MASKED: 1,
            DataClassification.P2_RUBRIC: 2,
            DataClassification.P3_AGGREGATED: 3,
        }[self]


# ── RetentionPolicy ───────────────────────────────────────────────────────


@dataclass
class RetentionPolicy:
    """保留策略"""

    retention_days: int
    allowed_roles: list[str]


# 各级别保留策略
CLASSIFICATION_CONFIG: dict[DataClassification, RetentionPolicy] = {
    DataClassification.P0_RAW: RetentionPolicy(retention_days=7, allowed_roles=["admin"]),
    DataClassification.P1_MASKED: RetentionPolicy(retention_days=30, allowed_roles=["admin", "doctor"]),
    DataClassification.P2_RUBRIC: RetentionPolicy(retention_days=90, allowed_roles=["admin", "doctor", "viewer"]),
    DataClassification.P3_AGGREGATED: RetentionPolicy(retention_days=365, allowed_roles=["admin", "doctor", "viewer"]),
}


# ── 脱敏正则 ──────────────────────────────────────────────────────────────

_PHONE_RE = re.compile(r"(?<!\d)(1[3-9]\d)(\d{4})(\d{4})(?!\d)")
_ID_CARD_RE = re.compile(r"(?<!\d)(\d{6})(\d{8})(\d{3})([\dXx])(?!\d)")
_NAME_RE = re.compile(r"((?:姓名|患者)[：:\s]*)([\u4e00-\u9fff]{2,4})")


def _sanitize_text(text: str) -> str:
    """脱敏字符串"""
    text = _PHONE_RE.sub(r"\1****\3", text)
    text = _ID_CARD_RE.sub(r"\1********\3\4", text)
    text = _NAME_RE.sub(lambda m: m.group(1) + m.group(2)[0] + "*" * (len(m.group(2)) - 1), text)
    return text


# ── TraceRedactor ─────────────────────────────────────────────────────────


class TraceRedactor:
    """Trace 脱敏器"""

    @staticmethod
    def redact(data: Any, target_level: DataClassification) -> Any:
        """递归脱敏"""
        if data is None:
            return None
        if isinstance(data, str):
            return _sanitize_text(data)
        if isinstance(data, dict):
            return {k: TraceRedactor.redact(v, target_level) for k, v in data.items()}
        if isinstance(data, list):
            return [TraceRedactor.redact(item, target_level) for item in data]
        return data


def redact_trace(payload: Any, target_level: DataClassification) -> Any:
    """脱敏 trace 数据

    Args:
        payload: 原始数据
        target_level: 目标数据级别

    Returns:
        脱敏后的数据
    """
    if payload is None:
        return None
    return TraceRedactor.redact(payload, target_level)


# ── purge_expired_trace ───────────────────────────────────────────────────


def purge_expired_trace(
    traces: list[dict],
    now: Optional[datetime] = None,
) -> list[dict]:
    """清理过期 trace

    Args:
        traces: trace 列表，每项包含 created_at 和 classification
        now: 当前时间

    Returns:
        未过期的 trace 列表
    """
    if not traces:
        return []

    now = now or datetime.now()
    surviving = []

    for trace in traces:
        created_at = trace.get("created_at")
        if not created_at:
            surviving.append(trace)
            continue

        classification_str = trace.get("classification", "p1_masked")
        # 找到对应的 DataClassification
        level = None
        for dc in DataClassification:
            if dc.value == classification_str:
                level = dc
                break
        if level is None:
            level = DataClassification.P1_MASKED

        policy = CLASSIFICATION_CONFIG.get(level)
        if not policy:
            surviving.append(trace)
            continue

        expiry = created_at + timedelta(days=policy.retention_days)
        if now < expiry:
            surviving.append(trace)

    return surviving


# ── ExportValidator ────────────────────────────────────────────────────────


class ExportValidator:
    """导出权限验证"""

    @staticmethod
    def validate(user: dict, scope: DataClassification) -> None:
        """验证用户是否有权导出指定级别的数据

        Raises:
            PermissionError: 无权导出
        """
        role = user.get("role", "")
        policy = CLASSIFICATION_CONFIG.get(scope)
        if not policy:
            raise PermissionError(f"未知的数据级别: {scope}")
        if role not in policy.allowed_roles:
            raise PermissionError(
                f"角色 '{role}' 无权导出 {scope.value} 级别数据，"
                f"允许角色: {policy.allowed_roles}"
            )


def validate_export_scope(user: dict, scope: DataClassification) -> None:
    """验证导出权限"""
    ExportValidator.validate(user, scope)
