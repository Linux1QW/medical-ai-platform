"""评估任务派发 Outbox 模型 — Transactional Outbox 保证最终投递

DispatchStatus: pending → leased → published → (terminal: cancelled / dead_letter)
"""

from __future__ import annotations

import re
from datetime import datetime
from typing import Any, Optional

from sqlalchemy import DateTime, Index, Integer, JSON, String, Text, ForeignKey
from sqlalchemy.orm import Mapped, mapped_column

from app.models.base import Base

# Payload 字段 allowlist — 只允许这些 key 出现在 outbox payload 中
_PAYLOAD_KEY_ALLOWLIST = frozenset({"run_id", "consultation_id", "trace_context"})

# trace_context 内允许的子字段（去标识化）
_TRACE_CONTEXT_ALLOWLIST = frozenset({"trace_id", "span_id", "sampled"})

# PII / 敏感字段黑名单（用于 trace_context 子字段校验）
_PII_FIELD_NAMES = re.compile(
    r"(patient|name|dialog|conversation|prompt|token|message|content)",
    re.IGNORECASE,
)


def validate_payload(payload: dict[str, Any]) -> None:
    """校验 payload 只包含 allowlist 字段，拒绝 PII。

    Raises:
        ValueError: 包含不允许的字段时
    """
    if not isinstance(payload, dict):
        raise ValueError("payload must be a dict")

    extra_keys = set(payload.keys()) - _PAYLOAD_KEY_ALLOWLIST
    if extra_keys:
        raise ValueError(
            f"payload contains disallowed keys: {extra_keys}. "
            f"Allowed: {_PAYLOAD_KEY_ALLOWLIST}"
        )

    # 校验 trace_context 子字段
    trace_ctx = payload.get("trace_context", {})
    if isinstance(trace_ctx, dict):
        for key in trace_ctx:
            if key not in _TRACE_CONTEXT_ALLOWLIST:
                raise ValueError(
                    f"trace_context field '{key}' not allowed. "
                    f"Allowed: {_TRACE_CONTEXT_ALLOWLIST}"
                )
            if _PII_FIELD_NAMES.search(key):
                raise ValueError(
                    f"trace_context field '{key}' looks like PII and is not allowed"
                )


class EvaluationDispatchOutbox(Base):
    """评估任务派发 Outbox 表

    与 EvaluationRun 在同一事务中写入，通过 outbox pattern 保证
    评估任务最终被投递到 Celery broker。
    """

    __tablename__ = "evaluation_dispatch_outbox"

    # 主键：event UUID
    event_id: Mapped[str] = mapped_column(
        String(36), primary_key=True, comment="Outbox event UUID"
    )

    # 关联 run（1:1），CASCADE 删除
    run_id: Mapped[str] = mapped_column(
        String(36),
        ForeignKey("evaluation_runs.id", ondelete="CASCADE"),
        nullable=False,
        unique=True,
        comment="关联的评估运行 ID",
    )

    # 派发状态
    status: Mapped[str] = mapped_column(
        String(20),
        nullable=False,
        default="pending",
        server_default="pending",
        comment="pending | leased | published | cancelled | dead_letter",
    )

    # Celery task 名称
    task_name: Mapped[str] = mapped_column(
        String(100), nullable=False, comment="Celery task 名称"
    )

    # 投递 payload（JSON，只允许 allowlist 字段）
    payload: Mapped[dict] = mapped_column(
        JSON, nullable=False, comment="投递 payload（去标识化）"
    )

    # 重试计数
    attempt: Mapped[int] = mapped_column(
        Integer, nullable=False, default=0, server_default="0", comment="已尝试次数"
    )

    # 下次重试时间
    next_attempt_at: Mapped[Optional[datetime]] = mapped_column(
        DateTime, nullable=True, comment="下次重试时间"
    )

    # 租约字段
    lease_owner: Mapped[Optional[str]] = mapped_column(
        String(160), nullable=True, comment="当前租约持有者 (hostname:pid:uuid)"
    )
    lease_expires_at: Mapped[Optional[datetime]] = mapped_column(
        DateTime, nullable=True, comment="租约过期时间"
    )

    # 错误追踪
    last_error_code: Mapped[Optional[str]] = mapped_column(
        String(100), nullable=True, comment="最近错误码"
    )
    last_task_id: Mapped[Optional[str]] = mapped_column(
        String(36), nullable=True, comment="最近 Celery task ID"
    )

    # 时间戳
    created_at: Mapped[datetime] = mapped_column(
        DateTime, nullable=False, default=datetime.utcnow, comment="创建时间"
    )
    updated_at: Mapped[datetime] = mapped_column(
        DateTime,
        nullable=False,
        default=datetime.utcnow,
        onupdate=datetime.utcnow,
        comment="更新时间",
    )
    published_at: Mapped[Optional[datetime]] = mapped_column(
        DateTime, nullable=True, comment="成功投递时间"
    )

    __table_args__ = (
        # 复合索引：按状态 + 下次重试时间查询可 claim 的行
        Index("ix_dispatch_outbox_status_next_attempt", "status", "next_attempt_at"),
        # 租约过期索引：用于回收过期租约
        Index("ix_dispatch_outbox_lease_expires", "lease_expires_at"),
        # 创建时间索引：用于 retention 清理
        Index("ix_dispatch_outbox_created_at", "created_at"),
    )
