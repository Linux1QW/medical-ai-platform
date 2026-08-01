# -*- coding: utf-8 -*-
"""Task 11 — 全链路 Trace 和观测

TraceContext 统一承载 trace_id、run_id、consultation_id、celery_task_id、
graph_thread_id 和 attempt，通过 contextvars 在 FastAPI → Celery → Graph → Agent
全链路传播。

脱敏函数 sanitize_for_observability 确保 Langfuse / Prometheus / 日志中
不出现未脱敏的患者姓名、电话、身份证号等 PII 数据。
"""

from __future__ import annotations

import logging
import re
import uuid
from contextvars import ContextVar, Token
from dataclasses import dataclass, field
from typing import Any, Optional

logger = logging.getLogger(__name__)


# ── ContextVar ─────────────────────────────────────────────────────────────

_current_trace_context: ContextVar[Optional["TraceContext"]] = ContextVar(
    "current_trace_context", default=None
)


# ── TraceContext ───────────────────────────────────────────────────────────


def new_trace_id() -> str:
    """生成新的 trace ID"""
    return f"trace-{uuid.uuid4().hex[:24]}"


@dataclass
class TraceContext:
    """评估全链路追踪上下文

    Attributes:
        trace_id: 全局唯一追踪 ID（自动生成）
        run_id: 评估 run ID
        consultation_id: 问诊 ID
        celery_task_id: Celery 任务 ID
        graph_thread_id: LangGraph 线程 ID
        attempt: 重试次数（首次为 1）
        node_name: 当前 graph node 名称
        agent_name: 当前 agent 名称
        tool_name: 当前 tool 名称
    """

    trace_id: str = field(default_factory=new_trace_id)
    run_id: Optional[str] = None
    consultation_id: Optional[str] = None
    celery_task_id: Optional[str] = None
    graph_thread_id: Optional[str] = None
    attempt: int = 1
    node_name: Optional[str] = None
    agent_name: Optional[str] = None
    tool_name: Optional[str] = None


# ── 上下文绑定 ─────────────────────────────────────────────────────────────


def bind_trace_context(ctx: TraceContext) -> Token:
    """绑定 trace context 到当前 asyncio task / 线程

    Returns:
        Token 用于后续恢复
    """
    return _current_trace_context.set(ctx)


def get_current_trace_context() -> Optional[TraceContext]:
    """获取当前 trace context"""
    return _current_trace_context.get()


def reset_trace_context(token: Token) -> None:
    """恢复之前的 trace context"""
    _current_trace_context.reset(token)


# ── 序列化 / 反序列化 ─────────────────────────────────────────────────────


def serialize_trace_context(ctx: TraceContext) -> dict:
    """序列化 trace context 为 dict（用于 Celery payload / 日志）"""
    return {
        "trace_id": ctx.trace_id,
        "run_id": ctx.run_id,
        "consultation_id": ctx.consultation_id,
        "celery_task_id": ctx.celery_task_id,
        "graph_thread_id": ctx.graph_thread_id,
        "attempt": ctx.attempt,
        "node_name": ctx.node_name,
        "agent_name": ctx.agent_name,
        "tool_name": ctx.tool_name,
    }


def restore_trace_context(payload: dict) -> TraceContext:
    """从 dict 恢复 trace context

    缺失字段使用默认值，不抛异常。
    """
    if not payload:
        return TraceContext()

    return TraceContext(
        trace_id=payload.get("trace_id") or new_trace_id(),
        run_id=payload.get("run_id"),
        consultation_id=payload.get("consultation_id"),
        celery_task_id=payload.get("celery_task_id"),
        graph_thread_id=payload.get("graph_thread_id"),
        attempt=payload.get("attempt", 1),
        node_name=payload.get("node_name"),
        agent_name=payload.get("agent_name"),
        tool_name=payload.get("tool_name"),
    )


# ── 脱敏函数 ──────────────────────────────────────────────────────────────

# 手机号：中国大陆 11 位
_PHONE_RE = re.compile(r"(?<!\d)(1[3-9]\d)(\d{4})(\d{4})(?!\d)")

# 身份证号：18 位
_ID_CARD_RE = re.compile(r"(?<!\d)(\d{6})(\d{8})(\d{3})([\dXx])(?!\d)")

# 中文姓名：2-4 个汉字（简单启发式，可能误伤，但安全优先）
# 匹配 "姓名" / "患者" 后的 2-4 个汉字
_NAME_RE = re.compile(r"((?:姓名|患者)[：:\s]*)([\u4e00-\u9fff]{2,4})")


def _sanitize_text(text: str) -> str:
    """脱敏单个字符串"""
    # 手机号：138****5678
    text = _PHONE_RE.sub(r"\1****\3", text)
    # 身份证号：110101********1234
    text = _ID_CARD_RE.sub(r"\1********\3\4", text)
    # 姓名：张**
    text = _NAME_RE.sub(lambda m: m.group(1) + m.group(2)[0] + "*" * (len(m.group(2)) - 1), text)
    return text


def sanitize_for_observability(data: Any) -> Any:
    """递归脱敏：确保 Langfuse / 日志 / Prometheus 中不含 PII

    支持 str / dict / list / None，其他类型原样返回。
    """
    if data is None:
        return None
    if isinstance(data, str):
        return _sanitize_text(data)
    if isinstance(data, dict):
        return {k: sanitize_for_observability(v) for k, v in data.items()}
    if isinstance(data, list):
        return [sanitize_for_observability(item) for item in data]
    # int / float / bool 等原样返回
    return data
