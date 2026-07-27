# -*- coding: utf-8 -*-
"""标准化失败原因码 — 评估运行失败打标

将异常归类为稳定的原因码写入 EvaluationRun.error_type，
替代裸异常类名，便于监控聚合与告警规则配置。
原类名与消息保留在 error_message 中，不丢失原始信息。
"""

import asyncio

# 原因码常量（小写蛇形，与 /metrics 标签风格一致）
REASON_TIMEOUT = "timeout"
REASON_CONNECTION_ERROR = "connection_error"
REASON_RATE_LIMITED = "rate_limited"
REASON_GRAPH_NOT_INITIALIZED = "graph_not_initialized"
REASON_DB_ERROR = "db_error"
REASON_VALIDATION_ERROR = "validation_error"
REASON_CANCELLED = "cancelled"
REASON_UNKNOWN = "unknown_error"

ALL_REASONS = frozenset({
    REASON_TIMEOUT,
    REASON_CONNECTION_ERROR,
    REASON_RATE_LIMITED,
    REASON_GRAPH_NOT_INITIALIZED,
    REASON_DB_ERROR,
    REASON_VALIDATION_ERROR,
    REASON_CANCELLED,
    REASON_UNKNOWN,
})


class EvaluationDeadlineExceeded(TimeoutError):
    """评估 run 超出总时长预算（评估级 deadline，classify_failure 归为 timeout）"""


def classify_failure(exc: BaseException) -> str:
    """将异常映射为标准化失败原因码

    分类优先级：取消 > 超时 > 限流 > 图未初始化 > 数据库 > 连接 > 校验 > 未知。
    类名关键词判断与 evaluation_task._is_retryable_error 保持同一套词表。
    """
    exc_name = type(exc).__name__.lower()
    exc_module = type(exc).__module__ or ""
    message = str(exc)

    if isinstance(exc, asyncio.CancelledError):
        return REASON_CANCELLED
    if isinstance(exc, TimeoutError) or "timeout" in exc_name:
        return REASON_TIMEOUT
    if "ratelimit" in exc_name or "rate_limit" in exc_name:
        return REASON_RATE_LIMITED
    if "图未初始化" in message:
        return REASON_GRAPH_NOT_INITIALIZED
    if exc_module.startswith("sqlalchemy") or exc_module.startswith("pymysql"):
        return REASON_DB_ERROR
    if isinstance(exc, (ConnectionError, OSError)) or any(
        kw in exc_name for kw in ("connection", "network", "unavailable")
    ):
        return REASON_CONNECTION_ERROR
    if isinstance(exc, (ValueError, TypeError)) or "validation" in exc_name:
        return REASON_VALIDATION_ERROR
    return REASON_UNKNOWN
