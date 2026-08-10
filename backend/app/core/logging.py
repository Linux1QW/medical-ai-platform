"""
结构化 JSON 日志配置（Task 7 隐私安全版）

通过 setup_logging() 初始化日志系统：
- LOG_FORMAT=json（默认）：使用 python-json-logger 输出 JSON 格式
- LOG_FORMAT=text：使用人类可读的传统文本格式
- LOG_LEVEL 控制日志级别（默认 INFO）

Production/Staging 隐私策略：
- 只允许 safe_log_event() 的结构化事件
- allowlist: request_id/trace_id/run_id/consultation_id/agent_name/tool_name/status/error_code/duration_ms/count
- 未带 safe_structured=true 的 LogRecord 被替换为固定 "unsafe_log_message_suppressed"
"""

import logging
import sys
from datetime import datetime, timezone
from typing import Any

from pythonjsonlogger.json import JsonFormatter as _BaseJsonFormatter

from app.core.config import settings

# ── Allowlist 字段 ────────────────────────────────────────────────────────────

_SAFE_LOG_FIELDS = {
    "request_id",
    "trace_id",
    "run_id",
    "consultation_id",
    "agent_name",
    "tool_name",
    "status",
    "error_code",
    "duration_ms",
    "count",
    "event_code",
    "safe_structured",
}


# ── Formatter ─────────────────────────────────────────────────────────────────


class CustomJsonFormatter(_BaseJsonFormatter):
    """
    自定义 JSON Formatter，注入默认字段。
    保留字段：request_id, method, path, status, duration_ms（由调用方通过 extra 传入）
    默认字段：service, environment, timestamp, level
    """

    def add_fields(self, log_record, record, message_dict):
        super().add_fields(log_record, record, message_dict)
        # 注入默认字段
        log_record.setdefault("service", "medical-ai-platform")
        log_record.setdefault("environment", settings.ENVIRONMENT)
        # 使用 ISO 8601 UTC 时间戳
        log_record["timestamp"] = datetime.fromtimestamp(record.created, tz=timezone.utc).isoformat()
        log_record["level"] = record.levelname


class SafeStructuredFormatter(CustomJsonFormatter):
    """Production/Staging 专用 Formatter：只输出 allowlist 字段"""

    def add_fields(self, log_record, record, message_dict):
        super().add_fields(log_record, record, message_dict)
        # 如果不是 safe_structured 事件，替换为 suppression 消息
        if not getattr(record, "safe_structured", False):
            # 清除所有可能包含敏感信息的字段
            log_record["message"] = "unsafe_log_message_suppressed"
            log_record["msg"] = "unsafe_log_message_suppressed"
            # 移除 args 和 exc_info（可能包含敏感信息）
            if "args" in log_record:
                log_record["args"] = "suppressed"
            if "exc_info" in log_record:
                log_record["exc_info"] = "suppressed"
            if "stack_info" in log_record:
                log_record["stack_info"] = "suppressed"
            # 只保留 allowlist 字段
            keys_to_remove = [k for k in log_record.keys() if k not in _SAFE_LOG_FIELDS and k not in {
                "service", "environment", "timestamp", "level", "message", "msg", "name"
            }]
            for k in keys_to_remove:
                del log_record[k]


# ── Filter ────────────────────────────────────────────────────────────────────


class ProductionLogFilter(logging.Filter):
    """Production/Staging 日志过滤器：标记非 safe_structured 事件"""

    def filter(self, record: logging.LogRecord) -> bool:
        # 如果已经有 safe_structured 标记，允许通过
        if getattr(record, "safe_structured", False):
            return True
        # 否则标记为需要 suppression（由 Formatter 处理）
        return True


# ── safe_log_event ────────────────────────────────────────────────────────────


def safe_log_event(
    logger: logging.Logger,
    level: str,
    event_code: str,
    error: Exception | None = None,
    **allowlisted_fields: Any,
) -> None:
    """安全结构化日志事件（Production/Staging 唯一允许的日志方式）

    Args:
        logger: 目标 logger
        level: 日志级别（"DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"）
        event_code: 事件代码（如 "eval_completed", "llm_call_failed"）
        error: 可选异常对象（只输出 class name 和 error_code）
        **allowlisted_fields: 只允许 allowlist 中的字段
    """
    # 过滤只保留 allowlist 字段
    safe_extra = {k: v for k, v in allowlisted_fields.items() if k in _SAFE_LOG_FIELDS}
    safe_extra["event_code"] = event_code
    safe_extra["safe_structured"] = True

    # 如果有 error，只输出 class name 和 error_code
    if error is not None:
        safe_extra["error_type"] = type(error).__name__
        if "error_code" not in safe_extra:
            safe_extra["error_code"] = type(error).__name__

    # 使用标准日志调用，extra 中的字段会注入到 LogRecord
    log_level = getattr(logging, level.upper(), logging.INFO)
    logger.log(log_level, event_code, extra=safe_extra)


# ── Setup ─────────────────────────────────────────────────────────────────────


def setup_logging() -> None:
    """
    初始化全局日志系统。
    根据 settings.LOG_FORMAT 选择 json 或 text 格式。
    根据 settings.LOG_LEVEL 设置日志级别。
    Production/Staging 环境自动启用安全日志策略。
    """
    log_level = getattr(logging, settings.LOG_LEVEL.upper(), logging.INFO)
    root_logger = logging.getLogger()
    root_logger.setLevel(log_level)

    # 清除已有 handler，避免重复配置
    root_logger.handlers.clear()

    handler = logging.StreamHandler(sys.stdout)
    handler.setLevel(log_level)

    # Production/Staging 使用 SafeStructuredFormatter
    is_production = settings.ENVIRONMENT in ("staging", "production")

    if is_production:
        formatter: logging.Formatter = SafeStructuredFormatter(
            "%(timestamp)s %(level)s %(name)s %(message)s"
        )
        handler.addFilter(ProductionLogFilter())
    elif settings.LOG_FORMAT.lower() == "json":
        formatter = CustomJsonFormatter(
            "%(timestamp)s %(level)s %(name)s %(message)s"
        )
    else:
        # text 格式（开发环境友好）
        formatter = logging.Formatter(
            "%(asctime)s %(levelname)-8s %(name)s %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        )

    handler.setFormatter(formatter)
    root_logger.addHandler(handler)
