"""
结构化 JSON 日志配置

通过 setup_logging() 初始化日志系统：
- LOG_FORMAT=json（默认）：使用 python-json-logger 输出 JSON 格式
- LOG_FORMAT=text：使用人类可读的传统文本格式
- LOG_LEVEL 控制日志级别（默认 INFO）
"""

import logging
import sys
from datetime import datetime, timezone

from pythonjsonlogger.json import JsonFormatter as _BaseJsonFormatter

from app.core.config import settings


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


def setup_logging() -> None:
    """
    初始化全局日志系统。
    根据 settings.LOG_FORMAT 选择 json 或 text 格式。
    根据 settings.LOG_LEVEL 设置日志级别。
    """
    log_level = getattr(logging, settings.LOG_LEVEL.upper(), logging.INFO)
    root_logger = logging.getLogger()
    root_logger.setLevel(log_level)

    # 清除已有 handler，避免重复配置
    root_logger.handlers.clear()

    handler = logging.StreamHandler(sys.stdout)
    handler.setLevel(log_level)

    if settings.LOG_FORMAT.lower() == "json":
        formatter = CustomJsonFormatter(
            "%(timestamp)s %(level)s %(name)s %(message)s"
        )
        handler.setFormatter(formatter)
    else:
        # text 格式（开发环境友好）
        formatter = logging.Formatter(
            "%(asctime)s %(levelname)-8s %(name)s %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        )
        handler.setFormatter(formatter)

    root_logger.addHandler(handler)
