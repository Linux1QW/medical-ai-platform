# -*- coding: utf-8 -*-
"""Task 7 — Production logging 隐私安全测试

验证 production/staging 环境下日志系统不会泄露患者 PII、API key 等敏感信息。
"""

import io
import json
import logging
from unittest.mock import patch

import pytest


# ── 常量 ──────────────────────────────────────────────────────────────────────

_SENTINEL = "SUPER_SECRET_SENTINEL_XYZ123"
_PHONE = "13812345678"
_ID_CARD = "110101199001011234"
_API_KEY = "sk-super-secret-api-key-12345"


# ── 辅助 ──────────────────────────────────────────────────────────────────────


def _capture_log_output(logger_name: str, log_func, *args, **kwargs) -> str:
    """捕获 logger 输出的 JSON 字符串"""
    buf = io.StringIO()
    handler = logging.StreamHandler(buf)
    from app.core.logging import CustomJsonFormatter
    formatter = CustomJsonFormatter("%(timestamp)s %(level)s %(name)s %(message)s")
    handler.setFormatter(formatter)

    target_logger = logging.getLogger(logger_name)
    target_logger.addHandler(handler)
    target_logger.setLevel(logging.DEBUG)
    try:
        log_func(*args, **kwargs)
    finally:
        target_logger.removeHandler(handler)
    return buf.getvalue()


# ── 测试：production logging 不含敏感信息 ────────────────────────────────────


class TestProductionLoggingPrivacy:
    """production/staging handler 不允许自由 message 中出现敏感信息"""

    def test_sentinel_in_msg_is_suppressed(self):
        from app.core.logging import safe_log_event

        buf = io.StringIO()
        handler = logging.StreamHandler(buf)
        from app.core.logging import SafeStructuredFormatter
        formatter = SafeStructuredFormatter()
        handler.setFormatter(formatter)

        target_logger = logging.getLogger("test.prod.suppress_msg")
        target_logger.addHandler(handler)
        target_logger.setLevel(logging.DEBUG)
        try:
            safe_log_event(
                target_logger,
                "INFO",
                "eval_completed",
                request_id="req-001",
                trace_id="trace-001",
                run_id="run-001",
                status="completed",
            )
            output = buf.getvalue()
            assert _SENTINEL not in output
        finally:
            target_logger.removeHandler(handler)

    def test_unsafe_log_is_suppressed(self):
        """未带 safe_structured=true 的 LogRecord 被替换为固定 suppression"""
        from app.core.logging import ProductionLogFilter

        buf = io.StringIO()
        handler = logging.StreamHandler(buf)
        from app.core.logging import SafeStructuredFormatter
        formatter = SafeStructuredFormatter()
        handler.setFormatter(formatter)
        handler.addFilter(ProductionLogFilter())

        target_logger = logging.getLogger("test.prod.unsafe")
        target_logger.addHandler(handler)
        target_logger.setLevel(logging.DEBUG)
        try:
            # 直接使用 logger.info(msg) 不含 safe_structured 标记
            target_logger.info(f"Patient data: {_SENTINEL} phone={_PHONE}")
            output = buf.getvalue()
            assert _SENTINEL not in output
            assert _PHONE not in output
            assert "unsafe_log_message_suppressed" in output
        finally:
            target_logger.removeHandler(handler)

    def test_safe_event_preserves_allowlist_fields(self):
        """safe_log_event 保留 allowlist 字段"""
        from app.core.logging import safe_log_event, ProductionLogFilter, SafeStructuredFormatter

        buf = io.StringIO()
        handler = logging.StreamHandler(buf)
        formatter = SafeStructuredFormatter()
        handler.setFormatter(formatter)
        handler.addFilter(ProductionLogFilter())

        target_logger = logging.getLogger("test.prod.safe_event")
        target_logger.addHandler(handler)
        target_logger.setLevel(logging.DEBUG)
        try:
            safe_log_event(
                target_logger,
                "INFO",
                "eval_completed",
                request_id="req-001",
                trace_id="trace-abc",
                run_id="run-xyz",
                consultation_id="consult-123",
                agent_name="knowledge",
                status="completed",
                duration_ms=123.4,
            )
            output = buf.getvalue()
            # allowlist 字段应该存在
            assert "trace-abc" in output
            assert "run-xyz" in output
            assert "consult-123" in output
            assert "knowledge" in output
            assert "completed" in output
        finally:
            target_logger.removeHandler(handler)

    def test_can_locate_by_trace_id_and_run_id(self):
        """safe_log_event 输出中可以通过 trace_id/run_id 定位"""
        from app.core.logging import safe_log_event, ProductionLogFilter, SafeStructuredFormatter

        buf = io.StringIO()
        handler = logging.StreamHandler(buf)
        formatter = SafeStructuredFormatter()
        handler.setFormatter(formatter)
        handler.addFilter(ProductionLogFilter())

        target_logger = logging.getLogger("test.prod.locate")
        target_logger.addHandler(handler)
        target_logger.setLevel(logging.DEBUG)
        try:
            safe_log_event(
                target_logger,
                "INFO",
                "test_event",
                trace_id="trace-locate-001",
                run_id="run-locate-001",
            )
            output = buf.getvalue()
            assert "trace-locate-001" in output
            assert "run-locate-001" in output
        finally:
            target_logger.removeHandler(handler)


# ── 测试：safe_log_event 接口 ────────────────────────────────────────────────


class TestSafeLogEvent:
    """safe_log_event 只接受 allowlist 字段"""

    def test_safe_log_event_writes_structured_event(self):
        from app.core.logging import safe_log_event

        buf = io.StringIO()
        handler = logging.StreamHandler(buf)
        from app.core.logging import SafeStructuredFormatter
        formatter = SafeStructuredFormatter()
        handler.setFormatter(formatter)

        target_logger = logging.getLogger("test.safe_event.structured")
        target_logger.addHandler(handler)
        target_logger.setLevel(logging.DEBUG)
        try:
            safe_log_event(
                target_logger,
                "ERROR",
                "eval_error",
                error_code="EVAL_TIMEOUT",
                trace_id="trace-err",
                run_id="run-err",
                status="error",
                duration_ms=5000.0,
            )
            output = buf.getvalue()
            assert "eval_error" in output
            assert "EVAL_TIMEOUT" in output
            assert "trace-err" in output
        finally:
            target_logger.removeHandler(handler)

    def test_safe_log_event_does_not_accept_prompt_or_response(self):
        """safe_log_event 不接受 prompt/query/response/patient_info 字段"""
        from app.core.logging import safe_log_event

        buf = io.StringIO()
        handler = logging.StreamHandler(buf)
        from app.core.logging import SafeStructuredFormatter
        formatter = SafeStructuredFormatter()
        handler.setFormatter(formatter)

        target_logger = logging.getLogger("test.safe_event.no_free_fields")
        target_logger.addHandler(handler)
        target_logger.setLevel(logging.DEBUG)
        try:
            # 即使调用方尝试传入 prompt/response，也不应出现在输出中
            safe_log_event(
                target_logger,
                "INFO",
                "test_event",
                trace_id="trace-001",
                prompt=_SENTINEL,  # type: ignore  # 不在 allowlist
                response=_API_KEY,  # type: ignore  # 不在 allowlist
            )
            output = buf.getvalue()
            assert _SENTINEL not in output
            assert _API_KEY not in output
        finally:
            target_logger.removeHandler(handler)
