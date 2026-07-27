# -*- coding: utf-8 -*-
"""评估任务重试分类测试：仅网络/超时类异常值得 Celery 重试"""

import pytest

from app.tasks.evaluation_task import _is_retryable_error

# ── 自定义异常：模拟第三方 SDK 的网络类错误命名 ────────────────────────────────


class QwenAPITimeoutError(Exception):
    pass


class ServiceUnavailableError(Exception):
    pass


class RateLimitExceeded(Exception):
    pass


class TemporaryFailure(Exception):
    pass


class DataValidationError(Exception):
    pass


@pytest.mark.parametrize("exc", [
    TimeoutError("timed out"),
    ConnectionError("refused"),
    OSError("network down"),
    ConnectionResetError("reset"),          # ConnectionError 子类
    QwenAPITimeoutError("llm timeout"),     # 类名含 timeout
    ServiceUnavailableError("503"),         # 类名含 unavailable
    RateLimitExceeded("429"),               # 类名含 ratelimit
    TemporaryFailure("retry later"),        # 类名含 temporar
])
def test_retryable_errors(exc):
    assert _is_retryable_error(exc) is True


@pytest.mark.parametrize("exc", [
    ValueError("bad input"),
    KeyError("missing"),
    RuntimeError("logic error"),
    DataValidationError("schema mismatch"),  # 业务校验错误不重试
    ZeroDivisionError(),
])
def test_non_retryable_errors(exc):
    assert _is_retryable_error(exc) is False
