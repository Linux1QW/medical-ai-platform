# -*- coding: utf-8 -*-
"""失败原因码分类测试"""

import asyncio

import pytest

from app.core.failure_reasons import ALL_REASONS, classify_failure


class QwenAPITimeoutError(Exception):
    pass


class ServiceUnavailableError(Exception):
    pass


class RateLimitError(Exception):
    pass


class SomeValidationError(Exception):
    pass


class WeirdBusinessError(Exception):
    pass


@pytest.mark.parametrize("exc,expected", [
    (TimeoutError("timed out"), "timeout"),
    (asyncio.TimeoutError(), "timeout"),
    (QwenAPITimeoutError("llm timeout"), "timeout"),
    (RateLimitError("429"), "rate_limited"),
    (RuntimeError("LangGraph 图未初始化，请检查配置"), "graph_not_initialized"),
    (ConnectionError("refused"), "connection_error"),
    (ConnectionResetError("reset"), "connection_error"),
    (OSError("network down"), "connection_error"),
    (ServiceUnavailableError("503"), "connection_error"),
    (ValueError("bad json"), "validation_error"),
    (TypeError("bad type"), "validation_error"),
    (SomeValidationError("schema"), "validation_error"),
    (asyncio.CancelledError(), "cancelled"),
    (WeirdBusinessError("boom"), "unknown_error"),
    (KeyError("missing"), "unknown_error"),
])
def test_classify_failure(exc, expected):
    assert classify_failure(exc) == expected


def test_all_codes_are_registered():
    """classify_failure 的所有可能返回值都在 ALL_REASONS 中"""
    samples = [
        TimeoutError(), RateLimitError(), RuntimeError("图未初始化"),
        ConnectionError(), ValueError(), asyncio.CancelledError(), KeyError(),
    ]
    for exc in samples:
        assert classify_failure(exc) in ALL_REASONS
