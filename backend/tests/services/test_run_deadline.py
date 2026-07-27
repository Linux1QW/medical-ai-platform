# -*- coding: utf-8 -*-
"""评估级 deadline 测试 — _invoke_graph_with_deadline 总时长预算"""

import asyncio

import pytest

from app.core.config import settings
from app.core.failure_reasons import REASON_TIMEOUT, EvaluationDeadlineExceeded, classify_failure
from app.services.evaluation_service import _invoke_graph_with_deadline


class _StubGraph:
    """可配置耗时的图桩"""

    def __init__(self, delay: float = 0.0, result=None):
        self._delay = delay
        self._result = result or {"evaluation_status": "completed"}

    async def ainvoke(self, initial_state, config=None):
        if self._delay:
            await asyncio.sleep(self._delay)
        return self._result


@pytest.mark.asyncio
async def test_returns_result_within_deadline(monkeypatch):
    monkeypatch.setattr(settings, "EVALUATION_RUN_TIMEOUT_SECONDS", 5)
    graph = _StubGraph(delay=0)

    result = await _invoke_graph_with_deadline(graph, {}, {})

    assert result["evaluation_status"] == "completed"


@pytest.mark.asyncio
async def test_zero_timeout_disables_deadline(monkeypatch):
    """0 = 不限制：不包裹 wait_for，慢图也能正常返回"""
    monkeypatch.setattr(settings, "EVALUATION_RUN_TIMEOUT_SECONDS", 0)
    graph = _StubGraph(delay=0.05)

    result = await _invoke_graph_with_deadline(graph, {}, {})

    assert result["evaluation_status"] == "completed"


@pytest.mark.asyncio
async def test_deadline_exceeded_raises_domain_error(monkeypatch):
    monkeypatch.setattr(settings, "EVALUATION_RUN_TIMEOUT_SECONDS", 1)
    graph = _StubGraph(delay=5)

    with pytest.raises(EvaluationDeadlineExceeded) as exc_info:
        await _invoke_graph_with_deadline(graph, {}, {})

    assert "总时长预算 1s" in str(exc_info.value)


def test_deadline_error_classified_as_timeout():
    """原因码归为 timeout，落库 error_type 供监控聚合"""
    assert classify_failure(EvaluationDeadlineExceeded("评估超出总时长预算 240s")) == REASON_TIMEOUT


def test_deadline_error_is_retryable():
    """TimeoutError 子类命中 Celery 可重试判定"""
    from app.tasks.evaluation_task import _is_retryable_error

    assert _is_retryable_error(EvaluationDeadlineExceeded("boom")) is True
