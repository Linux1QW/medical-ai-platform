# -*- coding: utf-8 -*-
"""评估任务取消测试 — 取消标志服务 + 取消看守 + 失败归类"""

import asyncio

import pytest

from app.core.config import settings
from app.core.failure_reasons import REASON_CANCELLED, EvaluationCancelled, classify_failure
from app.services import evaluation_cancel
from app.services.evaluation_service import _invoke_graph_with_deadline


class _FakeRedis:
    """内存字典模拟 Redis（set/get/delete）"""

    def __init__(self):
        self.store: dict[str, str] = {}

    async def set(self, key, value, ex=None):
        self.store[key] = value

    async def get(self, key):
        return self.store.get(key)

    async def delete(self, key):
        self.store.pop(key, None)


class _BrokenRedis:
    """所有操作抛异常的 Redis 桩"""

    def __getattr__(self, name):
        async def _fail(*args, **kwargs):
            raise ConnectionError("redis down")

        return _fail


@pytest.fixture
def fake_redis(monkeypatch):
    redis = _FakeRedis()

    async def _get(*args, **kwargs):
        return redis

    monkeypatch.setattr(evaluation_cancel, "_get_redis", _get)
    return redis


# ── 取消标志服务 ──────────────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_request_cancel_sets_flag(fake_redis):
    await evaluation_cancel.request_cancel(1)

    assert await evaluation_cancel.is_cancel_requested(1) is True
    assert await evaluation_cancel.is_cancel_requested(2) is False


@pytest.mark.asyncio
async def test_clear_cancel_flag(fake_redis):
    await evaluation_cancel.request_cancel(1)
    await evaluation_cancel.clear_cancel_flag(1)

    assert await evaluation_cancel.is_cancel_requested(1) is False


@pytest.mark.asyncio
async def test_request_cancel_raises_when_redis_unavailable(monkeypatch):
    """写路径：Redis 不可用时明确报错，不假装取消成功"""

    async def _get(*args, **kwargs):
        return None

    monkeypatch.setattr(evaluation_cancel, "_get_redis", _get)

    with pytest.raises(RuntimeError):
        await evaluation_cancel.request_cancel(1)


@pytest.mark.asyncio
async def test_read_paths_are_best_effort(monkeypatch):
    """读路径：Redis 异常时不阻断主流程"""

    async def _get(*args, **kwargs):
        return _BrokenRedis()

    monkeypatch.setattr(evaluation_cancel, "_get_redis", _get)

    assert await evaluation_cancel.is_cancel_requested(1) is False
    assert await evaluation_cancel.get_task_id(1) is None
    await evaluation_cancel.clear_cancel_flag(1)  # 不抛异常
    await evaluation_cancel.store_task_id(1, "tid")  # 不抛异常


@pytest.mark.asyncio
async def test_store_and_get_task_id(fake_redis):
    await evaluation_cancel.store_task_id(1, "celery-task-abc")

    assert await evaluation_cancel.get_task_id(1) == "celery-task-abc"
    assert await evaluation_cancel.get_task_id(2) is None


# ── 取消看守（_invoke_graph_with_deadline）────────────────────────────────


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
async def test_cancel_during_execution_raises_evaluation_cancelled(monkeypatch):
    """执行中命中取消标志 → 看守 cancel 图任务 → 转换为 EvaluationCancelled"""
    import app.services.evaluation_service as svc

    monkeypatch.setattr(settings, "EVALUATION_RUN_TIMEOUT_SECONDS", 0)
    monkeypatch.setattr(settings, "EVAL_CANCEL_POLL_SECONDS", 0.05)

    async def _cancelled(consultation_id):
        return True

    monkeypatch.setattr(svc, "is_cancel_requested", _cancelled)
    graph = _StubGraph(delay=5)

    with pytest.raises(EvaluationCancelled):
        await _invoke_graph_with_deadline(graph, {}, {}, consultation_id=1)


@pytest.mark.asyncio
async def test_external_cancel_reraises_cancelled_error(monkeypatch):
    """外部取消（无取消标志，如 Celery 强杀）不吞 CancelledError"""
    import app.services.evaluation_service as svc

    monkeypatch.setattr(settings, "EVALUATION_RUN_TIMEOUT_SECONDS", 0)
    monkeypatch.setattr(settings, "EVAL_CANCEL_POLL_SECONDS", 0.05)

    async def _not_cancelled(consultation_id):
        return False

    monkeypatch.setattr(svc, "is_cancel_requested", _not_cancelled)
    graph = _StubGraph(delay=5)

    task = asyncio.ensure_future(
        _invoke_graph_with_deadline(graph, {}, {}, consultation_id=1)
    )
    await asyncio.sleep(0.1)
    task.cancel()

    with pytest.raises(asyncio.CancelledError):
        await task


@pytest.mark.asyncio
async def test_completes_normally_with_watcher(monkeypatch):
    """未取消时看守不干扰正常返回，且被 finally 回收"""
    import app.services.evaluation_service as svc

    monkeypatch.setattr(settings, "EVALUATION_RUN_TIMEOUT_SECONDS", 5)
    monkeypatch.setattr(settings, "EVAL_CANCEL_POLL_SECONDS", 0.05)

    async def _not_cancelled(consultation_id):
        return False

    monkeypatch.setattr(svc, "is_cancel_requested", _not_cancelled)
    graph = _StubGraph(delay=0)

    result = await _invoke_graph_with_deadline(graph, {}, {}, consultation_id=1)

    assert result["evaluation_status"] == "completed"


@pytest.mark.asyncio
async def test_poll_disabled_falls_back_to_plain_path(monkeypatch):
    """EVAL_CANCEL_POLL_SECONDS=0 时禁用看守，保持原有最简路径"""
    monkeypatch.setattr(settings, "EVALUATION_RUN_TIMEOUT_SECONDS", 5)
    monkeypatch.setattr(settings, "EVAL_CANCEL_POLL_SECONDS", 0)
    graph = _StubGraph(delay=0)

    result = await _invoke_graph_with_deadline(graph, {}, {}, consultation_id=1)

    assert result["evaluation_status"] == "completed"


# ── 失败归类与重试判定 ────────────────────────────────────────────────────


def test_cancelled_classified_as_cancelled():
    assert classify_failure(EvaluationCancelled("用户取消")) == REASON_CANCELLED


def test_cancelled_is_not_retryable():
    """取消不触发 Celery 重试（类名不含网络/超时类关键词）"""
    from app.tasks.evaluation_task import _is_retryable_error

    assert _is_retryable_error(EvaluationCancelled("用户取消")) is False


def test_cancelled_is_normal_exception():
    """必须是普通 Exception（而非 CancelledError/BaseException），
    否则会逃逸 evaluation_service 的 except Exception 导致 run 状态不落库"""
    assert issubclass(EvaluationCancelled, Exception)
    assert not issubclass(EvaluationCancelled, asyncio.CancelledError)
