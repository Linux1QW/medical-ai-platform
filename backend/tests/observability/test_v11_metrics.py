# -*- coding: utf-8 -*-
"""V1.1 可运营指标测试

验证所有 V1.1 指标：
- 注册成功（名称存在）
- 标签正确（低基数，不含 run_id / consultation_id / user_id）
- Counter / Gauge / Histogram 类型正确
- 指标更新时数值变化正确
"""

from __future__ import annotations

import pytest
from prometheus_client import Counter, Gauge, Histogram

# ── 导入待测模块（Phase 1 时这些名称尚不存在，测试应先失败）──────────────
from app.services.observability import metrics as metrics_mod

# ── Helpers ──────────────────────────────────────────────────────────────────

# 高基数标签 — 指标中绝对不能出现
_HIGH_CARD_LABELS = frozenset({"run_id", "consultation_id", "user_id", "model"})


def _get_metric(name: str):
    """从 prometheus_client REGISTRY 中按名称获取指标

    prometheus_client 会自动去掉 Counter 的 ``_total`` 后缀存储在 _name 中，
    因此需要同时尝试原始名称和去掉 ``_total`` 的版本。
    """
    # prometheus_client strips ``_total`` from Counter names internally
    candidates = {name}
    if name.endswith("_total"):
        candidates.add(name[: -len("_total")])
    for metric in metrics_mod.__dict__.values():
        if isinstance(metric, (Counter, Gauge, Histogram)):
            if metric._name in candidates:
                return metric
    return None


# ── 1. 指标注册测试 ──────────────────────────────────────────────────────────

REQUIRED_METRICS = {
    # name → expected type
    "evaluation_runs_total": Counter,
    "evaluation_run_duration_seconds": Histogram,
    "evaluation_queue_wait_seconds": Histogram,
    "evaluation_active_runs": Gauge,
    "evaluation_retries_total": Counter,
    "evaluation_cancellations_total": Counter,
    "evaluation_progress_publish_total": Counter,
    "evaluation_progress_delivery_seconds": Histogram,
    "evaluation_stale_runs_total": Counter,
    "evaluation_run_lease_lost_total": Counter,
    "evaluation_outbox_events_total": Counter,
    "evaluation_outbox_pending": Gauge,
    "evaluation_outbox_oldest_seconds": Gauge,
    "evaluation_dispatch_duration_seconds": Histogram,
    "evaluation_dispatch_dead_letter_total": Counter,
    "evaluation_dispatch_breaker_open": Gauge,
    "redis_dependency_status": Gauge,
    "backup_last_success_timestamp_seconds": Gauge,
    "review_queue_depth": Gauge,
    "review_completion_seconds": Histogram,
}


@pytest.mark.parametrize("name,expected_type", list(REQUIRED_METRICS.items()))
def test_metric_registered(name: str, expected_type: type) -> None:
    """每个 V1.1 指标必须已注册"""
    m = _get_metric(name)
    assert m is not None, f"Metric '{name}' not found in metrics module"
    assert isinstance(m, expected_type), (
        f"Metric '{name}' expected {expected_type.__name__}, got {type(m).__name__}"
    )


# ── 2. 低基数标签测试 ────────────────────────────────────────────────────────

@pytest.mark.parametrize("name", list(REQUIRED_METRICS.keys()))
def test_no_high_cardinality_labels(name: str) -> None:
    """指标标签不得包含高基数字段"""
    m = _get_metric(name)
    assert m is not None
    label_names = set(m._labelnames) if m._labelnames else set()
    offending = label_names & _HIGH_CARD_LABELS
    assert not offending, (
        f"Metric '{name}' has high-cardinality labels: {offending}"
    )


# ── 3. 特定标签结构测试 ──────────────────────────────────────────────────────

def test_runs_total_labels() -> None:
    """evaluation_runs_total 必须有 status + error_code 标签"""
    m = _get_metric("evaluation_runs_total")
    assert m is not None
    assert set(m._labelnames) == {"status", "error_code"}


def test_active_runs_labels() -> None:
    """evaluation_active_runs 必须有 status 标签"""
    m = _get_metric("evaluation_active_runs")
    assert m is not None
    assert set(m._labelnames) == {"status"}


def test_retries_total_labels() -> None:
    """evaluation_retries_total 必须有 error_code 标签"""
    m = _get_metric("evaluation_retries_total")
    assert m is not None
    assert set(m._labelnames) == {"error_code"}


def test_cancellations_total_labels() -> None:
    """evaluation_cancellations_total 必须有 phase 标签"""
    m = _get_metric("evaluation_cancellations_total")
    assert m is not None
    assert set(m._labelnames) == {"phase"}


def test_progress_publish_total_labels() -> None:
    """evaluation_progress_publish_total 必须有 result 标签"""
    m = _get_metric("evaluation_progress_publish_total")
    assert m is not None
    assert set(m._labelnames) == {"result"}


def test_stale_runs_total_labels() -> None:
    """evaluation_stale_runs_total 必须有 reason 标签"""
    m = _get_metric("evaluation_stale_runs_total")
    assert m is not None
    assert set(m._labelnames) == {"reason"}


def test_dispatch_duration_labels() -> None:
    """evaluation_dispatch_duration_seconds 必须有 result 标签"""
    m = _get_metric("evaluation_dispatch_duration_seconds")
    assert m is not None
    assert set(m._labelnames) == {"result"}


def test_dead_letter_labels() -> None:
    """evaluation_dispatch_dead_letter_total 必须有 reason 标签"""
    m = _get_metric("evaluation_dispatch_dead_letter_total")
    assert m is not None
    assert set(m._labelnames) == {"reason"}


def test_redis_dependency_labels() -> None:
    """redis_dependency_status 必须有 role 标签"""
    m = _get_metric("redis_dependency_status")
    assert m is not None
    assert set(m._labelnames) == {"role"}


# ── 4. 指标更新行为测试 ──────────────────────────────────────────────────────

def test_runs_total_increment() -> None:
    """Counter 递增后值增加"""
    m = _get_metric("evaluation_runs_total")
    assert m is not None
    before = m.labels(status="completed", error_code="")._value.get()
    m.labels(status="completed", error_code="").inc()
    after = m.labels(status="completed", error_code="")._value.get()
    assert after == before + 1


def test_active_runs_gauge_set() -> None:
    """Gauge set 后值正确"""
    m = _get_metric("evaluation_active_runs")
    assert m is not None
    m.labels(status="running").set(5)
    assert m.labels(status="running")._value.get() == 5
    m.labels(status="running").set(3)
    assert m.labels(status="running")._value.get() == 3


def test_run_duration_observe() -> None:
    """Histogram observe 后 _sum 增加"""
    m = _get_metric("evaluation_run_duration_seconds")
    assert m is not None
    before_sum = m.labels(status="completed")._sum.get()
    m.labels(status="completed").observe(2.5)
    after_sum = m.labels(status="completed")._sum.get()
    assert after_sum == pytest.approx(before_sum + 2.5)


def test_outbox_pending_gauge() -> None:
    """evaluation_outbox_pending 可 set"""
    m = _get_metric("evaluation_outbox_pending")
    assert m is not None
    m.set(10)
    assert m._value.get() == 10
    m.set(0)
    assert m._value.get() == 0


def test_dispatch_breaker_open_gauge() -> None:
    """evaluation_dispatch_breaker_open 可 set 0/1"""
    m = _get_metric("evaluation_dispatch_breaker_open")
    assert m is not None
    m.set(1)
    assert m._value.get() == 1
    m.set(0)
    assert m._value.get() == 0


def test_backup_timestamp_gauge() -> None:
    """backup_last_success_timestamp_seconds 可 set"""
    m = _get_metric("backup_last_success_timestamp_seconds")
    assert m is not None
    m.set(1700000000)
    assert m._value.get() == 1700000000


def test_review_queue_depth_gauge() -> None:
    """review_queue_depth 可 inc/set"""
    m = _get_metric("review_queue_depth")
    assert m is not None
    m.set(7)
    assert m._value.get() == 7
