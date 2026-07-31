# -*- coding: utf-8 -*-
"""Task 10 — RunBudget 并发、Token、成本和缓存预算测试

覆盖：
- RunBudget 配置与默认值
- BudgetDecision 枚举语义
- acquire_agent_slot: 并发上限、安全路径豁免、超预算拒绝
- release_agent_slot: 释放后可再获取
- record_usage: token / cost 累计
- 取消/异常后 slot 释放
- cache key 版本隔离
- 预算超限转 degraded 状态
"""

import asyncio
import time
import pytest

from app.services.run_budget import (
    BudgetDecision,
    RunBudget,
    RunBudgetManager,
    CacheKeyBuilder,
    UsageRecord,
)


# ── RunBudget 配置 ─────────────────────────────────────────────────────────


class TestRunBudgetConfig:
    def test_default_values(self):
        b = RunBudget()
        assert b.max_total_tokens > 0
        assert b.max_total_cost > 0
        assert b.max_parallel_agents > 0
        assert b.max_duration_seconds > 0

    def test_custom_values(self):
        b = RunBudget(
            max_total_tokens=5000,
            max_total_cost=1.0,
            max_parallel_agents=3,
            max_duration_seconds=120,
        )
        assert b.max_total_tokens == 5000
        assert b.max_total_cost == 1.0
        assert b.max_parallel_agents == 3
        assert b.max_duration_seconds == 120


# ── BudgetDecision ─────────────────────────────────────────────────────────


class TestBudgetDecision:
    def test_decision_values(self):
        assert BudgetDecision.ACCEPTED.value == "accepted"
        assert BudgetDecision.REJECTED.value == "rejected"
        assert BudgetDecision.DEGRADED.value == "degraded"

    def test_is_allowed(self):
        assert BudgetDecision.ACCEPTED.is_allowed is True
        assert BudgetDecision.REJECTED.is_allowed is False
        assert BudgetDecision.DEGRADED.is_allowed is True


# ── acquire_agent_slot ────────────────────────────────────────────────────


class TestAcquireAgentSlot:
    def test_acquire_within_limit(self):
        mgr = RunBudgetManager(RunBudget(max_parallel_agents=3))
        run_id = "run-001"
        d1 = mgr.acquire_agent_slot(run_id, "inquiry")
        assert d1 == BudgetDecision.ACCEPTED
        d2 = mgr.acquire_agent_slot(run_id, "knowledge")
        assert d2 == BudgetDecision.ACCEPTED

    def test_acquire_exceeds_limit(self):
        mgr = RunBudgetManager(RunBudget(max_parallel_agents=2))
        run_id = "run-002"
        mgr.acquire_agent_slot(run_id, "inquiry")
        mgr.acquire_agent_slot(run_id, "knowledge")
        d3 = mgr.acquire_agent_slot(run_id, "diagnosis")
        assert d3 == BudgetDecision.REJECTED

    def test_safety_agent_exempt_from_limit(self):
        """安全 Agent 不受并发上限约束"""
        mgr = RunBudgetManager(RunBudget(max_parallel_agents=1))
        run_id = "run-003"
        mgr.acquire_agent_slot(run_id, "inquiry")
        # safety 路径应豁免
        d = mgr.acquire_agent_slot(run_id, "safety")
        assert d == BudgetDecision.ACCEPTED

    def test_review_agent_exempt_from_limit(self):
        """复核 Agent 不受并发上限约束"""
        mgr = RunBudgetManager(RunBudget(max_parallel_agents=1))
        run_id = "run-004"
        mgr.acquire_agent_slot(run_id, "inquiry")
        d = mgr.acquire_agent_slot(run_id, "review")
        assert d == BudgetDecision.ACCEPTED

    def test_different_runs_independent(self):
        mgr = RunBudgetManager(RunBudget(max_parallel_agents=1))
        mgr.acquire_agent_slot("run-a", "inquiry")
        d = mgr.acquire_agent_slot("run-b", "inquiry")
        assert d == BudgetDecision.ACCEPTED


# ── release_agent_slot ────────────────────────────────────────────────────


class TestReleaseAgentSlot:
    def test_release_allows_reacquire(self):
        mgr = RunBudgetManager(RunBudget(max_parallel_agents=1))
        run_id = "run-010"
        mgr.acquire_agent_slot(run_id, "inquiry")
        mgr.release_agent_slot(run_id, "inquiry")
        d = mgr.acquire_agent_slot(run_id, "knowledge")
        assert d == BudgetDecision.ACCEPTED

    def test_release_unknown_no_error(self):
        mgr = RunBudgetManager()
        mgr.release_agent_slot("no-run", "no-agent")  # 不抛异常

    def test_release_only_specific_agent(self):
        mgr = RunBudgetManager(RunBudget(max_parallel_agents=2))
        run_id = "run-011"
        mgr.acquire_agent_slot(run_id, "inquiry")
        mgr.acquire_agent_slot(run_id, "knowledge")
        # 满员时拒绝
        d_reject = mgr.acquire_agent_slot(run_id, "diagnosis")
        assert d_reject == BudgetDecision.REJECTED
        # 释放 inquiry 后只剩 knowledge，可以再获取
        mgr.release_agent_slot(run_id, "inquiry")
        d = mgr.acquire_agent_slot(run_id, "diagnosis")
        assert d == BudgetDecision.ACCEPTED


# ── record_usage ──────────────────────────────────────────────────────────


class TestRecordUsage:
    def test_record_tokens_and_cost(self):
        mgr = RunBudgetManager(RunBudget(max_total_tokens=10000, max_total_cost=5.0))
        run_id = "run-020"
        mgr.record_usage(run_id, UsageRecord(tokens=500, cost=0.01, model="gpt-4", latency_ms=200))
        mgr.record_usage(run_id, UsageRecord(tokens=300, cost=0.005, model="gpt-4", latency_ms=150))
        summary = mgr.get_run_summary(run_id)
        assert summary["total_tokens"] == 800
        assert abs(summary["total_cost"] - 0.015) < 1e-6

    def test_over_budget_returns_degraded(self):
        mgr = RunBudgetManager(RunBudget(max_total_tokens=1000))
        run_id = "run-021"
        mgr.record_usage(run_id, UsageRecord(tokens=900, cost=0.0))
        decision = mgr.record_usage(run_id, UsageRecord(tokens=200, cost=0.0))
        assert decision == BudgetDecision.DEGRADED

    def test_over_cost_budget_returns_degraded(self):
        mgr = RunBudgetManager(RunBudget(max_total_cost=0.05))
        run_id = "run-022"
        mgr.record_usage(run_id, UsageRecord(tokens=10, cost=0.04))
        decision = mgr.record_usage(run_id, UsageRecord(tokens=10, cost=0.02))
        assert decision == BudgetDecision.DEGRADED


# ── cancel / exception slot release ──────────────────────────────────────


class TestSlotCleanup:
    def test_cancel_run_releases_all_slots(self):
        mgr = RunBudgetManager(RunBudget(max_parallel_agents=2))
        run_id = "run-030"
        mgr.acquire_agent_slot(run_id, "inquiry")
        mgr.acquire_agent_slot(run_id, "knowledge")
        mgr.cancel_run(run_id)
        # 取消后 slot 应全部释放
        d = mgr.acquire_agent_slot(run_id, "diagnosis")
        assert d == BudgetDecision.ACCEPTED

    def test_context_manager_releases_on_exception(self):
        mgr = RunBudgetManager(RunBudget(max_parallel_agents=1))
        run_id = "run-031"
        with pytest.raises(ValueError):
            with mgr.slot_context(run_id, "inquiry"):
                raise ValueError("模拟异常")
        # 异常后 slot 应释放
        d = mgr.acquire_agent_slot(run_id, "knowledge")
        assert d == BudgetDecision.ACCEPTED


# ── CacheKeyBuilder ───────────────────────────────────────────────────────


class TestCacheKeyBuilder:
    def test_same_input_same_key(self):
        key1 = CacheKeyBuilder.build(
            model_version="gpt-4-0125",
            prompt_version="v1.2",
            kb_version="kb-2026-07",
            temperature=0.0,
            tenant="tenant-a",
            content_hash="abc123",
        )
        key2 = CacheKeyBuilder.build(
            model_version="gpt-4-0125",
            prompt_version="v1.2",
            kb_version="kb-2026-07",
            temperature=0.0,
            tenant="tenant-a",
            content_hash="abc123",
        )
        assert key1 == key2

    def test_different_model_version_different_key(self):
        k1 = CacheKeyBuilder.build("gpt-4-v1", "p1", "kb1", 0.0, "t1", "c1")
        k2 = CacheKeyBuilder.build("gpt-4-v2", "p1", "kb1", 0.0, "t1", "c1")
        assert k1 != k2

    def test_different_prompt_version_different_key(self):
        k1 = CacheKeyBuilder.build("m1", "p-v1", "kb1", 0.0, "t1", "c1")
        k2 = CacheKeyBuilder.build("m1", "p-v2", "kb1", 0.0, "t1", "c1")
        assert k1 != k2

    def test_different_kb_version_different_key(self):
        k1 = CacheKeyBuilder.build("m1", "p1", "kb-v1", 0.0, "t1", "c1")
        k2 = CacheKeyBuilder.build("m1", "p1", "kb-v2", 0.0, "t1", "c1")
        assert k1 != k2

    def test_different_temperature_different_key(self):
        k1 = CacheKeyBuilder.build("m1", "p1", "kb1", 0.0, "t1", "c1")
        k2 = CacheKeyBuilder.build("m1", "p1", "kb1", 0.7, "t1", "c1")
        assert k1 != k2

    def test_different_tenant_different_key(self):
        k1 = CacheKeyBuilder.build("m1", "p1", "kb1", 0.0, "tenant-a", "c1")
        k2 = CacheKeyBuilder.build("m1", "p1", "kb1", 0.0, "tenant-b", "c1")
        assert k1 != k2

    def test_key_format_is_deterministic(self):
        """多次调用结果一致"""
        args = ("m1", "p1", "kb1", 0.0, "t1", "content-hash")
        keys = {CacheKeyBuilder.build(*args) for _ in range(50)}
        assert len(keys) == 1


# ── Run summary ───────────────────────────────────────────────────────────


class TestRunSummary:
    def test_summary_empty_run(self):
        mgr = RunBudgetManager()
        s = mgr.get_run_summary("nonexistent")
        assert s["total_tokens"] == 0
        assert s["total_cost"] == 0.0
        assert s["active_agents"] == []

    def test_summary_with_usage(self):
        mgr = RunBudgetManager()
        run_id = "run-040"
        mgr.acquire_agent_slot(run_id, "inquiry")
        mgr.record_usage(run_id, UsageRecord(tokens=100, cost=0.01, model="gpt-4", latency_ms=50))
        s = mgr.get_run_summary(run_id)
        assert s["total_tokens"] == 100
        assert "inquiry" in s["active_agents"]

    def test_summary_includes_usage_records(self):
        mgr = RunBudgetManager()
        run_id = "run-041"
        mgr.record_usage(run_id, UsageRecord(tokens=100, cost=0.01, model="gpt-4", latency_ms=50))
        mgr.record_usage(run_id, UsageRecord(tokens=200, cost=0.02, model="gpt-3.5", latency_ms=30))
        s = mgr.get_run_summary(run_id)
        assert s["total_tokens"] == 300
        assert s["usage_count"] == 2
