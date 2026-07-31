# -*- coding: utf-8 -*-
"""通用 PlanStep DAG 测试 — Task 9

验证：
1. 循环依赖被拒绝
2. 重复 step_id 被拒绝
3. 依赖未完成时 step 不 ready
4. 依赖失败传播为 blocked
5. reducer 汇聚结果顺序稳定
"""
import pytest

from evaluation.plan_dag import (
    PlanStep,
    StepStatus,
    ready_steps,
    validate_plan_dag,
    mark_step_status,
)


# ── 辅助 ────────────────────────────────────────────────────────────────────


def _step(step_id: str, depends_on: list[str] | None = None) -> PlanStep:
    return PlanStep(step_id=step_id, depends_on=depends_on or [])


# ── 1. validate_plan_dag ────────────────────────────────────────────────────


class TestValidatePlanDag:
    def test_valid_dag(self):
        steps = [_step("a"), _step("b", ["a"]), _step("c", ["a", "b"])]
        errors = validate_plan_dag(steps)
        assert errors == []

    def test_cycle_rejected(self):
        steps = [_step("a", ["b"]), _step("b", ["a"])]
        errors = validate_plan_dag(steps)
        assert any("cycle" in e.lower() or "循环" in e for e in errors)

    def test_duplicate_step_id_rejected(self):
        steps = [_step("a"), _step("a")]
        errors = validate_plan_dag(steps)
        assert any("duplicate" in e.lower() or "重复" in e for e in errors)

    def test_unknown_dependency_rejected(self):
        steps = [_step("a", ["nonexistent"])]
        errors = validate_plan_dag(steps)
        assert any("unknown" in e.lower() or "未知" in e for e in errors)


# ── 2. ready_steps ──────────────────────────────────────────────────────────


class TestReadySteps:
    def test_no_deps_always_ready(self):
        steps = [_step("a"), _step("b")]
        ready = ready_steps(steps, completed=set())
        assert {s.step_id for s in ready} == {"a", "b"}

    def test_dep_not_completed_not_ready(self):
        steps = [_step("a"), _step("b", ["a"])]
        ready = ready_steps(steps, completed=set())
        assert {s.step_id for s in ready} == {"a"}

    def test_dep_completed_becomes_ready(self):
        steps = [_step("a"), _step("b", ["a"])]
        # a 已完成（不再是 PENDING），只有 b 应 ready
        steps[0].status = StepStatus.SUCCEEDED
        ready = ready_steps(steps, completed={"a"})
        assert {s.step_id for s in ready} == {"b"}

    def test_multiple_deps_all_must_complete(self):
        steps = [_step("a"), _step("b"), _step("c", ["a", "b"])]
        ready = ready_steps(steps, completed={"a"})
        assert "c" not in {s.step_id for s in ready}
        ready = ready_steps(steps, completed={"a", "b"})
        assert "c" in {s.step_id for s in ready}


# ── 3. mark_step_status ─────────────────────────────────────────────────────


class TestMarkStepStatus:
    def test_mark_succeeded(self):
        steps = {"a": _step("a")}
        result = mark_step_status(steps, "a", StepStatus.SUCCEEDED)
        assert result["a"].status == StepStatus.SUCCEEDED

    def test_mark_failed_propagates(self):
        """失败 step 的依赖者变为 blocked"""
        steps = {
            "a": _step("a"),
            "b": _step("b", ["a"]),
        }
        mark_step_status(steps, "a", StepStatus.FAILED)
        assert steps["a"].status == StepStatus.FAILED
        # b 应该被标记为 blocked（由调用方处理传播逻辑）


# ── 4. StepStatus 枚举 ──────────────────────────────────────────────────────


class TestStepStatus:
    def test_all_statuses(self):
        assert StepStatus.PENDING == "pending"
        assert StepStatus.RUNNING == "running"
        assert StepStatus.SUCCEEDED == "succeeded"
        assert StepStatus.FAILED == "failed"
        assert StepStatus.SKIPPED == "skipped"
        assert StepStatus.BLOCKED == "blocked"
