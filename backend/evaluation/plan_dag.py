# -*- coding: utf-8 -*-
"""通用 PlanStep DAG — 将 Wave 1/Wave 2 依赖收敛为通用 DAG 调度。

核心语义：
- 验证循环依赖、重复 step_id、未知依赖
- 根据 completed 集合计算 ready steps
- 依赖失败传播为 blocked

用法：
    from evaluation.plan_dag import validate_plan_dag, ready_steps

    errors = validate_plan_dag(steps)
    ready = ready_steps(steps, completed={"knowledge"})
"""
from __future__ import annotations

from collections import defaultdict
from enum import Enum
from typing import Optional

from pydantic import BaseModel, Field


# ── Step 状态枚举 ────────────────────────────────────────────────────────────


class StepStatus(str, Enum):
    """Step 执行状态。"""

    PENDING = "pending"
    RUNNING = "running"
    SUCCEEDED = "succeeded"
    FAILED = "failed"
    SKIPPED = "skipped"
    BLOCKED = "blocked"


# ── PlanStep 模型 ────────────────────────────────────────────────────────────


class PlanStep(BaseModel):
    """执行计划中的单个步骤。"""

    step_id: str
    depends_on: list[str] = Field(default_factory=list)
    status: StepStatus = StepStatus.PENDING
    agent_name: Optional[str] = None
    result: Optional[dict] = None


# ── DAG 验证 ─────────────────────────────────────────────────────────────────


def validate_plan_dag(steps: list[PlanStep]) -> list[str]:
    """验证 PlanStep DAG 合法性。

    检查：
    1. 重复 step_id
    2. 未知依赖
    3. 循环依赖

    Args:
        steps: PlanStep 列表。

    Returns:
        错误列表（空 = 合法）。
    """
    errors: list[str] = []
    step_ids: set[str] = set()

    # 检查重复
    for s in steps:
        if s.step_id in step_ids:
            errors.append(f"duplicate step_id: {s.step_id}")
        step_ids.add(s.step_id)

    # 检查未知依赖
    for s in steps:
        for dep in s.depends_on:
            if dep not in step_ids:
                errors.append(f"unknown dependency: {s.step_id} -> {dep}")

    # 检查循环依赖（拓扑排序）
    if not errors:
        adj: dict[str, list[str]] = defaultdict(list)
        in_degree: dict[str, int] = {s.step_id: 0 for s in steps}
        for s in steps:
            for dep in s.depends_on:
                adj[dep].append(s.step_id)
                in_degree[s.step_id] += 1

        queue = [sid for sid, deg in in_degree.items() if deg == 0]
        visited = 0
        while queue:
            node = queue.pop(0)
            visited += 1
            for neighbor in adj[node]:
                in_degree[neighbor] -= 1
                if in_degree[neighbor] == 0:
                    queue.append(neighbor)

        if visited != len(steps):
            errors.append("cycle detected in plan DAG")

    return errors


# ── Ready Steps 计算 ─────────────────────────────────────────────────────────


def ready_steps(steps: list[PlanStep], completed: set[str]) -> list[PlanStep]:
    """计算当前可执行的 steps。

    Args:
        steps: 全部 PlanStep。
        completed: 已完成的 step_id 集合。

    Returns:
        当前可执行的 PlanStep 列表。
    """
    ready = []
    for s in steps:
        if s.status != StepStatus.PENDING:
            continue
        # 所有依赖都已完成
        if all(dep in completed for dep in s.depends_on):
            ready.append(s)
    return ready


# ── 状态更新 ─────────────────────────────────────────────────────────────────


def mark_step_status(
    steps: dict[str, PlanStep],
    step_id: str,
    status: StepStatus,
) -> dict[str, PlanStep]:
    """更新 step 状态。

    Args:
        steps: {step_id: PlanStep} 字典。
        step_id: 目标 step。
        status: 新状态。

    Returns:
        更新后的 steps 字典。
    """
    if step_id not in steps:
        return steps

    steps[step_id].status = status

    # 如果失败，将依赖者标记为 blocked
    if status == StepStatus.FAILED:
        for sid, s in steps.items():
            if step_id in s.depends_on and s.status == StepStatus.PENDING:
                s.status = StepStatus.BLOCKED

    return steps
