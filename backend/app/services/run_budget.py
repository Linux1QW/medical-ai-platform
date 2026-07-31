# -*- coding: utf-8 -*-
"""Task 10 — 并发、Token、成本和缓存预算

RunBudgetManager 管理每次评估 run 的并发 Agent slot、Token 预算和成本上限。
CacheKeyBuilder 提供版本感知的 LLM 缓存键隔离。

设计原则：
- 安全 Agent (safety) 和复核 Agent (review) 豁免并发上限
- 超预算时返回 DEGRADED 而非直接拒绝，保留安全路径
- 异常/取消时通过 context manager 自动释放 slot
- cache key 绑定 model_version + prompt_version + kb_version + temperature + tenant
"""

from __future__ import annotations

import hashlib
import logging
import threading
from contextlib import contextmanager
from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, List, Optional, Set

logger = logging.getLogger(__name__)


# ── 安全路径 Agent 白名单（豁免并发上限）────────────────────────────────────

SAFETY_EXEMPT_AGENTS: frozenset = frozenset({"safety", "review"})


# ── BudgetDecision ─────────────────────────────────────────────────────────


class BudgetDecision(str, Enum):
    """预算决策结果"""

    ACCEPTED = "accepted"      # 正常接受
    REJECTED = "rejected"      # 拒绝（并发超限且非安全路径）
    DEGRADED = "degraded"      # 预算超限但允许（安全路径/降级模式）

    @property
    def is_allowed(self) -> bool:
        return self in (BudgetDecision.ACCEPTED, BudgetDecision.DEGRADED)


# ── RunBudget 配置 ─────────────────────────────────────────────────────────


@dataclass
class RunBudget:
    """单次评估 run 的预算配置"""

    max_total_tokens: int = 200_000          # 单次 run 最大 token
    max_total_cost: float = 10.0             # 单次 run 最大成本（元）
    max_parallel_agents: int = 5             # 最大并发 Agent 数
    max_duration_seconds: float = 600.0      # 最大运行时长（秒）

    # 每模型成本上限（0 = 不限）
    max_per_model_cost: float = 0.0


# ── UsageRecord ────────────────────────────────────────────────────────────


@dataclass
class UsageRecord:
    """单次 LLM 调用用量记录"""

    tokens: int = 0
    cost: float = 0.0
    model: str = ""
    latency_ms: float = 0.0


# ── CacheKeyBuilder ────────────────────────────────────────────────────────


class CacheKeyBuilder:
    """版本感知的 LLM 缓存键构建器

    确保不同 model_version / prompt_version / kb_version / temperature / tenant
    之间不会发生缓存污染。
    """

    PREFIX = "llm_vcache"

    @staticmethod
    def build(
        model_version: str,
        prompt_version: str,
        kb_version: str,
        temperature: float,
        tenant: str,
        content_hash: str,
    ) -> str:
        """生成确定性缓存键"""
        raw = f"{model_version}|{prompt_version}|{kb_version}|{temperature}|{tenant}|{content_hash}"
        digest = hashlib.sha256(raw.encode("utf-8")).hexdigest()[:20]
        return f"{CacheKeyBuilder.PREFIX}:{digest}"


# ── RunBudgetManager ───────────────────────────────────────────────────────


class RunBudgetManager:
    """管理评估 run 的并发 slot、Token 和成本预算

    线程安全：使用 threading.Lock 保护内部状态。
    """

    def __init__(self, budget: Optional[RunBudget] = None):
        self.budget = budget or RunBudget()
        self._lock = threading.Lock()

        # run_id -> set of agent_names (active slots)
        self._active_slots: Dict[str, Set[str]] = {}

        # run_id -> list of UsageRecord
        self._usage_records: Dict[str, List[UsageRecord]] = {}

        # run_id -> total tokens
        self._total_tokens: Dict[str, int] = {}

        # run_id -> total cost
        self._total_cost: Dict[str, float] = {}

        # run_id -> per-model cost
        self._model_cost: Dict[str, Dict[str, float]] = {}

    def acquire_agent_slot(
        self, run_id: str, agent_name: str
    ) -> BudgetDecision:
        """获取 Agent slot

        Returns:
            BudgetDecision.ACCEPTED — 正常获取
            BudgetDecision.REJECTED — 并发超限且非安全路径
            BudgetDecision.DEGRADED — 预算超限但安全路径豁免
        """
        with self._lock:
            # 安全路径豁免并发上限
            is_exempt = agent_name in SAFETY_EXEMPT_AGENTS

            slots = self._active_slots.setdefault(run_id, set())

            # 已在运行中则直接接受
            if agent_name in slots:
                return BudgetDecision.ACCEPTED

            # 检查并发上限
            if not is_exempt and len(slots) >= self.budget.max_parallel_agents:
                logger.warning(
                    f"[RunBudget] 并发上限: run={run_id}, agent={agent_name}, "
                    f"active={len(slots)}/{self.budget.max_parallel_agents}"
                )
                return BudgetDecision.REJECTED

            # 检查 token 预算
            total_tokens = self._total_tokens.get(run_id, 0)
            if total_tokens >= self.budget.max_total_tokens:
                if is_exempt:
                    logger.info(
                        f"[RunBudget] token 超限但安全路径豁免: run={run_id}, agent={agent_name}"
                    )
                    slots.add(agent_name)
                    return BudgetDecision.DEGRADED
                return BudgetDecision.DEGRADED

            slots.add(agent_name)
            return BudgetDecision.ACCEPTED

    def release_agent_slot(self, run_id: str, agent_name: str) -> None:
        """释放 Agent slot"""
        with self._lock:
            slots = self._active_slots.get(run_id)
            if slots and agent_name in slots:
                slots.discard(agent_name)

    def cancel_run(self, run_id: str) -> None:
        """取消 run，释放所有 slot"""
        with self._lock:
            self._active_slots.pop(run_id, None)

    def record_usage(
        self, run_id: str, usage: UsageRecord
    ) -> BudgetDecision:
        """记录 LLM 调用用量

        Returns:
            BudgetDecision.ACCEPTED — 预算内
            BudgetDecision.DEGRADED — 超预算
        """
        with self._lock:
            records = self._usage_records.setdefault(run_id, [])
            records.append(usage)

            total_tokens = self._total_tokens.get(run_id, 0) + usage.tokens
            self._total_tokens[run_id] = total_tokens

            total_cost = self._total_cost.get(run_id, 0.0) + usage.cost
            self._total_cost[run_id] = total_cost

            # 按模型累计成本
            if usage.model:
                model_costs = self._model_cost.setdefault(run_id, {})
                model_costs[usage.model] = model_costs.get(usage.model, 0.0) + usage.cost

            # 检查是否超预算
            over_tokens = total_tokens > self.budget.max_total_tokens
            over_cost = total_cost > self.budget.max_total_cost
            over_model_cost = (
                self.budget.max_per_model_cost > 0
                and usage.model
                and self._model_cost.get(run_id, {}).get(usage.model, 0.0)
                > self.budget.max_per_model_cost
            )

            if over_tokens or over_cost or over_model_cost:
                logger.warning(
                    f"[RunBudget] 预算超限: run={run_id}, "
                    f"tokens={total_tokens}/{self.budget.max_total_tokens}, "
                    f"cost={total_cost:.4f}/{self.budget.max_total_cost}"
                )
                return BudgetDecision.DEGRADED

            return BudgetDecision.ACCEPTED

    @contextmanager
    def slot_context(self, run_id: str, agent_name: str):
        """上下文管理器：自动获取和释放 slot

        Usage:
            with mgr.slot_context(run_id, "inquiry") as decision:
                # 执行 agent
                ...
            # 异常或正常退出都会释放 slot
        """
        decision = self.acquire_agent_slot(run_id, agent_name)
        try:
            yield decision
        finally:
            self.release_agent_slot(run_id, agent_name)

    def get_run_summary(self, run_id: str) -> dict:
        """获取 run 预算摘要"""
        with self._lock:
            slots = self._active_slots.get(run_id, set())
            records = self._usage_records.get(run_id, [])
            total_tokens = self._total_tokens.get(run_id, 0)
            total_cost = self._total_cost.get(run_id, 0.0)

            return {
                "run_id": run_id,
                "total_tokens": total_tokens,
                "total_cost": round(total_cost, 6),
                "max_total_tokens": self.budget.max_total_tokens,
                "max_total_cost": self.budget.max_total_cost,
                "active_agents": sorted(slots),
                "usage_count": len(records),
                "budget_exceeded": (
                    total_tokens > self.budget.max_total_tokens
                    or total_cost > self.budget.max_total_cost
                ),
            }
