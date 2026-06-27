# -*- coding: utf-8 -*-
"""工具预算管理器 — 控制工具使用成本与配额

核心功能：
1. 单次会话预算：限制单次对话的工具调用次数
2. 总体预算：限制全局工具调用总量
3. 成本跟踪：记录每次工具调用的耗时和估算成本
4. 预警机制：预算即将耗尽时触发预警
5. 动态调整：根据运行状态动态调整预算分配
"""
from __future__ import annotations

import time
import logging
from dataclasses import dataclass, field
from typing import Any, Optional, TYPE_CHECKING
from enum import Enum

if TYPE_CHECKING:
    from .budget import ToolBudget

logger = logging.getLogger(__name__)


# ── 预警级别 ──────────────────────────────────────────────────────────────────


class BudgetAlertLevel(Enum):
    NORMAL = "normal"          # 正常
    WARNING = "warning"        # 预算使用超过 70%
    CRITICAL = "critical"      # 预算使用超过 90%
    EXHAUSTED = "exhausted"    # 预算已耗尽


# ── 成本配置 ──────────────────────────────────────────────────────────────────


@dataclass
class ToolCostConfig:
    """工具成本配置（单位：元/次调用）

    用于估算工具调用的经济成本，便于做成本效益分析。
    默认所有工具成本为 0（本地调用无额外费用）。
    """
    # LLM 相关工具（涉及外部 API 调用）
    expand_query_cost: float = 0.002        # MQE 涉及 LLM 调用
    generate_hyde_query_cost: float = 0.003  # HyDE 涉及 LLM 调用
    rerank_evidence_cost: float = 0.001      # Reranker 调用

    # 本地工具（向量检索等）
    search_medical_kb_cost: float = 0.0     # ChromaDB 本地检索
    default_cost: float = 0.0               # 默认成本

    def get_cost(self, tool_name: str) -> float:
        """获取指定工具的调用成本"""
        cost_attr = f"{tool_name}_cost"
        return getattr(self, cost_attr, self.default_cost)


# ── 预算配置 ──────────────────────────────────────────────────────────────────


@dataclass
class BudgetConfig:
    """预算配置"""
    # 单次会话限制
    session_max_total_calls: int = 50        # 单次会话最大总调用次数
    session_max_per_tool: int = 10           # 单次会话单工具最大调用次数
    session_max_cost: float = 0.1            # 单次会话最大成本（元）

    # 全局限制（0 = 不限制）
    global_max_calls: int = 0                # 全局最大调用次数
    global_max_cost: float = 0.0             # 全局最大成本

    # 预警阈值
    warning_threshold: float = 0.7           # 70% 时预警
    critical_threshold: float = 0.9          # 90% 时严重预警

    # 每工具默认配额
    default_per_tool_limits: dict[str, int] = field(default_factory=lambda: {
        "search_medical_kb": 8,
        "expand_query": 5,
        "generate_hyde_query": 3,
        "rerank_evidence": 5,
    })


# ── 调用记录 ──────────────────────────────────────────────────────────────────


@dataclass
class ToolCallRecord:
    """单次工具调用记录"""
    tool_name: str
    timestamp: float
    latency_ms: float
    cost: float
    success: bool
    session_id: str
    trace_id: str = ""


# ── 会话预算 ──────────────────────────────────────────────────────────────────


@dataclass
class SessionBudget:
    """单个会话的预算状态"""
    session_id: str
    per_tool_limits: dict[str, int]
    per_tool_used: dict[str, int] = field(default_factory=dict)
    total_calls: int = 0
    total_cost: float = 0.0
    created_at: float = field(default_factory=time.time)
    records: list[ToolCallRecord] = field(default_factory=list)

    def check_tool(self, tool_name: str) -> bool:
        """检查工具是否还有预算"""
        # 检查单工具配额
        limit = self.per_tool_limits.get(tool_name)
        if limit is not None:
            if self.per_tool_used.get(tool_name, 0) >= limit:
                return False
        return True

    def check_total(self, max_total: int) -> bool:
        """检查总调用次数是否超限"""
        if max_total <= 0:
            return True  # 0 = 不限制
        return self.total_calls < max_total

    def check_cost(self, max_cost: float) -> bool:
        """检查成本是否超限"""
        if max_cost <= 0:
            return True
        return self.total_cost < max_cost

    def consume(self, tool_name: str, cost: float, record: ToolCallRecord) -> None:
        """消耗预算"""
        self.total_calls += 1
        self.total_cost += cost
        self.per_tool_used[tool_name] = self.per_tool_used.get(tool_name, 0) + 1
        self.records.append(record)

    def remaining(self, tool_name: str) -> int:
        """获取工具剩余配额"""
        limit = self.per_tool_limits.get(tool_name)
        if limit is None:
            return -1  # 未设限
        return max(0, limit - self.per_tool_used.get(tool_name, 0))

    def get_alert_level(self, config: BudgetConfig) -> BudgetAlertLevel:
        """获取当前预警级别"""
        # 检查总调用次数
        if config.session_max_total_calls > 0:
            ratio = self.total_calls / config.session_max_total_calls
            if ratio >= 1.0:
                return BudgetAlertLevel.EXHAUSTED
            if ratio >= config.critical_threshold:
                return BudgetAlertLevel.CRITICAL
            if ratio >= config.warning_threshold:
                return BudgetAlertLevel.WARNING

        # 检查成本
        if config.session_max_cost > 0 and self.total_cost > 0:
            cost_ratio = self.total_cost / config.session_max_cost
            if cost_ratio >= 1.0:
                return BudgetAlertLevel.EXHAUSTED
            if cost_ratio >= config.critical_threshold:
                return BudgetAlertLevel.CRITICAL
            if cost_ratio >= config.warning_threshold:
                return BudgetAlertLevel.WARNING

        return BudgetAlertLevel.NORMAL

    def summary(self) -> dict:
        """获取预算摘要"""
        return {
            "session_id": self.session_id,
            "total_calls": self.total_calls,
            "total_cost": round(self.total_cost, 6),
            "per_tool_used": dict(self.per_tool_used),
            "per_tool_remaining": {
                name: self.remaining(name)
                for name in self.per_tool_limits
            },
            "uptime_seconds": round(time.time() - self.created_at, 1),
        }


# ── 预算管理器 ─────────────────────────────────────────────────────────────────


class ToolBudgetManager:
    """工具预算管理器

    管理会话级和全局级的工具调用预算，提供成本跟踪和预警机制。

    用法：
        manager = ToolBudgetManager()
        session = manager.get_or_create_session("session-123")

        # 执行前检查
        if manager.check_budget("search_medical_kb", session):
            # 执行工具...
            manager.record_call("search_medical_kb", session, latency_ms=100, success=True)
    """

    def __init__(
        self,
        config: BudgetConfig | None = None,
        cost_config: ToolCostConfig | None = None,
    ):
        self.config = config or BudgetConfig()
        self.cost_config = cost_config or ToolCostConfig()

        # 会话预算
        self._sessions: dict[str, SessionBudget] = {}

        # 全局统计
        self._global_total_calls: int = 0
        self._global_total_cost: float = 0.0
        self._global_start_time: float = time.time()

        # 预警回调
        self._alert_callbacks: list = []

    def get_or_create_session(
        self,
        session_id: str,
        per_tool_limits: dict[str, int] | None = None,
    ) -> SessionBudget:
        """获取或创建会话预算"""
        if session_id in self._sessions:
            return self._sessions[session_id]

        limits = per_tool_limits or dict(self.config.default_per_tool_limits)
        session = SessionBudget(
            session_id=session_id,
            per_tool_limits=limits,
        )
        self._sessions[session_id] = session
        logger.debug(f"[BudgetManager] 创建会话预算: {session_id}")
        return session

    def check_budget(self, tool_name: str, session: SessionBudget) -> bool:
        """综合检查预算是否充足

        检查顺序：单工具配额 → 会话总次数 → 会话成本 → 全局限制
        """
        # 1. 单工具配额
        if not session.check_tool(tool_name):
            logger.warning(
                f"[BudgetManager] 工具 {tool_name} 会话配额已耗尽 "
                f"(session={session.session_id})"
            )
            return False

        # 2. 会话总次数
        if not session.check_total(self.config.session_max_total_calls):
            logger.warning(
                f"[BudgetManager] 会话总调用次数超限 "
                f"(session={session.session_id}, calls={session.total_calls})"
            )
            return False

        # 3. 会话成本
        if not session.check_cost(self.config.session_max_cost):
            logger.warning(
                f"[BudgetManager] 会话成本超限 "
                f"(session={session.session_id}, cost={session.total_cost:.4f})"
            )
            return False

        # 4. 全局限制
        if self.config.global_max_calls > 0:
            if self._global_total_calls >= self.config.global_max_calls:
                logger.warning("[BudgetManager] 全局调用次数超限")
                return False

        if self.config.global_max_cost > 0:
            if self._global_total_cost >= self.config.global_max_cost:
                logger.warning("[BudgetManager] 全局成本超限")
                return False

        return True

    def record_call(
        self,
        tool_name: str,
        session: SessionBudget,
        *,
        latency_ms: float = 0.0,
        success: bool = True,
        trace_id: str = "",
    ) -> ToolCallRecord:
        """记录一次工具调用，更新预算和统计"""
        cost = self.cost_config.get_cost(tool_name)

        record = ToolCallRecord(
            tool_name=tool_name,
            timestamp=time.time(),
            latency_ms=latency_ms,
            cost=cost,
            success=success,
            session_id=session.session_id,
            trace_id=trace_id,
        )

        session.consume(tool_name, cost, record)
        self._global_total_calls += 1
        self._global_total_cost += cost

        # 检查预警
        alert_level = session.get_alert_level(self.config)
        if alert_level != BudgetAlertLevel.NORMAL:
            self._fire_alert(alert_level, session, tool_name)

        return record

    def get_alert_level(self, session: SessionBudget) -> BudgetAlertLevel:
        """获取会话当前预警级别"""
        return session.get_alert_level(self.config)

    def register_alert_callback(self, callback) -> None:
        """注册预警回调函数

        callback 签名: callback(level: BudgetAlertLevel, session: SessionBudget, tool_name: str)
        """
        self._alert_callbacks.append(callback)

    def _fire_alert(
        self, level: BudgetAlertLevel, session: SessionBudget, tool_name: str
    ) -> None:
        """触发预警"""
        logger.warning(
            f"[BudgetManager] 预算预警 [{level.value}]: "
            f"session={session.session_id}, tool={tool_name}, "
            f"calls={session.total_calls}, cost={session.total_cost:.4f}"
        )
        for cb in self._alert_callbacks:
            try:
                cb(level, session, tool_name)
            except Exception as e:
                logger.error(f"[BudgetManager] 预警回调失败: {e}")

    def get_session_summary(self, session_id: str) -> dict | None:
        """获取会话预算摘要"""
        session = self._sessions.get(session_id)
        if session is None:
            return None
        return session.summary()

    def get_global_stats(self) -> dict:
        """获取全局统计"""
        return {
            "total_sessions": len(self._sessions),
            "global_total_calls": self._global_total_calls,
            "global_total_cost": round(self._global_total_cost, 6),
            "uptime_seconds": round(time.time() - self._global_start_time, 1),
        }

    def cleanup_session(self, session_id: str) -> None:
        """清理过期会话预算"""
        if session_id in self._sessions:
            del self._sessions[session_id]
            logger.debug(f"[BudgetManager] 清理会话预算: {session_id}")

    def cleanup_expired_sessions(self, max_age_seconds: float = 3600.0) -> int:
        """清理所有过期会话，返回清理数量"""
        now = time.time()
        expired = [
            sid for sid, s in self._sessions.items()
            if (now - s.created_at) > max_age_seconds
        ]
        for sid in expired:
            del self._sessions[sid]
        if expired:
            logger.info(f"[BudgetManager] 清理 {len(expired)} 个过期会话预算")
        return len(expired)

    # ── 与旧 ToolBudget 兼容的工厂方法 ──────────────────────────────────────

    def create_tool_budget_for_session(
        self, session_id: str
    ) -> "ToolBudget":
        """为会话创建兼容旧接口的 ToolBudget 实例"""
        from .budget import ToolBudget
        session = self.get_or_create_session(session_id)
        return ToolBudget(session.per_tool_limits)
