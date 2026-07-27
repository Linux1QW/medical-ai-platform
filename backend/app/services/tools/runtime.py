# -*- coding: utf-8 -*-
"""工具 Harness 运行时 — 进程级共享的加固组件装配

将三个加固组件接入实际执行链路：
1. RobustToolExecutor：重试/熔断/结果校验。熔断器与调用统计使用进程级
   共享字典（跨评估累计才有意义），trace 仍为单次运行独立。
2. ToolBudgetManager：以 run_id 为会话粒度，在单工具配额之上叠加
   会话总次数、成本上限与三级预警。
3. ToolHealthChecker：可选后台探测 + 执行前门控（探测会真实调用工具，
   含 LLM 成本，由 TOOL_HEALTH_CHECK_ENABLED 控制，默认关闭）。

Agent 侧统一通过 create_tool_executor / create_tool_budget 工厂构建，
TOOL_EXECUTOR_HARDENED=False 时回退轻量版，保持向后兼容。
"""

import logging
from typing import Optional

from app.core.config import settings

from .budget import ToolBudget
from .executor import ToolExecutor
from .registry import ToolRegistry
from .robust_tool_executor import (
    CircuitBreaker,
    RetryPolicy,
    RobustToolExecutor,
    ToolCallStats,
)
from .tool_budget_manager import ToolBudgetManager
from .tool_health_checker import HealthCheckConfig, ToolHealthChecker

logger = logging.getLogger(__name__)

# ── 进程级共享状态 ────────────────────────────────────────────────────────────

# 熔断器与调用统计跨评估共享：单次评估内很难触发 5 次连续失败，
# 只有进程级累计才能让熔断器真正保护下游（ChromaDB / LLM API）
_shared_circuit_breakers: dict[str, CircuitBreaker] = {}
_shared_stats: dict[str, ToolCallStats] = {}

# 全局预算管理器（会话 = 一次评估 run）
budget_manager = ToolBudgetManager()

# 健康检查器（仅 TOOL_HEALTH_CHECK_ENABLED=True 时由 lifespan 启动）
_health_checker: Optional[ToolHealthChecker] = None


# ── 执行器工厂 ────────────────────────────────────────────────────────────────


def create_tool_executor(registry: ToolRegistry, max_result_chars: int = 6000):
    """创建工具执行器：按配置返回加固版（共享熔断/统计）或轻量版"""
    if not settings.TOOL_EXECUTOR_HARDENED:
        return ToolExecutor(registry, max_result_chars=max_result_chars)

    return RobustToolExecutor(
        registry,
        max_result_chars=max_result_chars,
        retry_policy=RetryPolicy(
            max_retries=settings.TOOL_EXECUTOR_MAX_RETRIES,
            base_delay=settings.TOOL_EXECUTOR_RETRY_BASE_DELAY,
        ),
        circuit_breaker_threshold=settings.TOOL_CIRCUIT_BREAKER_THRESHOLD,
        circuit_breaker_recovery=settings.TOOL_CIRCUIT_BREAKER_RECOVERY,
        circuit_breakers=_shared_circuit_breakers,
        stats=_shared_stats,
        health_checker=_health_checker,
    )


# ── 预算工厂 ──────────────────────────────────────────────────────────────────


class ManagedToolBudget(ToolBudget):
    """桥接 ToolBudgetManager 的预算实现

    兼容 ToolExecutor 的 check/consume 接口；check 走管理器的综合检查
    （单工具配额 → 会话总次数 → 会话成本 → 全局限制），consume 同时
    维护本地计数（供 remaining/summary）并向管理器记账（成本/预警）。
    """

    def __init__(self, manager: ToolBudgetManager, session_id: str, limits: dict[str, int]):
        super().__init__(limits)
        self._manager = manager
        self._session = manager.get_or_create_session(session_id, per_tool_limits=dict(limits))

    def check(self, tool_name: str) -> bool:
        return self._manager.check_budget(tool_name, self._session)

    def consume(self, tool_name: str) -> None:
        super().consume(tool_name)
        self._manager.record_call(tool_name, self._session)


def create_tool_budget(budgets: dict[str, int], session_id: str) -> ToolBudget:
    """创建预算控制器：按配置返回管理器托管版或纯计数版"""
    if not settings.TOOL_BUDGET_MANAGER_ENABLED:
        return ToolBudget(budgets)
    # 顺带清理过期会话，避免长期运行下会话字典无限增长
    budget_manager.cleanup_expired_sessions()
    return ManagedToolBudget(budget_manager, session_id, budgets)


# ── 健康检查生命周期（由 FastAPI lifespan 调用）──────────────────────────────


async def start_tool_health_checks() -> None:
    """启动后台工具健康探测（TOOL_HEALTH_CHECK_ENABLED=False 时无操作）"""
    global _health_checker
    if not settings.TOOL_HEALTH_CHECK_ENABLED or _health_checker is not None:
        return

    from . import register_all_tools  # 延迟导入避免与包 __init__ 循环依赖

    registry = ToolRegistry()
    register_all_tools(registry)
    _health_checker = ToolHealthChecker(
        registry,
        HealthCheckConfig(check_interval=settings.TOOL_HEALTH_CHECK_INTERVAL),
    )
    await _health_checker.start()
    logger.info("工具健康探测已启动 (interval=%ss)", settings.TOOL_HEALTH_CHECK_INTERVAL)


async def stop_tool_health_checks() -> None:
    """停止后台工具健康探测"""
    global _health_checker
    if _health_checker is not None:
        await _health_checker.stop()
        _health_checker = None


def get_health_checker() -> Optional[ToolHealthChecker]:
    return _health_checker


# ── 可观测性快照 ──────────────────────────────────────────────────────────────


def get_tool_runtime_snapshot() -> dict:
    """返回工具 Harness 运行时快照（供 /health、监控端点使用）"""
    return {
        "hardened_executor": settings.TOOL_EXECUTOR_HARDENED,
        "circuit_breakers": {
            name: cb.snapshot() for name, cb in _shared_circuit_breakers.items()
        },
        "tool_stats": {
            name: {
                "total_calls": s.total_calls,
                "success_rate": round(s.success_rate, 4),
                "avg_latency_ms": round(s.avg_latency_ms, 2),
                "timeout_calls": s.timeout_calls,
                "retry_calls": s.retry_calls,
            }
            for name, s in _shared_stats.items()
        },
        "budget": budget_manager.get_global_stats(),
        "health": _health_checker.get_all_health() if _health_checker else None,
    }
