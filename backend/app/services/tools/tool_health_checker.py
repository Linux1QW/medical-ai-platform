# -*- coding: utf-8 -*-
"""工具健康检查器 — 定期检查工具可用性，实现降级与备用方案

核心功能：
1. 定期健康检查：探测工具是否可用
2. 工具状态管理：healthy / degraded / unavailable
3. 降级策略：工具不可用时返回降级结果
4. 备用方案：关键工具失败时切换到备用实现
5. 自动恢复：定期检查是否恢复可用
"""

import asyncio
import logging
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable

from .base import BaseTool, ToolContext
from .registry import ToolRegistry

logger = logging.getLogger(__name__)


# ── 工具健康状态 ────────────────────────────────────────────────────────────────


class ToolHealthStatus(Enum):
    HEALTHY = "healthy"          # 健康
    DEGRADED = "degraded"        # 降级（可用但性能下降）
    UNAVAILABLE = "unavailable"  # 不可用
    UNKNOWN = "unknown"          # 未检查


@dataclass
class ToolHealthReport:
    """工具健康检查报告"""
    tool_name: str
    status: ToolHealthStatus
    latency_ms: float
    last_check_time: float
    consecutive_failures: int
    message: str = ""
    details: dict = field(default_factory=dict)


# ── 降级结果构建器 ──────────────────────────────────────────────────────────────


class DegradedResultBuilder:
    """为不同工具构建降级结果"""

    _builders: dict[str, Callable] = {}

    @classmethod
    def register(cls, tool_name: str, builder: Callable[[], dict]) -> None:
        """注册工具降级结果构建函数"""
        cls._builders[tool_name] = builder

    @classmethod
    def build(cls, tool_name: str) -> dict:
        """构建降级结果"""
        builder = cls._builders.get(tool_name)
        if builder:
            return builder()
        # 默认降级结果
        return {
            "degraded": True,
            "message": f"工具 '{tool_name}' 当前不可用，返回降级结果",
            "data": None,
        }

    @classmethod
    def build_with_fallback(cls, tool_name: str, fallback_data: Any = None) -> dict:
        """构建带备用数据的降级结果"""
        builder = cls._builders.get(tool_name)
        if builder:
            result = builder()
            if fallback_data is not None:
                result["fallback_data"] = fallback_data
            return result
        return {
            "degraded": True,
            "message": f"工具 '{tool_name}' 当前不可用",
            "data": fallback_data,
        }


# 注册默认降级结果
DegradedResultBuilder.register("search_medical_kb", lambda: {
    "evidence": [],
    "retrieval_level": "degraded",
    "total_found": 0,
    "degraded": True,
    "message": "医学知识库检索暂不可用，未获取到循证证据",
})

DegradedResultBuilder.register("expand_query", lambda: {
    "expanded_queries": [],
    "original_query": "",
    "degraded": True,
    "message": "查询扩展暂不可用",
})

DegradedResultBuilder.register("generate_hyde_query", lambda: {
    "hyde_query": "",
    "query_type": "",
    "degraded": True,
    "message": "HyDE 查询生成暂不可用",
})

DegradedResultBuilder.register("rerank_evidence", lambda: {
    "reranked_evidence": [],
    "total_candidates": 0,
    "returned": 0,
    "degraded": True,
    "message": "证据重排序暂不可用",
})


# ── 健康检查器 ──────────────────────────────────────────────────────────────────


@dataclass
class HealthCheckConfig:
    """健康检查配置"""
    check_interval: float = 60.0         # 检查间隔（秒）
    check_timeout: float = 5.0           # 单次检查超时（秒）
    unhealthy_threshold: int = 3         # 连续失败次数阈值 → 标记为 unavailable
    recovery_threshold: int = 2          # 连续成功次数阈值 → 恢复为 healthy
    probe_query: str = "健康检查"         # 探测查询


class ToolHealthChecker:
    """工具健康检查器

    定期检查注册工具的可用性，维护健康状态，
    在工具不可用时提供降级结果和备用方案。

    用法：
        checker = ToolHealthChecker(registry)
        await checker.start()  # 启动后台检查

        # 执行工具前检查
        if checker.is_healthy("search_medical_kb"):
            result = await executor.execute(...)
        else:
            result = checker.get_degraded_result("search_medical_kb")

        await checker.stop()  # 停止检查
    """

    def __init__(
        self,
        registry: ToolRegistry,
        config: HealthCheckConfig | None = None,
    ):
        self.registry = registry
        self.config = config or HealthCheckConfig()

        # 工具健康状态
        self._status: dict[str, ToolHealthStatus] = {}
        self._consecutive_failures: dict[str, int] = {}
        self._consecutive_successes: dict[str, int] = {}
        self._last_check_time: dict[str, float] = {}
        self._last_latency: dict[str, float] = {}
        self._last_message: dict[str, str] = {}

        # 备用工具映射：主工具 → 备用工具名
        self._fallback_map: dict[str, str] = {}

        # 后台任务
        self._check_task: asyncio.Task | None = None
        self._running = False

        # 初始化所有已注册工具的状态
        for tool_name in registry.list_tools():
            self._status[tool_name] = ToolHealthStatus.UNKNOWN
            self._consecutive_failures[tool_name] = 0
            self._consecutive_successes[tool_name] = 0

    def register_fallback(self, primary_tool: str, fallback_tool: str) -> None:
        """注册备用工具映射"""
        self._fallback_map[primary_tool] = fallback_tool
        logger.info(f"[HealthChecker] 注册备用: {primary_tool} → {fallback_tool}")

    async def start(self) -> None:
        """启动后台健康检查"""
        if self._running:
            return
        self._running = True
        self._check_task = asyncio.create_task(self._check_loop())
        logger.info("[HealthChecker] 后台健康检查已启动")

    async def stop(self) -> None:
        """停止后台健康检查"""
        self._running = False
        if self._check_task:
            self._check_task.cancel()
            try:
                await self._check_task
            except asyncio.CancelledError:
                pass
            self._check_task = None
        logger.info("[HealthChecker] 后台健康检查已停止")

    async def check_tool(self, tool_name: str) -> ToolHealthReport:
        """对单个工具执行健康检查"""
        tool = self.registry.get(tool_name)
        if tool is None:
            report = ToolHealthReport(
                tool_name=tool_name,
                status=ToolHealthStatus.UNAVAILABLE,
                latency_ms=0,
                last_check_time=time.time(),
                consecutive_failures=self._consecutive_failures.get(tool_name, 0),
                message="工具未注册",
            )
            self._update_status(tool_name, report)
            return report

        start = time.monotonic()
        try:
            # 构造探测参数（根据工具类型）
            probe_args = self._build_probe_args(tool)
            context = ToolContext(run_id="health-check", agent_name="health_checker")

            result = await asyncio.wait_for(
                tool.execute(probe_args, context),
                timeout=self.config.check_timeout,
            )
            latency = (time.monotonic() - start) * 1000

            # 检查结果有效性
            if isinstance(result, dict) and result.get("degraded") is True:
                status = ToolHealthStatus.DEGRADED
                message = "工具返回降级结果"
            else:
                status = ToolHealthStatus.HEALTHY
                message = "OK"

            report = ToolHealthReport(
                tool_name=tool_name,
                status=status,
                latency_ms=latency,
                last_check_time=time.time(),
                consecutive_failures=0,
                message=message,
            )

        except asyncio.TimeoutError:
            latency = (time.monotonic() - start) * 1000
            report = ToolHealthReport(
                tool_name=tool_name,
                status=ToolHealthStatus.UNAVAILABLE,
                latency_ms=latency,
                last_check_time=time.time(),
                consecutive_failures=self._consecutive_failures.get(tool_name, 0) + 1,
                message=f"健康检查超时 ({self.config.check_timeout}s)",
            )

        except Exception as e:
            latency = (time.monotonic() - start) * 1000
            report = ToolHealthReport(
                tool_name=tool_name,
                status=ToolHealthStatus.UNAVAILABLE,
                latency_ms=latency,
                last_check_time=time.time(),
                consecutive_failures=self._consecutive_failures.get(tool_name, 0) + 1,
                message=f"健康检查异常: {e}",
            )

        self._update_status(tool_name, report)
        return report

    def is_healthy(self, tool_name: str) -> bool:
        """检查工具是否健康"""
        status = self._status.get(tool_name, ToolHealthStatus.UNKNOWN)
        return status in (ToolHealthStatus.HEALTHY, ToolHealthStatus.UNKNOWN)

    def get_status(self, tool_name: str) -> ToolHealthStatus:
        """获取工具健康状态"""
        return self._status.get(tool_name, ToolHealthStatus.UNKNOWN)

    def get_degraded_result(self, tool_name: str) -> dict:
        """获取工具降级结果"""
        result = DegradedResultBuilder.build(tool_name)

        # 检查是否有备用工具
        fallback = self._fallback_map.get(tool_name)
        if fallback:
            fallback_tool = self.registry.get(fallback)
            if fallback_tool and self.is_healthy(fallback):
                result["fallback_available"] = True
                result["fallback_tool"] = fallback
            else:
                result["fallback_available"] = False

        return result

    def get_all_health(self) -> dict[str, dict]:
        """获取所有工具的健康状态摘要"""
        result = {}
        for tool_name in self.registry.list_tools():
            result[tool_name] = {
                "status": self._status.get(tool_name, ToolHealthStatus.UNKNOWN).value,
                "last_check_time": self._last_check_time.get(tool_name, 0),
                "last_latency_ms": round(self._last_latency.get(tool_name, 0), 2),
                "consecutive_failures": self._consecutive_failures.get(tool_name, 0),
                "last_message": self._last_message.get(tool_name, ""),
                "has_fallback": tool_name in self._fallback_map,
            }
        return result

    # ── 内部方法 ──────────────────────────────────────────────────────────────

    def _update_status(self, tool_name: str, report: ToolHealthReport) -> None:
        """根据检查报告更新工具状态"""
        self._last_check_time[tool_name] = report.last_check_time
        self._last_latency[tool_name] = report.latency_ms
        self._last_message[tool_name] = report.message

        if report.status in (ToolHealthStatus.UNAVAILABLE, ToolHealthStatus.DEGRADED):
            self._consecutive_failures[tool_name] = \
                self._consecutive_failures.get(tool_name, 0) + 1
            self._consecutive_successes[tool_name] = 0

            failures = self._consecutive_failures[tool_name]
            if failures >= self.config.unhealthy_threshold:
                self._status[tool_name] = ToolHealthStatus.UNAVAILABLE
                if report.status == ToolHealthStatus.DEGRADED:
                    self._status[tool_name] = ToolHealthStatus.DEGRADED
            else:
                self._status[tool_name] = ToolHealthStatus.DEGRADED

            logger.warning(
                f"[HealthChecker] {tool_name} 健康检查失败 "
                f"(连续{failures}次): {report.message}"
            )

        else:  # HEALTHY
            self._consecutive_successes[tool_name] = \
                self._consecutive_successes.get(tool_name, 0) + 1
            self._consecutive_failures[tool_name] = 0

            successes = self._consecutive_successes[tool_name]
            if successes >= self.config.recovery_threshold:
                self._status[tool_name] = ToolHealthStatus.HEALTHY

            if self._status.get(tool_name) != ToolHealthStatus.HEALTHY:
                logger.info(f"[HealthChecker] {tool_name} 健康检查通过，恢复为 HEALTHY")
            self._status[tool_name] = ToolHealthStatus.HEALTHY

    def _build_probe_args(self, tool: BaseTool):
        """根据工具类型构建探测参数"""
        from .medical_retrieval import (
            ExpandQueryArgs,
            GenerateHydeQueryArgs,
            RerankEvidenceArgs,
            SearchMedicalKBArgs,
        )

        probe = self.config.probe_query

        if tool.name == "search_medical_kb":
            return SearchMedicalKBArgs(query=probe, top_k=1)
        elif tool.name == "expand_query":
            return ExpandQueryArgs(original_query=probe, max_queries=1)
        elif tool.name == "generate_hyde_query":
            return GenerateHydeQueryArgs(case_summary=probe, query_type="case")
        elif tool.name == "rerank_evidence":
            return RerankEvidenceArgs(query=probe, candidate_citation_ids=[], top_k=1)
        else:
            # 通用探测：返回空参数
            if tool.args_schema:
                try:
                    return tool.args_schema()
                except Exception:
                    pass
            return {}

    async def _check_loop(self) -> None:
        """后台检查循环"""
        while self._running:
            try:
                tools = self.registry.list_tools()
                for tool_name in tools:
                    if not self._running:
                        break
                    await self.check_tool(tool_name)

                await asyncio.sleep(self.config.check_interval)

            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"[HealthChecker] 健康检查循环异常: {e}")
                await asyncio.sleep(self.config.check_interval)
