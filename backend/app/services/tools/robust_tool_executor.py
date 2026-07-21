# -*- coding: utf-8 -*-
"""健壮工具执行器 — 为 Tool Use 系统提供端到端加固机制

核心功能：
1. 重试机制：指数退避 + 抖动，避免雪崩
2. 超时控制：分级超时 + 全局超时
3. 熔断器模式：防止级联故障
4. 监控与日志：全链路 trace
5. 结果验证：确保工具返回值符合预期结构
"""

import asyncio
import enum
import json
import logging
import random
import time
import uuid
from dataclasses import dataclass

from .base import BaseTool, ToolContext
from .budget import ToolBudget
from .registry import ToolRegistry

logger = logging.getLogger(__name__)


# ── 熔断器状态 ─────────────────────────────────────────────────────────────────


class CircuitState(enum.Enum):
    CLOSED = "closed"        # 正常
    OPEN = "open"            # 熔断中，拒绝请求
    HALF_OPEN = "half_open"  # 试探性放行少量请求


@dataclass
class CircuitBreaker:
    """熔断器 — 每个工具独立维护

    当连续失败次数达到 threshold 时进入 OPEN 状态，
    经过 recovery_timeout 秒后自动切换到 HALF_OPEN，
    允许一次试探请求：成功则恢复 CLOSED，失败则重新 OPEN。
    """
    name: str
    failure_threshold: int = 5
    recovery_timeout: float = 30.0  # 秒
    half_open_max_calls: int = 1

    _state: CircuitState = CircuitState.CLOSED
    _failure_count: int = 0
    _success_count_in_half_open: int = 0
    _last_failure_time: float = 0.0
    _state_changed_at: float = 0.0

    @property
    def state(self) -> CircuitState:
        """获取当前状态，自动检查是否应从 OPEN 转为 HALF_OPEN"""
        if self._state == CircuitState.OPEN:
            elapsed = time.monotonic() - self._state_changed_at
            if elapsed >= self.recovery_timeout:
                self._transition_to(CircuitState.HALF_OPEN)
        return self._state

    def allow_request(self) -> bool:
        """判断是否允许本次请求通过"""
        s = self.state
        if s == CircuitState.CLOSED:
            return True
        if s == CircuitState.HALF_OPEN:
            return self._success_count_in_half_open < self.half_open_max_calls
        return False  # OPEN

    def record_success(self) -> None:
        """记录成功"""
        if self._state == CircuitState.HALF_OPEN:
            self._success_count_in_half_open += 1
            if self._success_count_in_half_open >= self.half_open_max_calls:
                self._transition_to(CircuitState.CLOSED)
        elif self._state == CircuitState.CLOSED:
            self._failure_count = 0  # 重置连续失败

    def record_failure(self) -> None:
        """记录失败"""
        self._failure_count += 1
        self._last_failure_time = time.monotonic()
        if self._state == CircuitState.HALF_OPEN:
            self._transition_to(CircuitState.OPEN)
        elif self._state == CircuitState.CLOSED:
            if self._failure_count >= self.failure_threshold:
                self._transition_to(CircuitState.OPEN)

    def _transition_to(self, new_state: CircuitState) -> None:
        old = self._state
        self._state = new_state
        self._state_changed_at = time.monotonic()
        if new_state == CircuitState.CLOSED:
            self._failure_count = 0
            self._success_count_in_half_open = 0
        elif new_state == CircuitState.HALF_OPEN:
            self._success_count_in_half_open = 0
        logger.info(
            f"[CircuitBreaker:{self.name}] {old.value} → {new_state.value} "
            f"(failures={self._failure_count})"
        )

    def snapshot(self) -> dict:
        return {
            "name": self.name,
            "state": self.state.value,
            "failure_count": self._failure_count,
            "last_failure_time": self._last_failure_time,
        }


# ── 重试策略 ──────────────────────────────────────────────────────────────────


@dataclass
class RetryPolicy:
    """重试策略配置"""
    max_retries: int = 2               # 最大重试次数（不含首次调用）
    base_delay: float = 0.5            # 基础延迟（秒）
    max_delay: float = 10.0            # 最大延迟（秒）
    exponential_base: float = 2.0      # 指数底数
    jitter: bool = True                # 是否添加随机抖动
    retryable_exceptions: tuple = (    # 可重试的异常类型
        asyncio.TimeoutError,
        ConnectionError,
        OSError,
    )

    def compute_delay(self, attempt: int) -> float:
        """计算第 attempt 次重试的等待时间（指数退避 + 可选抖动）"""
        delay = min(
            self.base_delay * (self.exponential_base ** attempt),
            self.max_delay,
        )
        if self.jitter:
            delay = delay * (0.5 + random.random())  # 50%-150% 的随机抖动
        return delay


# ── 执行统计 ──────────────────────────────────────────────────────────────────


@dataclass
class ToolCallStats:
    """单个工具的调用统计"""
    total_calls: int = 0
    success_calls: int = 0
    failed_calls: int = 0
    timeout_calls: int = 0
    retry_calls: int = 0
    total_latency_ms: float = 0.0
    max_latency_ms: float = 0.0

    @property
    def avg_latency_ms(self) -> float:
        return self.total_latency_ms / self.total_calls if self.total_calls else 0.0

    @property
    def success_rate(self) -> float:
        return self.success_calls / self.total_calls if self.total_calls else 0.0

    def record(self, latency_ms: float, status: str) -> None:
        self.total_calls += 1
        self.total_latency_ms += latency_ms
        self.max_latency_ms = max(self.max_latency_ms, latency_ms)
        if status == "success":
            self.success_calls += 1
        elif status == "timeout":
            self.timeout_calls += 1
            self.failed_calls += 1
        else:
            self.failed_calls += 1


# ── 结果验证器 ─────────────────────────────────────────────────────────────────


class ResultValidator:
    """工具调用结果验证器"""

    @staticmethod
    def validate(result: dict, tool: BaseTool) -> tuple[bool, str | None]:
        """验证工具返回结果是否符合预期结构

        Returns:
            (is_valid, error_message)
        """
        if not isinstance(result, dict):
            return False, f"工具返回类型错误: 期望 dict，实际 {type(result).__name__}"

        # 检查必要字段：至少包含 ok/data 或 evidence/results 等常见结构
        has_standard_keys = "ok" in result or "data" in result
        has_evidence_keys = "evidence" in result or "results" in result
        has_content = has_standard_keys or has_evidence_keys or len(result) > 0

        if not has_content:
            return False, "工具返回空结果"

        # 如果有 result_schema，进行结构校验
        if tool.result_schema:
            try:
                tool.result_schema(**result)
            except Exception as e:
                return False, f"结果结构校验失败: {e}"

        return True, None


# ── 健壮工具执行器 ─────────────────────────────────────────────────────────────


class RobustToolExecutor:
    """健壮的工具执行器，集成重试、熔断、监控、结果验证

    用法：
        executor = RobustToolExecutor(registry)
        result = await executor.execute("search_medical_kb", '{"query": "..."}', context=ctx)
    """

    def __init__(
        self,
        registry: ToolRegistry,
        *,
        max_result_chars: int = 6000,
        retry_policy: RetryPolicy | None = None,
        circuit_breaker_threshold: int = 5,
        circuit_breaker_recovery: float = 30.0,
        enable_retry: bool = True,
        enable_circuit_breaker: bool = True,
        enable_validation: bool = True,
    ):
        self.registry = registry
        self.max_result_chars = max_result_chars
        self.retry_policy = retry_policy or RetryPolicy()
        self.enable_retry = enable_retry
        self.enable_circuit_breaker = enable_circuit_breaker
        self.enable_validation = enable_validation

        # 每个工具独立的熔断器
        self._circuit_breakers: dict[str, CircuitBreaker] = {}
        self._cb_threshold = circuit_breaker_threshold
        self._cb_recovery = circuit_breaker_recovery

        # 执行统计
        self._stats: dict[str, ToolCallStats] = {}

        # 执行 trace
        self.traces: list[dict] = []

        # 结果验证器
        self._validator = ResultValidator()

    def _get_circuit_breaker(self, tool_name: str) -> CircuitBreaker:
        if tool_name not in self._circuit_breakers:
            self._circuit_breakers[tool_name] = CircuitBreaker(
                name=tool_name,
                failure_threshold=self._cb_threshold,
                recovery_timeout=self._cb_recovery,
            )
        return self._circuit_breakers[tool_name]

    def _get_stats(self, tool_name: str) -> ToolCallStats:
        if tool_name not in self._stats:
            self._stats[tool_name] = ToolCallStats()
        return self._stats[tool_name]

    async def execute(
        self,
        tool_name: str,
        arguments_json: str,
        context: ToolContext | None = None,
        budget: ToolBudget | None = None,
    ) -> dict:
        """执行工具调用（带完整加固机制）

        Returns:
            {"ok": bool, "data": ..., "error": ..., "degraded": bool,
             "trace_id": str, "retries": int, "circuit_state": str}
        """
        trace_id = f"tool-{uuid.uuid4().hex[:8]}"
        start_time = time.monotonic()
        context = context or ToolContext()
        retries_done = 0

        # ── 1. 白名单检查 ──
        tool = self.registry.get(tool_name)
        if tool is None:
            elapsed = self._elapsed(start_time)
            self._record_trace(trace_id, tool_name, {}, "error", elapsed,
                               error=f"Unknown tool: {tool_name}")
            return self._error_result(trace_id, "unknown_tool",
                                      f"Tool '{tool_name}' is not registered")

        # ── 2. 预算检查 ──
        if budget and not budget.check(tool_name):
            elapsed = self._elapsed(start_time)
            self._record_trace(trace_id, tool_name, {}, "budget_exceeded", elapsed,
                               error="Budget exceeded")
            return self._error_result(trace_id, "budget_exceeded",
                                      f"Tool '{tool_name}' budget exceeded")

        # ── 3. 熔断器检查 ──
        cb = self._get_circuit_breaker(tool_name)
        if self.enable_circuit_breaker and not cb.allow_request():
            elapsed = self._elapsed(start_time)
            self._record_trace(trace_id, tool_name, {}, "circuit_open", elapsed,
                               error=f"Circuit breaker OPEN for {tool_name}")
            stats = self._get_stats(tool_name)
            stats.record(elapsed, "failed")
            return self._error_result(
                trace_id, "circuit_open",
                f"Tool '{tool_name}' circuit breaker is OPEN (failures={cb._failure_count})",
                degraded=tool.critical,
                circuit_state=cb.state.value,
            )

        # ── 4. 参数解析和校验 ──
        try:
            args_dict = json.loads(arguments_json) if isinstance(arguments_json, str) else arguments_json
        except json.JSONDecodeError as e:
            elapsed = self._elapsed(start_time)
            self._record_trace(trace_id, tool_name, {"raw": str(arguments_json)[:200]},
                               "error", elapsed, error=f"JSON parse error: {e}")
            return self._error_result(trace_id, "invalid_arguments",
                                      f"Invalid JSON arguments: {e}")

        if tool.args_schema:
            try:
                validated_args = tool.args_schema(**args_dict)
            except Exception as e:
                elapsed = self._elapsed(start_time)
                self._record_trace(trace_id, tool_name, args_dict, "error", elapsed,
                                   error=f"Validation error: {e}")
                return self._error_result(trace_id, "validation_error",
                                          f"Argument validation failed: {e}")
        else:
            validated_args = args_dict

        # ── 5. 执行（带重试） ──
        max_attempts = (self.retry_policy.max_retries + 1) if self.enable_retry else 1

        for attempt in range(max_attempts):
            if attempt > 0:
                retries_done = attempt
                delay = self.retry_policy.compute_delay(attempt - 1)
                logger.info(
                    f"[RobustExecutor] 重试 {tool_name} 第 {attempt} 次，"
                    f"等待 {delay:.2f}s"
                )
                stats = self._get_stats(tool_name)
                stats.retry_calls += 1
                await asyncio.sleep(delay)

            try:
                timeout = tool.timeout_seconds
                result = await asyncio.wait_for(
                    tool.execute(validated_args, context),
                    timeout=timeout,
                )
                elapsed = self._elapsed(start_time)

                # ── 结果验证 ──
                if self.enable_validation and isinstance(result, dict):
                    is_valid, err_msg = self._validator.validate(result, tool)
                    if not is_valid:
                        logger.warning(
                            f"[RobustExecutor] {tool_name} 结果验证失败: {err_msg}"
                        )
                        # 验证失败视为可重试错误
                        if attempt < max_attempts - 1:
                            cb.record_failure()
                            continue
                        # 最后一次仍然失败，标记降级但返回结果
                        result = {**result, "_validation_warning": err_msg}

                # ── 结果截断 ──
                result_str = json.dumps(result, ensure_ascii=False, default=str)
                if len(result_str) > self.max_result_chars:
                    result_str = result_str[:self.max_result_chars] + "...[truncated]"
                    result = {"truncated": True, "data": result_str}

                # ── 消耗预算 ──
                if budget:
                    budget.consume(tool_name)

                # ── 记录成功 ──
                cb.record_success()
                stats = self._get_stats(tool_name)
                stats.record(elapsed, "success")

                args_summary = {
                    k: str(v)[:100]
                    for k, v in (args_dict if isinstance(args_dict, dict) else {}).items()
                }
                self._record_trace(
                    trace_id, tool_name, args_summary, "success", elapsed,
                    retries=retries_done,
                )

                return {
                    "ok": True,
                    "data": result,
                    "error": None,
                    "degraded": False,
                    "trace_id": trace_id,
                    "retries": retries_done,
                    "circuit_state": cb.state.value,
                }

            except asyncio.TimeoutError:
                elapsed = self._elapsed(start_time)
                cb.record_failure()
                stats = self._get_stats(tool_name)
                stats.record(elapsed, "timeout")
                logger.warning(
                    f"[RobustExecutor] {tool_name} 超时 ({tool.timeout_seconds}s)，"
                    f"attempt={attempt + 1}/{max_attempts}"
                )
                if attempt >= max_attempts - 1:
                    self._record_trace(
                        trace_id, tool_name,
                        args_dict if isinstance(args_dict, dict) else {},
                        "timeout", elapsed,
                        error=f"Timeout after {tool.timeout_seconds}s",
                        retries=retries_done,
                    )
                    return self._error_result(
                        trace_id, "timeout",
                        f"Tool '{tool_name}' timed out after {tool.timeout_seconds}s",
                        degraded=tool.critical,
                        circuit_state=cb.state.value,
                        retries=retries_done,
                    )

            except Exception as e:
                elapsed = self._elapsed(start_time)
                cb.record_failure()
                stats = self._get_stats(tool_name)
                stats.record(elapsed, "error")

                # 判断是否可重试
                is_retryable = isinstance(e, self.retry_policy.retryable_exceptions)
                if is_retryable and attempt < max_attempts - 1:
                    logger.warning(
                        f"[RobustExecutor] {tool_name} 可重试异常: {e}，"
                        f"attempt={attempt + 1}/{max_attempts}"
                    )
                    continue

                logger.exception(f"[RobustExecutor] Tool '{tool_name}' execution failed")
                self._record_trace(
                    trace_id, tool_name,
                    args_dict if isinstance(args_dict, dict) else {},
                    "error", elapsed, error=str(e),
                    retries=retries_done,
                )
                return self._error_result(
                    trace_id, "execution_error",
                    f"Tool '{tool_name}' failed: {e}",
                    degraded=tool.critical,
                    circuit_state=cb.state.value,
                    retries=retries_done,
                )

        # 不应到达此处，但作为安全兜底
        elapsed = self._elapsed(start_time)
        return self._error_result(
            trace_id, "max_retries_exceeded",
            f"Tool '{tool_name}' exceeded max retries ({self.retry_policy.max_retries})",
            degraded=True,
            retries=retries_done,
        )

    # ── 监控接口 ──────────────────────────────────────────────────────────────

    def get_stats(self) -> dict[str, dict]:
        """获取所有工具的调用统计"""
        return {
            name: {
                "total_calls": s.total_calls,
                "success_rate": round(s.success_rate, 4),
                "avg_latency_ms": round(s.avg_latency_ms, 2),
                "max_latency_ms": round(s.max_latency_ms, 2),
                "timeout_calls": s.timeout_calls,
                "retry_calls": s.retry_calls,
            }
            for name, s in self._stats.items()
        }

    def get_circuit_breaker_states(self) -> dict[str, dict]:
        """获取所有熔断器状态"""
        return {
            name: cb.snapshot()
            for name, cb in self._circuit_breakers.items()
        }

    def get_traces(self) -> list[dict]:
        return list(self.traces)

    def clear_traces(self) -> None:
        self.traces.clear()

    def reset_stats(self) -> None:
        """重置统计和熔断器"""
        self._stats.clear()
        self._circuit_breakers.clear()

    # ── 内部辅助 ──────────────────────────────────────────────────────────────

    def _elapsed(self, start_time: float) -> float:
        return (time.monotonic() - start_time) * 1000

    def _error_result(
        self, trace_id: str, code: str, message: str,
        degraded: bool = False, circuit_state: str = "closed", retries: int = 0,
    ) -> dict:
        return {
            "ok": False,
            "data": None,
            "error": {"code": code, "message": message},
            "degraded": degraded,
            "trace_id": trace_id,
            "circuit_state": circuit_state,
            "retries": retries,
        }

    def _record_trace(
        self, trace_id: str, tool_name: str, args: dict,
        status: str, elapsed_ms: float, error: str | None = None,
        retries: int = 0,
    ):
        self.traces.append({
            "trace_id": trace_id,
            "tool_name": tool_name,
            "arguments_summary": args,
            "status": status,
            "elapsed_ms": round(elapsed_ms, 2),
            "error": error,
            "retries": retries,
        })
