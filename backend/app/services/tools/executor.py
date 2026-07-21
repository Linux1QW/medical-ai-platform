import asyncio
import json
import logging
import time
import uuid

from .base import ToolContext
from .budget import ToolBudget
from .registry import ToolRegistry

logger = logging.getLogger(__name__)


class ToolExecutor:
    """统一工具执行器

    支持可选的加固模式：
    - 重试机制（指数退避）
    - 熔断器（防止级联故障）
    - 结果验证
    - 健康检查集成

    加固功能默认关闭以保持向后兼容，通过 RobustToolExecutor 启用完整加固。
    """

    def __init__(
        self,
        registry: ToolRegistry,
        max_result_chars: int = 6000,
        *,
        enable_retry: bool = False,
        max_retries: int = 2,
        retry_base_delay: float = 0.5,
        enable_circuit_breaker: bool = False,
        circuit_breaker_threshold: int = 5,
    ):
        self.registry = registry
        self.max_result_chars = max_result_chars
        self.traces: list[dict] = []  # 执行 trace 记录

        # 加固配置
        self.enable_retry = enable_retry
        self.max_retries = max_retries
        self.retry_base_delay = retry_base_delay
        self.enable_circuit_breaker = enable_circuit_breaker
        self.circuit_breaker_threshold = circuit_breaker_threshold

        # 熔断器状态（每个工具独立维护）
        self._circuit_failures: dict[str, int] = {}
        self._circuit_open_until: dict[str, float] = {}

    async def execute(
        self,
        tool_name: str,
        arguments_json: str,
        context: ToolContext | None = None,
        budget: ToolBudget | None = None,
    ) -> dict:
        """
        执行工具调用

        Args:
            tool_name: 工具名称
            arguments_json: JSON 字符串格式的参数
            context: 工具上下文
            budget: 预算控制器

        Returns:
            统一格式 dict: {"ok": bool, "data": ..., "error": ..., "degraded": bool, "trace_id": str}
        """
        trace_id = f"tool-{uuid.uuid4().hex[:8]}"
        start_time = time.monotonic()

        # 1. 白名单检查
        tool = self.registry.get(tool_name)
        if tool is None:
            elapsed = (time.monotonic() - start_time) * 1000
            self._record_trace(trace_id, tool_name, {}, "error", elapsed, error=f"Unknown tool: {tool_name}")
            return self._error_result(trace_id, "unknown_tool", f"Tool '{tool_name}' is not registered")

        # 2. 预算检查
        if budget and not budget.check(tool_name):
            elapsed = (time.monotonic() - start_time) * 1000
            self._record_trace(trace_id, tool_name, {}, "budget_exceeded", elapsed, error="Budget exceeded")
            return self._error_result(trace_id, "budget_exceeded", f"Tool '{tool_name}' budget exceeded")

        # 2.5 熔断器检查
        if self.enable_circuit_breaker and not self._circuit_allow_request(tool_name):
            elapsed = (time.monotonic() - start_time) * 1000
            self._record_trace(trace_id, tool_name, {}, "circuit_open", elapsed,
                               error=f"Circuit breaker OPEN for {tool_name}")
            return self._error_result(
                trace_id, "circuit_open",
                f"Tool '{tool_name}' circuit breaker is OPEN",
                degraded=tool.critical,
            )

        # 3. 参数解析和校验
        try:
            args_dict = json.loads(arguments_json) if isinstance(arguments_json, str) else arguments_json
        except json.JSONDecodeError as e:
            elapsed = (time.monotonic() - start_time) * 1000
            self._record_trace(trace_id, tool_name, {"raw": arguments_json[:200]}, "error", elapsed, error=f"JSON parse error: {e}")
            return self._error_result(trace_id, "invalid_arguments", f"Invalid JSON arguments: {e}")

        if tool.args_schema:
            try:
                validated_args = tool.args_schema(**args_dict)
            except Exception as e:
                elapsed = (time.monotonic() - start_time) * 1000
                self._record_trace(trace_id, tool_name, args_dict, "error", elapsed, error=f"Validation error: {e}")
                return self._error_result(trace_id, "validation_error", f"Argument validation failed: {e}")
        else:
            validated_args = args_dict

        # 4. 执行工具（带重试和超时）
        max_attempts = (self.max_retries + 1) if self.enable_retry else 1
        last_error = None

        for attempt in range(max_attempts):
            if attempt > 0:
                import random
                delay = min(
                    self.retry_base_delay * (2 ** (attempt - 1)),
                    10.0,
                ) * (0.5 + random.random())
                logger.info(f"[ToolExecutor] 重试 {tool_name} 第 {attempt} 次，等待 {delay:.2f}s")
                await asyncio.sleep(delay)

            try:
                timeout = tool.timeout_seconds
                result = await asyncio.wait_for(
                    tool.execute(validated_args, context or ToolContext()),
                    timeout=timeout,
                )
                elapsed = (time.monotonic() - start_time) * 1000

                # 5. 结果截断
                result_str = json.dumps(result, ensure_ascii=False, default=str)
                if len(result_str) > self.max_result_chars:
                    result_str = result_str[: self.max_result_chars] + "...[truncated]"
                    result = {"truncated": True, "data": result_str}

                # 6. 消耗预算
                if budget:
                    budget.consume(tool_name)

                # 7. 熔断器记录成功
                if self.enable_circuit_breaker:
                    self._circuit_record_success(tool_name)

                # 8. 记录 trace
                args_summary = {k: str(v)[:100] for k, v in (args_dict if isinstance(args_dict, dict) else {}).items()}
                self._record_trace(trace_id, tool_name, args_summary, "success", elapsed,
                                   retries=attempt)

                return {
                    "ok": True,
                    "data": result,
                    "error": None,
                    "degraded": False,
                    "trace_id": trace_id,
                    "retries": attempt,
                }

            except asyncio.TimeoutError:
                last_error = f"Timeout after {tool.timeout_seconds}s"
                if self.enable_circuit_breaker:
                    self._circuit_record_failure(tool_name)
                logger.warning(
                    f"[ToolExecutor] {tool_name} 超时 ({tool.timeout_seconds}s)，"
                    f"attempt={attempt + 1}/{max_attempts}"
                )
                if attempt >= max_attempts - 1:
                    elapsed = (time.monotonic() - start_time) * 1000
                    self._record_trace(
                        trace_id, tool_name, args_dict if isinstance(args_dict, dict) else {},
                        "timeout", elapsed, error=last_error, retries=attempt,
                    )
                    return self._error_result(
                        trace_id, "timeout",
                        f"Tool '{tool_name}' timed out after {tool.timeout_seconds}s",
                        degraded=tool.critical,
                        retries=attempt,
                    )

            except Exception as e:
                last_error = str(e)
                if self.enable_circuit_breaker:
                    self._circuit_record_failure(tool_name)
                logger.exception(f"Tool '{tool_name}' execution failed")
                if attempt >= max_attempts - 1:
                    elapsed = (time.monotonic() - start_time) * 1000
                    self._record_trace(
                        trace_id, tool_name, args_dict if isinstance(args_dict, dict) else {},
                        "error", elapsed, error=last_error, retries=attempt,
                    )
                    return self._error_result(
                        trace_id, "execution_error",
                        f"Tool '{tool_name}' failed: {e}",
                        degraded=tool.critical,
                        retries=attempt,
                    )

        # 兜底
        elapsed = (time.monotonic() - start_time) * 1000
        return self._error_result(
            trace_id, "max_retries_exceeded",
            f"Tool '{tool_name}' exceeded max retries",
            degraded=True,
        )

    # ── 熔断器辅助方法 ────────────────────────────────────────────────────────

    def _circuit_allow_request(self, tool_name: str) -> bool:
        """检查熔断器是否允许请求通过"""
        open_until = self._circuit_open_until.get(tool_name, 0)
        if time.monotonic() < open_until:
            return False
        # 超过恢复时间（30秒），允许试探
        return True

    def _circuit_record_success(self, tool_name: str) -> None:
        """记录成功，重置失败计数"""
        self._circuit_failures[tool_name] = 0

    def _circuit_record_failure(self, tool_name: str) -> None:
        """记录失败，达到阈值时开启熔断"""
        self._circuit_failures[tool_name] = self._circuit_failures.get(tool_name, 0) + 1
        if self._circuit_failures[tool_name] >= self.circuit_breaker_threshold:
            self._circuit_open_until[tool_name] = time.monotonic() + 30.0
            logger.warning(
                f"[ToolExecutor] 熔断器开启: {tool_name} "
                f"(连续失败 {self._circuit_failures[tool_name]} 次)"
            )

    # ── 通用辅助方法 ──────────────────────────────────────────────────────────

    def _error_result(self, trace_id: str, code: str, message: str,
                      degraded: bool = False, retries: int = 0) -> dict:
        return {
            "ok": False,
            "data": None,
            "error": {"code": code, "message": message},
            "degraded": degraded,
            "trace_id": trace_id,
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

    def get_traces(self) -> list[dict]:
        return list(self.traces)

    def clear_traces(self) -> None:
        self.traces.clear()
