import asyncio
import json
import time
import uuid
import logging
from typing import Any

from .base import ToolContext
from .registry import ToolRegistry
from .budget import ToolBudget

logger = logging.getLogger(__name__)


class ToolExecutor:
    """统一工具执行器"""

    def __init__(self, registry: ToolRegistry, max_result_chars: int = 6000):
        self.registry = registry
        self.max_result_chars = max_result_chars
        self.traces: list[dict] = []  # 执行 trace 记录

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

        # 4. 执行工具（带超时）
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

            # 7. 记录 trace
            args_summary = {k: str(v)[:100] for k, v in (args_dict if isinstance(args_dict, dict) else {}).items()}
            self._record_trace(trace_id, tool_name, args_summary, "success", elapsed)

            return {
                "ok": True,
                "data": result,
                "error": None,
                "degraded": False,
                "trace_id": trace_id,
            }

        except asyncio.TimeoutError:
            elapsed = (time.monotonic() - start_time) * 1000
            self._record_trace(
                trace_id, tool_name, args_dict if isinstance(args_dict, dict) else {},
                "timeout", elapsed, error=f"Timeout after {tool.timeout_seconds}s",
            )
            return self._error_result(
                trace_id, "timeout",
                f"Tool '{tool_name}' timed out after {tool.timeout_seconds}s",
                degraded=tool.critical,
            )

        except Exception as e:
            elapsed = (time.monotonic() - start_time) * 1000
            logger.exception(f"Tool '{tool_name}' execution failed")
            self._record_trace(
                trace_id, tool_name, args_dict if isinstance(args_dict, dict) else {},
                "error", elapsed, error=str(e),
            )
            return self._error_result(
                trace_id, "execution_error",
                f"Tool '{tool_name}' failed: {e}",
                degraded=tool.critical,
            )

    def _error_result(self, trace_id: str, code: str, message: str, degraded: bool = False) -> dict:
        return {
            "ok": False,
            "data": None,
            "error": {"code": code, "message": message},
            "degraded": degraded,
            "trace_id": trace_id,
        }

    def _record_trace(
        self, trace_id: str, tool_name: str, args: dict,
        status: str, elapsed_ms: float, error: str | None = None,
    ):
        self.traces.append({
            "trace_id": trace_id,
            "tool_name": tool_name,
            "arguments_summary": args,
            "status": status,
            "elapsed_ms": round(elapsed_ms, 2),
            "error": error,
        })

    def get_traces(self) -> list[dict]:
        return list(self.traces)

    def clear_traces(self) -> None:
        self.traces.clear()
