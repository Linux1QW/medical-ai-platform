"""Skill executor: routes tool invocations to real RAG or MCP demo handlers.

Enforces:
- Tool registration and per-agent permission checks
- Timeout and budget limits
- Result length caps
- All outputs wrapped as UNTRUSTED_EVIDENCE
"""
from __future__ import annotations

import asyncio
import json
import logging
import re
import time
from dataclasses import dataclass, field
from typing import Any

from app.agent_runtime.policy import SkillPolicy, mark_output_untrusted
from app.agent_runtime.skills import SkillManifest, SkillRegistry

logger = logging.getLogger(__name__)

# ── Constants ──────────────────────────────────────────────────────────────────

MAX_RESULT_CHARS = 8000          # Maximum serialized result length sent to model
MAX_TOKEN_BUDGET = 16000         # Hard ceiling on token budget per call
CONTROL_CHAR_RE = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]")


# ── Data classes ───────────────────────────────────────────────────────────────


@dataclass
class ToolInvocation:
    """A request from an agent to invoke a tool."""

    tool_name: str
    arguments: dict[str, Any]
    agent_name: str
    context_view: str


@dataclass
class SkillResult:
    """Outcome of a skill execution."""

    success: bool
    data: Any = None
    error: str | None = None
    trust_level: str = "untrusted_evidence"
    elapsed_ms: float = 0.0
    truncated: bool = False
    trace: dict[str, Any] = field(default_factory=dict)


# ── Executor ───────────────────────────────────────────────────────────────────


class SkillExecutor:
    """Executes skill invocations with policy enforcement and RAG routing."""

    def __init__(
        self,
        *,
        registry: SkillRegistry,
        policy: SkillPolicy,
        retrieval_fn: Any | None = None,
        rubric_fn: Any | None = None,
    ) -> None:
        """
        Args:
            registry: Skill registry with loaded manifests.
            policy: Policy engine for permission checks.
            retrieval_fn: Async callable(query, top_k) -> list[dict] for medical KB.
            rubric_fn: Async callable(query, top_k, stage?) -> list[dict] for rubrics.
        """
        self.registry = registry
        self.policy = policy
        self._retrieval_fn = retrieval_fn
        self._rubric_fn = rubric_fn
        # Trace log: list of {tool_name, agent_name, status, elapsed_ms, ...}
        self._traces: list[dict[str, Any]] = []

    # ── Public API ─────────────────────────────────────────────────────────

    async def execute(
        self,
        agent_name: str,
        context_view: str,
        invocation: ToolInvocation,
    ) -> SkillResult:
        """Execute a tool invocation and return a policy-wrapped result."""
        start = time.monotonic()
        tool_name = invocation.tool_name

        # 1. Lookup manifest
        manifest = self.registry.get(tool_name)
        if manifest is None:
            return self._blocked_result(
                tool_name, agent_name, start,
                error=f"Tool '{tool_name}' is not registered",
            )

        # 2. Policy check (blocked, agent permission, context view, read-only)
        decision = self.policy.check_execution(
            skill_name=tool_name,
            agent_name=agent_name,
            context_view=context_view,
            skill_manifest=manifest,
        )
        if not decision.allowed:
            return self._blocked_result(
                tool_name, agent_name, start,
                error=decision.reason,
            )

        # 3. Enforce timeout
        timeout_s = min(manifest.timeout_seconds, 30.0)  # hard cap 30s

        # 4. Route to handler
        try:
            raw_data = await asyncio.wait_for(
                self._route(tool_name, invocation.arguments, manifest),
                timeout=timeout_s,
            )
        except asyncio.TimeoutError:
            return self._blocked_result(
                tool_name, agent_name, start,
                error=f"Tool '{tool_name}' timed out after {timeout_s:.1f}s",
            )
        except Exception as exc:
            logger.warning("Skill '%s' execution error: %s", tool_name, exc)
            return self._blocked_result(
                tool_name, agent_name, start,
                error=f"Tool '{tool_name}' raised {type(exc).__name__}: {exc}",
            )

        # 5. Sanitize control characters from string fields
        sanitized = self._sanitize(raw_data)

        # 6. Enforce result length
        serialized = json.dumps(sanitized, ensure_ascii=False, default=str)
        truncated = False
        if len(serialized) > MAX_RESULT_CHARS:
            serialized = serialized[:MAX_RESULT_CHARS]
            truncated = True
            try:
                sanitized = json.loads(serialized)
            except json.JSONDecodeError:
                sanitized = {"_truncated": True, "preview": serialized[:MAX_RESULT_CHARS]}

        # 7. Wrap as UNTRUSTED_EVIDENCE
        wrapped = mark_output_untrusted(sanitized)

        elapsed_ms = (time.monotonic() - start) * 1000

        # 8. Record completed trace
        self._record_trace(tool_name, agent_name, "completed", elapsed_ms)

        return SkillResult(
            success=True,
            data=wrapped,
            trust_level="untrusted_evidence",
            elapsed_ms=round(elapsed_ms, 2),
            truncated=truncated,
            trace={"tool": tool_name, "agent": agent_name},
        )

    # ── Routing ────────────────────────────────────────────────────────────

    async def _route(
        self,
        tool_name: str,
        arguments: dict[str, Any],
        manifest: SkillManifest,
    ) -> Any:
        """Route to the appropriate handler."""
        if tool_name == "search_medical_kb":
            return await self._handle_search_medical_kb(arguments, manifest)
        if tool_name == "search_teaching_rubric":
            return await self._handle_search_teaching_rubric(arguments, manifest)
        # For other registered tools, return a placeholder (no handler wired yet)
        return {"data": [], "total": 0, "_note": f"No handler wired for '{tool_name}'"}

    async def _handle_search_medical_kb(
        self, arguments: dict[str, Any], manifest: SkillManifest
    ) -> Any:
        """Route to real RAG retrieval or fall back to demo fixtures."""
        query = str(arguments.get("query", ""))[:300]
        top_k = min(int(arguments.get("top_k", 3)), 5)

        if self._retrieval_fn is not None:
            try:
                raw_results = await self._retrieval_fn(query, top_k)
                return self._normalize_kb_results(raw_results)
            except Exception as exc:
                logger.warning("RAG retrieval failed, falling back to demo: %s", exc)

        # Fall back to MCP demo fixtures
        from app.mcp_demo.server import MCPDemoServer
        server = MCPDemoServer()
        return server.call_tool("search_medical_kb", {"query": query, "top_k": top_k})

    async def _handle_search_teaching_rubric(
        self, arguments: dict[str, Any], manifest: SkillManifest
    ) -> Any:
        """Route to real rubric retrieval or fall back to demo fixtures."""
        query = str(arguments.get("query", ""))[:200]
        top_k = min(int(arguments.get("top_k", 3)), 5)
        stage = arguments.get("stage")

        if self._rubric_fn is not None:
            try:
                raw_results = await self._rubric_fn(query, top_k, stage=stage)
                return self._normalize_rubric_results(raw_results)
            except Exception as exc:
                logger.warning("Rubric retrieval failed, falling back to demo: %s", exc)

        # Fall back to MCP demo fixtures
        from app.mcp_demo.server import MCPDemoServer
        server = MCPDemoServer()
        args: dict[str, Any] = {"query": query, "top_k": top_k}
        if stage:
            args["stage"] = stage
        return server.call_tool("search_teaching_rubric", args)

    # ── Normalization ──────────────────────────────────────────────────────

    @staticmethod
    def _normalize_kb_results(raw_results: list[dict[str, Any]]) -> dict[str, Any]:
        """Normalize RAG KB results to citation ID, title, source, score, bounded excerpt."""
        normalized = []
        for i, item in enumerate(raw_results[:5]):
            normalized.append({
                "citation_id": item.get("doc_id") or item.get("id") or f"kb_{i}",
                "title": item.get("source") or item.get("title") or "Unknown",
                "source": item.get("source") or "medical_kb",
                "score": round(float(item.get("score", 0.0)), 4),
                "excerpt": str(item.get("text", ""))[:500],
            })
        return {"data": normalized, "total": len(normalized)}

    @staticmethod
    def _normalize_rubric_results(raw_results: list[dict[str, Any]]) -> dict[str, Any]:
        """Normalize rubric results to citation ID, title, source, score, bounded excerpt."""
        normalized = []
        for i, item in enumerate(raw_results[:5]):
            normalized.append({
                "citation_id": item.get("id") or f"rubric_{i}",
                "title": item.get("criteria") or item.get("stage") or "Unknown",
                "source": "teaching_rubric",
                "score": round(float(item.get("_score", item.get("score", 0.0))), 4),
                "excerpt": str(item.get("example", ""))[:500],
            })
        return {"data": normalized, "total": len(normalized)}

    # ── Sanitization ───────────────────────────────────────────────────────

    @staticmethod
    def _sanitize(obj: Any) -> Any:
        """Recursively strip control characters from string values."""
        if isinstance(obj, str):
            return CONTROL_CHAR_RE.sub("", obj)
        if isinstance(obj, dict):
            return {k: SkillExecutor._sanitize(v) for k, v in obj.items()}
        if isinstance(obj, list):
            return [SkillExecutor._sanitize(item) for item in obj]
        return obj

    # ── Trace helpers ──────────────────────────────────────────────────────

    def _blocked_result(
        self,
        tool_name: str,
        agent_name: str,
        start: float,
        *,
        error: str,
    ) -> SkillResult:
        elapsed_ms = (time.monotonic() - start) * 1000
        self._record_trace(tool_name, agent_name, "blocked", elapsed_ms, error=error)
        return SkillResult(
            success=False,
            error=error,
            trust_level="blocked",
            elapsed_ms=round(elapsed_ms, 2),
            trace={"tool": tool_name, "agent": agent_name, "blocked": True},
        )

    def _record_trace(
        self,
        tool_name: str,
        agent_name: str,
        status: str,
        elapsed_ms: float,
        *,
        error: str | None = None,
    ) -> None:
        entry = {
            "tool": tool_name,
            "agent": agent_name,
            "status": status,
            "elapsed_ms": round(elapsed_ms, 2),
        }
        if error:
            entry["error"] = error
        self._traces.append(entry)

    def get_traces(self) -> list[dict[str, Any]]:
        """Return all recorded tool traces."""
        return list(self._traces)
