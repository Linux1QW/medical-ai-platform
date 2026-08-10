"""Skill execution policy enforcement.

All tool outputs are wrapped as UNTRUSTED_EVIDENCE for prompt injection defense.
Control instruction patterns are sanitized.
Serialized output length is bounded.
Blocked and completed tool invocations are recorded as traces.
"""
from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from enum import Enum
from typing import Any


# ── Constants ──────────────────────────────────────────────────────────────────

MAX_SERIALIZED_OUTPUT_CHARS = 8000

# Patterns that look like control instructions (prompt injection attempts)
_CONTROL_INSTRUCTION_PATTERNS = [
    re.compile(r"ignore\s+(all\s+)?(previous|above|prior)\s+instructions", re.IGNORECASE),
    re.compile(r"you\s+are\s+now\s+a", re.IGNORECASE),
    re.compile(r"system\s*:\s*", re.IGNORECASE),
    re.compile(r"<\|im_start\|>", re.IGNORECASE),
    re.compile(r"\[INST\]", re.IGNORECASE),
    re.compile(r"do\s+not\s+follow\s+(any|your)\s+(rules|instructions)", re.IGNORECASE),
    re.compile(r"override\s+(safety|policy|rules)", re.IGNORECASE),
]

# Control characters to strip
_CONTROL_CHAR_RE = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]")


# ── Trust levels ───────────────────────────────────────────────────────────────


class TrustLevel(Enum):
    """Trust level for tool outputs."""

    TRUSTED_SYSTEM = "trusted_system"
    UNTRUSTED_EVIDENCE = "untrusted_evidence"
    BLOCKED = "blocked"


# ── Policy decision ────────────────────────────────────────────────────────────


@dataclass
class PolicyDecision:
    """Result of a policy check."""

    allowed: bool
    trust_level: TrustLevel
    reason: str = ""
    sanitized_output: Any = None


# ── Trace entry ────────────────────────────────────────────────────────────────


@dataclass
class ToolTrace:
    """A recorded tool invocation trace."""

    tool_name: str
    agent_name: str
    status: str  # "blocked" or "completed"
    reason: str = ""


# ── SkillPolicy ────────────────────────────────────────────────────────────────


class SkillPolicy:
    """Enforces skill execution policies.

    - All tool outputs wrapped as UNTRUSTED_EVIDENCE
    - Control instructions sanitized
    - Serialized output length limited
    - Blocked and completed tool traces recorded
    """

    def __init__(self) -> None:
        self._blocked_tools: set[str] = set()
        self._agent_permissions: dict[str, set[str]] = {}  # agent → allowed skill names
        self._traces: list[ToolTrace] = []

    def check_execution(
        self,
        *,
        skill_name: str,
        agent_name: str,
        context_view: str,
        skill_manifest: Any,
    ) -> PolicyDecision:
        """Check if an agent can execute a skill in a given context view.

        Checks:
        1. Tool not blocked
        2. Agent is in allowed_agents (or allowed_agents is empty = all allowed)
        3. Context view is in allowed_context_views (or empty = all allowed)
        4. Skill is read_only (write skills require explicit approval)
        """
        # Check blocked
        if skill_name in self._blocked_tools:
            self._record_trace(skill_name, agent_name, "blocked", "Tool is blocked")
            return PolicyDecision(
                allowed=False,
                trust_level=TrustLevel.BLOCKED,
                reason=f"Tool '{skill_name}' is blocked",
            )

        # Check agent permissions
        if skill_manifest.allowed_agents:
            if agent_name not in skill_manifest.allowed_agents:
                self._record_trace(
                    skill_name, agent_name, "blocked",
                    f"Agent '{agent_name}' not in allowed_agents",
                )
                return PolicyDecision(
                    allowed=False,
                    trust_level=TrustLevel.BLOCKED,
                    reason=f"Agent '{agent_name}' not allowed for skill '{skill_name}'",
                )

        # Check context view
        if skill_manifest.allowed_context_views:
            if context_view not in skill_manifest.allowed_context_views:
                self._record_trace(
                    skill_name, agent_name, "blocked",
                    f"Context view '{context_view}' not allowed",
                )
                return PolicyDecision(
                    allowed=False,
                    trust_level=TrustLevel.BLOCKED,
                    reason=f"Context view '{context_view}' not allowed for skill '{skill_name}'",
                )

        # Write skills require explicit approval
        if not skill_manifest.read_only:
            self._record_trace(
                skill_name, agent_name, "blocked",
                "Write operation requires approval",
            )
            return PolicyDecision(
                allowed=False,
                trust_level=TrustLevel.BLOCKED,
                reason=f"Skill '{skill_name}' is not read-only; write operations require approval",
            )

        # All tool outputs from skills are marked as untrusted evidence
        # (prompt injection defense)
        self._record_trace(skill_name, agent_name, "completed", "Allowed; output marked untrusted")
        return PolicyDecision(
            allowed=True,
            trust_level=TrustLevel.UNTRUSTED_EVIDENCE,
            reason="Skill execution allowed; output marked as untrusted evidence",
        )

    def block_tool(self, tool_name: str) -> None:
        """Block a tool by name."""
        self._blocked_tools.add(tool_name)

    def unblock_tool(self, tool_name: str) -> None:
        """Unblock a tool by name."""
        self._blocked_tools.discard(tool_name)

    def set_agent_permissions(self, agent_name: str, allowed_skills: set[str]) -> None:
        """Set allowed skills for an agent."""
        self._agent_permissions[agent_name] = allowed_skills

    def get_traces(self) -> list[ToolTrace]:
        """Return all recorded tool traces."""
        return list(self._traces)

    def clear_traces(self) -> None:
        """Clear recorded traces."""
        self._traces.clear()

    def _record_trace(
        self, tool_name: str, agent_name: str, status: str, reason: str
    ) -> None:
        self._traces.append(
            ToolTrace(
                tool_name=tool_name,
                agent_name=agent_name,
                status=status,
                reason=reason,
            )
        )


# ── Output wrapping & sanitization ────────────────────────────────────────────


def mark_output_untrusted(output: Any) -> dict:
    """Mark a tool output as untrusted evidence for prompt injection defense.

    Wraps the output in a metadata envelope that LLM prompts can reference
    to distinguish trusted system content from untrusted tool results.
    """
    return {
        "__trust_level__": "untrusted_evidence",
        "__source__": "tool_output",
        "data": output,
    }


def sanitize_control_instructions(text: str) -> str:
    """Remove or neutralize control instruction patterns from text.

    This is a defense-in-depth measure against prompt injection via tool outputs.
    """
    for pattern in _CONTROL_INSTRUCTION_PATTERNS:
        text = pattern.sub("[REDACTED]", text)
    return text


def sanitize_output(output: Any) -> Any:
    """Recursively sanitize tool output: strip control chars and redact injections."""
    if isinstance(output, str):
        output = _CONTROL_CHAR_RE.sub("", output)
        output = sanitize_control_instructions(output)
        return output
    if isinstance(output, dict):
        return {k: sanitize_output(v) for k, v in output.items()}
    if isinstance(output, list):
        return [sanitize_output(item) for item in output]
    return output


def limit_serialized_output(output: Any, max_chars: int = MAX_SERIALIZED_OUTPUT_CHARS) -> tuple[Any, bool]:
    """Serialize and truncate output to max_chars.

    Returns (possibly_truncated_output, was_truncated).
    """
    serialized = json.dumps(output, ensure_ascii=False, default=str)
    if len(serialized) <= max_chars:
        return output, False

    # Truncate and try to parse back; if invalid, return truncated string
    truncated_str = serialized[:max_chars]
    try:
        return json.loads(truncated_str), True
    except json.JSONDecodeError:
        return {"_truncated": True, "preview": truncated_str}, True
