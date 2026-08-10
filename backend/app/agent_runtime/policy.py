"""Skill execution policy enforcement."""
from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Any


class TrustLevel(Enum):
    """Trust level for tool outputs."""

    TRUSTED_SYSTEM = "trusted_system"
    UNTRUSTED_EVIDENCE = "untrusted_evidence"
    BLOCKED = "blocked"


@dataclass
class PolicyDecision:
    """Result of a policy check."""

    allowed: bool
    trust_level: TrustLevel
    reason: str = ""
    sanitized_output: Any = None


class SkillPolicy:
    """Enforces skill execution policies."""

    def __init__(self) -> None:
        self._blocked_tools: set[str] = set()
        self._agent_permissions: dict[str, set[str]] = {}  # agent → allowed skill names

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
            return PolicyDecision(
                allowed=False,
                trust_level=TrustLevel.BLOCKED,
                reason=f"Tool '{skill_name}' is blocked",
            )

        # Check agent permissions
        if skill_manifest.allowed_agents:
            if agent_name not in skill_manifest.allowed_agents:
                return PolicyDecision(
                    allowed=False,
                    trust_level=TrustLevel.BLOCKED,
                    reason=f"Agent '{agent_name}' not allowed for skill '{skill_name}'",
                )

        # Check context view
        if skill_manifest.allowed_context_views:
            if context_view not in skill_manifest.allowed_context_views:
                return PolicyDecision(
                    allowed=False,
                    trust_level=TrustLevel.BLOCKED,
                    reason=f"Context view '{context_view}' not allowed for skill '{skill_name}'",
                )

        # Write skills require explicit approval
        if not skill_manifest.read_only:
            return PolicyDecision(
                allowed=False,
                trust_level=TrustLevel.BLOCKED,
                reason=f"Skill '{skill_name}' is not read-only; write operations require approval",
            )

        # All tool outputs from skills are marked as untrusted evidence
        # (prompt injection defense)
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
