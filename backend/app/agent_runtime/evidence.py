"""Evidence agent: restricted to read-only knowledge retrieval skills."""
from __future__ import annotations

from typing import Any

from app.agent_runtime.policy import SkillPolicy, mark_output_untrusted
from app.agent_runtime.skills import SkillManifest, SkillRegistry


class EvidenceAgent:
    """Evidence retrieval agent.

    Can ONLY call:
    - search_teaching_rubric
    - search_medical_kb

    All outputs are marked as untrusted evidence.
    """

    ALLOWED_SKILLS = {"search_teaching_rubric", "search_medical_kb"}

    def __init__(
        self,
        *,
        registry: SkillRegistry | None = None,
        policy: SkillPolicy | None = None,
    ) -> None:
        self.registry = registry
        self.policy = policy or SkillPolicy()

    def search(self, skill_name: str, query: str, **kwargs: Any) -> dict[str, Any]:
        """Execute a search skill. Returns untrusted evidence envelope."""
        if skill_name not in self.ALLOWED_SKILLS:
            return {"error": f"Skill '{skill_name}' not allowed for EvidenceAgent", "data": None}

        manifest: SkillManifest | None = self.registry.get(skill_name) if self.registry else None
        if manifest is None:
            # Demo mode: return empty results
            return {"data": [], "total": 0}

        # Policy check
        decision = self.policy.check_execution(
            skill_name=skill_name,
            agent_name="evidence_agent",
            context_view="coach",
            skill_manifest=manifest,
        )
        if not decision.allowed:
            return {"error": decision.reason, "data": None}

        # Execute (demo mode returns empty)
        result: dict[str, Any] = {"data": [], "total": 0}
        return mark_output_untrusted(result)
