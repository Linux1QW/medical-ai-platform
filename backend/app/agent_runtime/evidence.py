"""Evidence agent: restricted to read-only knowledge retrieval skills.

Routes search_medical_kb and search_teaching_rubric to the existing hybrid
retrieval service when available, otherwise falls back to MCP demo fixtures.
All outputs are normalized to citation records and wrapped as UNTRUSTED_EVIDENCE.
"""
from __future__ import annotations

import logging
from typing import Any, Callable, Awaitable

from app.agent_runtime.policy import SkillPolicy, mark_output_untrusted
from app.agent_runtime.skills import SkillManifest, SkillRegistry

logger = logging.getLogger(__name__)

# ── Constants ──────────────────────────────────────────────────────────────────

MAX_EXCERPT_CHARS = 500  # Bounded excerpt length sent to model


# ── Types ──────────────────────────────────────────────────────────────────────

# Async callable: (query: str, top_k: int) -> list[dict]
RetrievalFn = Callable[[str, int], Awaitable[list[dict[str, Any]]]]
# Async callable: (query: str, top_k: int, stage: str | None) -> list[dict]
RubricFn = Callable[[str, int, str | None], Awaitable[list[dict[str, Any]]]]


# ── Normalization helpers ─────────────────────────────────────────────────────


def _normalize_kb_hit(item: dict[str, Any], index: int) -> dict[str, Any]:
    """Normalize a raw KB hit to a citation record."""
    return {
        "citation_id": item.get("doc_id") or item.get("id") or f"kb_{index}",
        "title": item.get("source") or item.get("title") or "Unknown",
        "source": item.get("source") or "medical_kb",
        "score": round(float(item.get("score", 0.0)), 4),
        "excerpt": str(item.get("text", ""))[:MAX_EXCERPT_CHARS],
    }


def _normalize_rubric_hit(item: dict[str, Any], index: int) -> dict[str, Any]:
    """Normalize a raw rubric hit to a citation record."""
    return {
        "citation_id": item.get("id") or f"rubric_{index}",
        "title": item.get("criteria") or item.get("stage") or "Unknown",
        "source": "teaching_rubric",
        "score": round(float(item.get("_score", item.get("score", 0.0))), 4),
        "excerpt": str(item.get("example", ""))[:MAX_EXCERPT_CHARS],
    }


# ── EvidenceAgent ──────────────────────────────────────────────────────────────


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
        retrieval_fn: RetrievalFn | None = None,
        rubric_fn: RubricFn | None = None,
    ) -> None:
        self.registry = registry
        self.policy = policy or SkillPolicy()
        self._retrieval_fn = retrieval_fn
        self._rubric_fn = rubric_fn

    async def search(
        self, skill_name: str, query: str, **kwargs: Any
    ) -> dict[str, Any]:
        """Execute a search skill. Returns untrusted evidence envelope.

        Routes to real hybrid retrieval if retrieval_fn/rubric_fn is provided;
        otherwise falls back to MCP demo fixtures.
        """
        if skill_name not in self.ALLOWED_SKILLS:
            return {"error": f"Skill '{skill_name}' not allowed for EvidenceAgent", "data": None}

        manifest: SkillManifest | None = self.registry.get(skill_name) if self.registry else None
        if manifest is None:
            # No registry: demo mode
            return self._demo_fallback(skill_name, query, **kwargs)

        # Policy check
        decision = self.policy.check_execution(
            skill_name=skill_name,
            agent_name="evidence_agent",
            context_view="coach",
            skill_manifest=manifest,
        )
        if not decision.allowed:
            return {"error": decision.reason, "data": None}

        # Route to real retrieval
        raw_results = await self._route(skill_name, query, **kwargs)
        return mark_output_untrusted(raw_results)

    # ── Routing ────────────────────────────────────────────────────────────

    async def _route(self, skill_name: str, query: str, **kwargs: Any) -> dict[str, Any]:
        """Route to real retrieval or demo fallback."""
        top_k = min(int(kwargs.get("top_k", 3)), 5)

        if skill_name == "search_medical_kb":
            if self._retrieval_fn is not None:
                try:
                    raw = await self._retrieval_fn(query, top_k)
                    normalized = [_normalize_kb_hit(item, i) for i, item in enumerate(raw[:5])]
                    return {"data": normalized, "total": len(normalized)}
                except Exception as exc:
                    logger.warning("RAG retrieval failed for search_medical_kb: %s", exc)
            return self._demo_fallback(skill_name, query, **kwargs)

        if skill_name == "search_teaching_rubric":
            stage = kwargs.get("stage")
            if self._rubric_fn is not None:
                try:
                    raw = await self._rubric_fn(query, top_k, stage)
                    normalized = [_normalize_rubric_hit(item, i) for i, item in enumerate(raw[:5])]
                    return {"data": normalized, "total": len(normalized)}
                except Exception as exc:
                    logger.warning("Rubric retrieval failed for search_teaching_rubric: %s", exc)
            return self._demo_fallback(skill_name, query, **kwargs)

        return {"data": [], "total": 0}

    # ── Demo fallback ──────────────────────────────────────────────────────

    @staticmethod
    def _demo_fallback(skill_name: str, query: str, **kwargs: Any) -> dict[str, Any]:
        """Fall back to MCP demo fixtures."""
        from app.mcp_demo.server import MCPDemoServer

        server = MCPDemoServer()
        args: dict[str, Any] = {"query": query, "top_k": min(int(kwargs.get("top_k", 3)), 5)}
        if "stage" in kwargs and kwargs["stage"]:
            args["stage"] = kwargs["stage"]
        if "topic" in kwargs and kwargs["topic"]:
            args["topic"] = kwargs["topic"]
        result = server.call_tool(skill_name, args)
        return mark_output_untrusted(result)
