"""Production Coach runtime factory.

Builds a CoachRuntime with real checkpointer, model gateway, and evidence agent.
Raises CoachUnavailableError when required dependencies are missing in production.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any, Awaitable, Callable

from langgraph.graph.state import CompiledStateGraph

from app.agent_runtime.evidence import EvidenceAgent
from app.agent_runtime.graph import CoachDependencies, build_coach_graph
from app.agent_runtime.model_gateway import CoachModelGateway, QwenModelGateway
from app.agent_runtime.policy import SkillPolicy
from app.agent_runtime.skills import SkillRegistry
from app.core.config import settings

logger = logging.getLogger(__name__)


class CoachUnavailableError(Exception):
    """Raised when Coach cannot start due to missing dependencies."""

    def __init__(self, error_code: str) -> None:
        self.error_code = error_code
        super().__init__(error_code)


# Type alias for the evidence function used by the graph
EvidenceFn = Callable[[str, str], Awaitable[list[dict[str, Any]]]]


@dataclass(frozen=True)
class CoachRuntime:
    """Immutable bundle of production Coach dependencies."""

    graph: CompiledStateGraph
    gateway: CoachModelGateway
    evidence_agent: EvidenceAgent


def build_evidence_fn(evidence_agent: EvidenceAgent) -> EvidenceFn:
    """Create an async evidence retrieval function from an EvidenceAgent."""

    async def _evidence_fn(intent: str, message: str) -> list[dict[str, Any]]:
        result = await evidence_agent.search(
            "search_medical_kb",
            message,
            top_k=3,
        )
        data = result.get("data")
        if isinstance(data, list):
            return data
        return []

    return _evidence_fn


def _build_production_evidence_agent() -> EvidenceAgent:
    """Build an EvidenceAgent with real skill registry and policy.

    Loads skills from the skills/ directory. If no skills are found,
    the agent will return empty results (no demo fallback in production).
    """
    import os

    registry = SkillRegistry(strict=False)
    skills_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), "skills")
    if os.path.isdir(skills_dir):
        loaded = registry.load_directory(skills_dir)
        logger.info("Loaded %d skill(s) from %s", loaded, skills_dir)

    policy = SkillPolicy()

    # Build retrieval function using the real RAG hybrid search if available
    retrieval_fn = _try_build_retrieval_fn()
    rubric_fn = _try_build_rubric_fn()

    return EvidenceAgent(
        registry=registry,
        policy=policy,
        retrieval_fn=retrieval_fn,
        rubric_fn=rubric_fn,
        use_demo_fallback=False,  # Production: no demo data
    )


def _try_build_retrieval_fn() -> Any:
    """Try to build a real retrieval function from the RAG system.

    Returns None if RAG dependencies are not available.
    """
    try:
        from app.rag.hybrid_search import hybrid_search

        async def retrieval_fn(query: str, top_k: int) -> list[dict[str, Any]]:
            results = await hybrid_search(query=query, top_k=top_k)
            return [
                {
                    "doc_id": r.get("doc_id", r.get("id", "")),
                    "source": r.get("source", "medical_kb"),
                    "text": r.get("text", r.get("content", "")),
                    "score": r.get("score", 0.0),
                }
                for r in results
            ]

        return retrieval_fn
    except Exception as exc:
        logger.warning("RAG retrieval not available: %s", exc)
        return None


def _try_build_rubric_fn() -> Any:
    """Try to build a real rubric retrieval function.

    Returns None if rubric dependencies are not available.
    """
    try:
        from app.rag.hybrid_search import hybrid_search

        async def rubric_fn(query: str, top_k: int, stage: str | None) -> list[dict[str, Any]]:
            results = await hybrid_search(query=query, top_k=top_k, collection="teaching_rubric")
            return [
                {
                    "id": r.get("doc_id", r.get("id", "")),
                    "criteria": r.get("criteria", ""),
                    "stage": r.get("stage", stage or ""),
                    "example": r.get("example", r.get("text", "")),
                    "score": r.get("score", 0.0),
                }
                for r in results
            ]

        return rubric_fn
    except Exception as exc:
        logger.warning("Rubric retrieval not available: %s", exc)
        return None


class CoachRuntimeFactory:
    """Creates production CoachRuntime instances.

    Validates that all required dependencies are available before
    constructing the runtime.
    """

    async def create(self) -> CoachRuntime:
        """Create a production CoachRuntime.

        Raises:
            CoachUnavailableError: If any required dependency is missing.
        """
        # 1. Check checkpointer
        from app.orchestration.checkpointer import get_checkpointer

        checkpointer = get_checkpointer()
        if checkpointer is None:
            raise CoachUnavailableError("COACH_CHECKPOINTER_UNAVAILABLE")

        # 2. Build model gateway
        model_name = settings.COACH_MODEL or None
        gateway = QwenModelGateway(
            model=model_name,
            temperature=settings.COACH_TEMPERATURE,
        )

        # 3. Build evidence agent
        evidence_agent = _build_production_evidence_agent()

        # 4. Build graph with real dependencies
        dependencies = CoachDependencies(
            gateway=gateway,
            evidence_fn=build_evidence_fn(evidence_agent),
        )
        graph = build_coach_graph(
            checkpointer=checkpointer,
            dependencies=dependencies,
        )

        return CoachRuntime(
            graph=graph,
            gateway=gateway,
            evidence_agent=evidence_agent,
        )


def validate_production_dependencies() -> list[str]:
    """Validate that all Coach production dependencies are available.

    Returns a list of missing dependency descriptions.
    Empty list means all dependencies are satisfied.
    """
    missing: list[str] = []

    # 1. Model credentials
    if not settings.llm_api_key:
        missing.append("LLM_API_KEY (model credentials)")

    # 2. Redis Checkpointer
    from app.orchestration.checkpointer import get_checkpointer

    if get_checkpointer() is None:
        missing.append("Redis Checkpointer (not initialized)")

    # 3. Prompt Bundle (system prompt + safety policy are built-in,
    #    but we check that they exist)
    from app.agent_runtime.context import COACH_SAFETY_POLICY, COACH_SYSTEM_PROMPT

    if not COACH_SYSTEM_PROMPT.strip():
        missing.append("Coach system prompt (empty)")
    if not COACH_SAFETY_POLICY.strip():
        missing.append("Coach safety policy (empty)")

    # 4. Skill Manifest
    import os

    skills_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), "skills")
    if not os.path.isdir(skills_dir) or not os.listdir(skills_dir):
        missing.append("Skill manifest directory (empty or missing)")

    # 5. RAG dependencies (soft check — log warning if not available)
    try:
        from app.rag.hybrid_search import hybrid_search  # noqa: F401
    except Exception:
        missing.append("RAG hybrid search (not available)")

    return missing
