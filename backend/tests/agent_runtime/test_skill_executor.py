"""Tests for SkillExecutor: forbidden tools, YAML validation, RAG routing."""
from __future__ import annotations

import pytest

from app.agent_runtime.policy import SkillPolicy
from app.agent_runtime.skill_executor import SkillExecutor, ToolInvocation
from app.agent_runtime.skills import SkillManifest, SkillRegistry, SkillValidationError, validate_manifest


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture()
def registry() -> SkillRegistry:
    reg = SkillRegistry()
    reg.register(SkillManifest(
        name="search_medical_kb",
        version="1.0.0",
        allowed_agents=["evidence_agent", "coach_agent"],
        allowed_context_views=["coach"],
        read_only=True,
        timeout_seconds=8.0,
        budget_tokens=3000,
    ))
    reg.register(SkillManifest(
        name="search_teaching_rubric",
        version="1.0.0",
        allowed_agents=["evidence_agent"],
        allowed_context_views=["coach"],
        read_only=True,
        timeout_seconds=5.0,
        budget_tokens=2000,
    ))
    reg.register(SkillManifest(
        name="dangerous_tool",
        version="1.0.0",
        allowed_agents=[],
        allowed_context_views=[],
        read_only=False,
        timeout_seconds=3.0,
    ))
    return reg


@pytest.fixture()
def policy() -> SkillPolicy:
    return SkillPolicy()


@pytest.fixture()
def executor(registry: SkillRegistry, policy: SkillPolicy) -> SkillExecutor:
    return SkillExecutor(registry=registry, policy=policy)


# ---------------------------------------------------------------------------
# Forbidden tool: not registered
# ---------------------------------------------------------------------------


class TestForbiddenTool:
    @pytest.mark.asyncio
    async def test_unregistered_tool_is_blocked(self, executor: SkillExecutor) -> None:
        invocation = ToolInvocation(
            tool_name="nonexistent_tool",
            arguments={"query": "test"},
            agent_name="evidence_agent",
            context_view="coach",
        )
        result = await executor.execute("evidence_agent", "coach", invocation)
        assert result.success is False
        assert result.trust_level == "blocked"
        assert "not registered" in result.error

    @pytest.mark.asyncio
    async def test_write_skill_is_blocked(self, executor: SkillExecutor) -> None:
        invocation = ToolInvocation(
            tool_name="dangerous_tool",
            arguments={},
            agent_name="evidence_agent",
            context_view="coach",
        )
        result = await executor.execute("evidence_agent", "coach", invocation)
        assert result.success is False
        assert result.trust_level == "blocked"
        assert "not read-only" in result.error

    @pytest.mark.asyncio
    async def test_wrong_agent_is_blocked(self, executor: SkillExecutor) -> None:
        invocation = ToolInvocation(
            tool_name="search_teaching_rubric",
            arguments={"query": "test"},
            agent_name="rogue_agent",
            context_view="coach",
        )
        result = await executor.execute("rogue_agent", "coach", invocation)
        assert result.success is False
        assert result.trust_level == "blocked"
        assert "not allowed" in result.error


# ---------------------------------------------------------------------------
# YAML validation
# ---------------------------------------------------------------------------


class TestYAMLValidation:
    def test_valid_manifest_passes(self) -> None:
        manifest = SkillManifest(
            name="valid_skill",
            version="1.0.0",
            allowed_agents=["agent_a"],
            allowed_context_views=["coach"],
            read_only=True,
            timeout_seconds=5.0,
            budget_tokens=1000,
        )
        errors = validate_manifest(manifest)
        assert errors == []

    def test_invalid_semver(self) -> None:
        manifest = SkillManifest(name="skill", version="1.0")
        errors = validate_manifest(manifest)
        assert any("semantic version" in e for e in errors)

    def test_invalid_name(self) -> None:
        manifest = SkillManifest(name="123bad", version="1.0.0")
        errors = validate_manifest(manifest)
        assert any("Invalid skill name" in e for e in errors)

    def test_invalid_context_view(self) -> None:
        manifest = SkillManifest(
            name="skill", version="1.0.0", allowed_context_views=["unknown_view"]
        )
        errors = validate_manifest(manifest)
        assert any("Unknown context_view" in e for e in errors)

    def test_timeout_out_of_range(self) -> None:
        manifest = SkillManifest(name="skill", version="1.0.0", timeout_seconds=100.0)
        errors = validate_manifest(manifest)
        assert any("timeout_seconds" in e for e in errors)

    def test_budget_tokens_out_of_range(self) -> None:
        manifest = SkillManifest(name="skill", version="1.0.0", budget_tokens=999999)
        errors = validate_manifest(manifest)
        assert any("budget_tokens" in e for e in errors)

    def test_strict_registry_raises_on_duplicate(self) -> None:
        registry = SkillRegistry(strict=True)
        registry.register(SkillManifest(name="skill", version="1.0.0"))
        with pytest.raises(SkillValidationError, match="Duplicate"):
            registry.register(SkillManifest(name="skill", version="1.0.0"))

    def test_strict_registry_raises_on_invalid(self) -> None:
        registry = SkillRegistry(strict=True)
        with pytest.raises(SkillValidationError, match="validation failed"):
            registry.register(SkillManifest(name="123bad", version="1.0"))


# ---------------------------------------------------------------------------
# RAG routing
# ---------------------------------------------------------------------------


class TestRAGRouting:
    @pytest.mark.asyncio
    async def test_search_medical_kb_routes_to_demo(self, executor: SkillExecutor) -> None:
        """Without a retrieval_fn, should fall back to MCP demo fixtures."""
        invocation = ToolInvocation(
            tool_name="search_medical_kb",
            arguments={"query": "hypertension", "top_k": 3},
            agent_name="evidence_agent",
            context_view="coach",
        )
        result = await executor.execute("evidence_agent", "coach", invocation)
        assert result.success is True
        assert result.trust_level == "untrusted_evidence"
        assert result.data is not None
        assert result.data["__trust_level__"] == "untrusted_evidence"

    @pytest.mark.asyncio
    async def test_search_medical_kb_with_custom_retrieval(self, registry, policy) -> None:
        """With a custom retrieval_fn, should route to it."""
        async def mock_retrieval(query: str, top_k: int) -> list[dict]:
            return [
                {"doc_id": "doc_1", "source": "Test Guide", "text": "Test content", "score": 0.95}
            ]

        executor = SkillExecutor(
            registry=registry, policy=policy, retrieval_fn=mock_retrieval
        )
        invocation = ToolInvocation(
            tool_name="search_medical_kb",
            arguments={"query": "test", "top_k": 3},
            agent_name="evidence_agent",
            context_view="coach",
        )
        result = await executor.execute("evidence_agent", "coach", invocation)
        assert result.success is True
        # result.data is the UNTRUSTED_EVIDENCE envelope
        # result.data["data"] is the actual result dict {"data": [...], "total": N}
        inner = result.data["data"]
        data = inner["data"]
        assert len(data) == 1
        assert data[0]["citation_id"] == "doc_1"
        assert data[0]["score"] == 0.95

    @pytest.mark.asyncio
    async def test_search_teaching_rubric_routes_to_demo(self, executor: SkillExecutor) -> None:
        invocation = ToolInvocation(
            tool_name="search_teaching_rubric",
            arguments={"query": "onset", "top_k": 2},
            agent_name="evidence_agent",
            context_view="coach",
        )
        result = await executor.execute("evidence_agent", "coach", invocation)
        assert result.success is True
        assert result.data["__trust_level__"] == "untrusted_evidence"


# ---------------------------------------------------------------------------
# Traces
# ---------------------------------------------------------------------------


class TestTraces:
    @pytest.mark.asyncio
    async def test_blocked_and_completed_traces(self, executor: SkillExecutor) -> None:
        # Blocked: unregistered tool
        inv1 = ToolInvocation(
            tool_name="unknown", arguments={}, agent_name="a", context_view="coach"
        )
        await executor.execute("a", "coach", inv1)

        # Completed: valid tool
        inv2 = ToolInvocation(
            tool_name="search_medical_kb",
            arguments={"query": "test"},
            agent_name="evidence_agent",
            context_view="coach",
        )
        await executor.execute("evidence_agent", "coach", inv2)

        traces = executor.get_traces()
        assert len(traces) >= 2
        statuses = {t["status"] for t in traces}
        assert "blocked" in statuses
        assert "completed" in statuses
