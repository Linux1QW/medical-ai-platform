"""Tests for skill manifest, registry, and policy enforcement."""
from __future__ import annotations

from pathlib import Path

from app.agent_runtime.policy import (
    SkillPolicy,
    TrustLevel,
    mark_output_untrusted,
)
from app.agent_runtime.skills import SkillManifest, SkillRegistry

SKILLS_DIR = Path(__file__).resolve().parent.parent.parent / "skills"


# ---------------------------------------------------------------------------
# SkillManifest
# ---------------------------------------------------------------------------


class TestSkillManifest:
    def test_skill_manifest_from_yaml(self) -> None:
        yaml_path = str(SKILLS_DIR / "search_teaching_rubric.yaml")
        manifest = SkillManifest.from_yaml(yaml_path)

        assert manifest.name == "search_teaching_rubric"
        assert manifest.version == "1.0.0"
        assert manifest.read_only is True
        assert manifest.timeout_seconds == 5.0
        assert manifest.budget_tokens == 2000
        assert "evidence_agent" in manifest.allowed_agents
        assert "coach_agent" in manifest.allowed_agents
        assert "coach" in manifest.allowed_context_views
        assert "query" in manifest.parameters
        assert manifest.parameters["query"]["type"] == "string"


# ---------------------------------------------------------------------------
# SkillRegistry
# ---------------------------------------------------------------------------


class TestSkillRegistry:
    def test_skill_registry_register_and_get(self) -> None:
        registry = SkillRegistry()
        manifest = SkillManifest(name="test_skill", version="1.0.0")
        registry.register(manifest)

        result = registry.get("test_skill")
        assert result is not None
        assert result.name == "test_skill"
        assert result.version == "1.0.0"

    def test_skill_registry_get_latest_version(self) -> None:
        registry = SkillRegistry()
        v1 = SkillManifest(name="skill_a", version="1.0.0", description="v1")
        v2 = SkillManifest(name="skill_a", version="2.0.0", description="v2")
        registry.register(v1)
        registry.register(v2)

        result = registry.get("skill_a")
        assert result is not None
        assert result.version == "2.0.0"
        assert result.description == "v2"

    def test_skill_registry_get_pinned_version(self) -> None:
        registry = SkillRegistry()
        v1 = SkillManifest(name="skill_a", version="1.0.0", description="v1")
        v2 = SkillManifest(name="skill_a", version="2.0.0", description="v2")
        registry.register(v1)
        registry.register(v2)

        result = registry.get("skill_a", version="1.0.0")
        assert result is not None
        assert result.description == "v1"

    def test_skill_registry_load_directory(self) -> None:
        registry = SkillRegistry()
        count = registry.load_directory(str(SKILLS_DIR))

        assert count >= 3
        assert registry.get("search_teaching_rubric") is not None
        assert registry.get("search_medical_kb") is not None
        assert registry.get("check_drug_interaction") is not None

    def test_skill_registry_list_skills(self) -> None:
        registry = SkillRegistry()
        registry.register(SkillManifest(name="s1", version="1.0.0"))
        registry.register(SkillManifest(name="s2", version="1.0.0"))

        skills = registry.list_skills()
        assert len(skills) == 2


# ---------------------------------------------------------------------------
# SkillPolicy
# ---------------------------------------------------------------------------


def _make_manifest(**overrides: object) -> SkillManifest:
    defaults = {
        "name": "test_skill",
        "version": "1.0.0",
        "allowed_agents": ["evidence_agent"],
        "allowed_context_views": ["coach"],
        "read_only": True,
    }
    defaults.update(overrides)
    return SkillManifest(**defaults)  # type: ignore[arg-type]


class TestSkillPolicy:
    def test_policy_allows_readonly_skill(self) -> None:
        policy = SkillPolicy()
        manifest = _make_manifest()
        decision = policy.check_execution(
            skill_name="test_skill",
            agent_name="evidence_agent",
            context_view="coach",
            skill_manifest=manifest,
        )
        assert decision.allowed is True
        assert decision.trust_level == TrustLevel.UNTRUSTED_EVIDENCE

    def test_policy_blocks_wrong_agent(self) -> None:
        policy = SkillPolicy()
        manifest = _make_manifest()
        decision = policy.check_execution(
            skill_name="test_skill",
            agent_name="rogue_agent",
            context_view="coach",
            skill_manifest=manifest,
        )
        assert decision.allowed is False
        assert decision.trust_level == TrustLevel.BLOCKED

    def test_policy_blocks_wrong_context_view(self) -> None:
        policy = SkillPolicy()
        manifest = _make_manifest()
        decision = policy.check_execution(
            skill_name="test_skill",
            agent_name="evidence_agent",
            context_view="evaluation",
            skill_manifest=manifest,
        )
        assert decision.allowed is False
        assert decision.trust_level == TrustLevel.BLOCKED

    def test_policy_blocks_write_skill(self) -> None:
        policy = SkillPolicy()
        manifest = _make_manifest(read_only=False)
        decision = policy.check_execution(
            skill_name="test_skill",
            agent_name="evidence_agent",
            context_view="coach",
            skill_manifest=manifest,
        )
        assert decision.allowed is False
        assert decision.trust_level == TrustLevel.BLOCKED

    def test_policy_blocks_blocked_tool(self) -> None:
        policy = SkillPolicy()
        policy.block_tool("test_skill")
        manifest = _make_manifest()
        decision = policy.check_execution(
            skill_name="test_skill",
            agent_name="evidence_agent",
            context_view="coach",
            skill_manifest=manifest,
        )
        assert decision.allowed is False
        assert decision.trust_level == TrustLevel.BLOCKED

    def test_policy_unblock_tool(self) -> None:
        policy = SkillPolicy()
        policy.block_tool("test_skill")
        policy.unblock_tool("test_skill")
        manifest = _make_manifest()
        decision = policy.check_execution(
            skill_name="test_skill",
            agent_name="evidence_agent",
            context_view="coach",
            skill_manifest=manifest,
        )
        assert decision.allowed is True

    def test_policy_empty_allowed_agents_means_all(self) -> None:
        policy = SkillPolicy()
        manifest = _make_manifest(allowed_agents=[])
        decision = policy.check_execution(
            skill_name="test_skill",
            agent_name="any_agent",
            context_view="coach",
            skill_manifest=manifest,
        )
        assert decision.allowed is True


# ---------------------------------------------------------------------------
# mark_output_untrusted
# ---------------------------------------------------------------------------


class TestMarkOutputUntrusted:
    def test_mark_output_untrusted(self) -> None:
        result = mark_output_untrusted({"key": "value"})
        assert result["__trust_level__"] == "untrusted_evidence"
        assert result["__source__"] == "tool_output"
        assert result["data"] == {"key": "value"}

    def test_mark_output_untrusted_string(self) -> None:
        result = mark_output_untrusted("hello")
        assert result["data"] == "hello"
        assert result["__trust_level__"] == "untrusted_evidence"
