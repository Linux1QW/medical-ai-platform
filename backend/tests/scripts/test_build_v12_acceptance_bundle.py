"""Fail-closed tests for the V1.2 acceptance bundle builder.

These tests verify that the bundle correctly fails (passed=false) when any
required evidence section is missing or invalid. This ensures no incomplete
release can be accidentally accepted.
"""
from __future__ import annotations

import sys
from pathlib import Path
from typing import Any
from unittest.mock import patch

# Add scripts directory to path
SCRIPTS_DIR = Path(__file__).resolve().parents[2] / "scripts" / "ci"
sys.path.insert(0, str(SCRIPTS_DIR))

from build_v12_acceptance_bundle import (  # noqa: E402
    REQUIRED_SECTIONS,
    _compute_signature,
    _evaluate_passed,
    build_bundle,
    generate_acceptance_summary_md,
    generate_final_acceptance_md,
)

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

def _make_minimal_bundle(**overrides: Any) -> dict[str, Any]:
    """Create a minimal valid bundle for testing."""
    bundle: dict[str, Any] = {
        "candidate_sha": "a" * 40,
        "passed": False,
        "ci": {
            "mypy": {"errors": 0, "passed": True},
            "ruff": {"errors": 0, "passed": True},
            "branch": "test",
            "timestamp": "2026-01-01T00:00:00Z",
        },
        "rc": {"workflow_run_url": "", "status": "simulated"},
        "metrics": {
            "backend_tests_collected": "2103+",
            "frontend_tests_passed": 106,
            "mypy_errors": 0,
            "ruff_errors": 0,
            "ci_status": "green",
        },
        "migration": {
            "alembic_head": "5e6f7a8b9c0d",
            "upgrade_tested": True,
            "downgrade_tested": True,
            "migration_files": ["4d5e6f7a8b9c.py", "5e6f7a8b9c0d.py"],
        },
        "e2e": {"scenarios": 25, "passed": True},
        "load": {"p95_latency_ms": "TBD", "error_rate": "TBD", "note": "Requires deployed environment"},
        "rollback": {"drill_documented": True, "drill_file": "rollback-drill.md", "rto_target_minutes": 15, "steps": 8},
        "approvals": {"code_owner_review": True, "security_review": True, "qa_sign_off": True},
        "artifacts": {"final_acceptance_json": "final-acceptance.json"},
        "provenance": {
            "generated_at": "2026-01-01T00:00:00Z",
            "generator_script": "build_v12_acceptance_bundle.py",
            "python_version": "3.10.0",
            "platform": "test",
            "git_sha": "a" * 40,
            "git_branch": "test",
        },
        "signature": {"algorithm": "sha256", "computed": False, "hash": ""},
        "voice": {"status": "NOT_ACCEPTED", "default_enabled": False, "reason": "No real LiveKit credentials configured; Voice feature is opt-in and disabled by default (VOICE_ENABLED=false). Not included in acceptance criteria."},
        "coach": {"status": "CODE_COMPLETE", "default_enabled": False, "safety_controls": []},
    }
    bundle.update(overrides)
    # Compute signature
    bundle["signature"] = _compute_signature(bundle)
    # Re-evaluate passed status
    bundle["passed"] = _evaluate_passed(
        ci=bundle["ci"],
        metrics=bundle["metrics"],
        migration=bundle["migration"],
        e2e=bundle["e2e"],
        load=bundle["load"],
        signature=bundle["signature"],
        rollback=bundle["rollback"],
        approvals=bundle["approvals"],
        artifacts=bundle["artifacts"],
        provenance=bundle["provenance"],
    )
    return bundle


# ---------------------------------------------------------------------------
# Tests: Fail-closed behavior
# ---------------------------------------------------------------------------

class TestFailClosed:
    """Verify that missing any required section causes passed=false."""

    def test_all_sections_present_passes(self) -> None:
        """A complete bundle with all sections should pass."""
        bundle = _make_minimal_bundle()
        assert _evaluate_passed(
            ci=bundle["ci"],
            metrics=bundle["metrics"],
            migration=bundle["migration"],
            e2e=bundle["e2e"],
            load=bundle["load"],
            signature=bundle["signature"],
            rollback=bundle["rollback"],
            approvals=bundle["approvals"],
            artifacts=bundle["artifacts"],
            provenance=bundle["provenance"],
        ) is True

    def test_missing_metrics_fails(self) -> None:
        """Missing metrics section must cause failure."""
        bundle = _make_minimal_bundle()
        assert _evaluate_passed(
            ci=bundle["ci"],
            metrics=None,  # type: ignore[arg-type]
            migration=bundle["migration"],
            e2e=bundle["e2e"],
            load=bundle["load"],
            signature=bundle["signature"],
            rollback=bundle["rollback"],
            approvals=bundle["approvals"],
            artifacts=bundle["artifacts"],
            provenance=bundle["provenance"],
        ) is False

    def test_missing_artifacts_fails(self) -> None:
        """Missing artifacts section must cause failure."""
        bundle = _make_minimal_bundle()
        assert _evaluate_passed(
            ci=bundle["ci"],
            metrics=bundle["metrics"],
            migration=bundle["migration"],
            e2e=bundle["e2e"],
            load=bundle["load"],
            signature=bundle["signature"],
            rollback=bundle["rollback"],
            approvals=bundle["approvals"],
            artifacts=None,  # type: ignore[arg-type]
            provenance=bundle["provenance"],
        ) is False

    def test_missing_provenance_fails(self) -> None:
        """Missing provenance section must cause failure."""
        bundle = _make_minimal_bundle()
        assert _evaluate_passed(
            ci=bundle["ci"],
            metrics=bundle["metrics"],
            migration=bundle["migration"],
            e2e=bundle["e2e"],
            load=bundle["load"],
            signature=bundle["signature"],
            rollback=bundle["rollback"],
            approvals=bundle["approvals"],
            artifacts=bundle["artifacts"],
            provenance=None,  # type: ignore[arg-type]
        ) is False

    def test_missing_migration_fails(self) -> None:
        """Missing migration section must cause failure."""
        bundle = _make_minimal_bundle()
        assert _evaluate_passed(
            ci=bundle["ci"],
            metrics=bundle["metrics"],
            migration=None,  # type: ignore[arg-type]
            e2e=bundle["e2e"],
            load=bundle["load"],
            signature=bundle["signature"],
            rollback=bundle["rollback"],
            approvals=bundle["approvals"],
            artifacts=bundle["artifacts"],
            provenance=bundle["provenance"],
        ) is False

    def test_missing_e2e_fails(self) -> None:
        """Missing e2e section must cause failure."""
        bundle = _make_minimal_bundle()
        assert _evaluate_passed(
            ci=bundle["ci"],
            metrics=bundle["metrics"],
            migration=bundle["migration"],
            e2e=None,  # type: ignore[arg-type]
            load=bundle["load"],
            signature=bundle["signature"],
            rollback=bundle["rollback"],
            approvals=bundle["approvals"],
            artifacts=bundle["artifacts"],
            provenance=bundle["provenance"],
        ) is False

    def test_missing_load_fails(self) -> None:
        """Missing load section must cause failure."""
        bundle = _make_minimal_bundle()
        assert _evaluate_passed(
            ci=bundle["ci"],
            metrics=bundle["metrics"],
            migration=bundle["migration"],
            e2e=bundle["e2e"],
            load=None,  # type: ignore[arg-type]
            signature=bundle["signature"],
            rollback=bundle["rollback"],
            approvals=bundle["approvals"],
            artifacts=bundle["artifacts"],
            provenance=bundle["provenance"],
        ) is False

    def test_missing_signature_fails(self) -> None:
        """Missing signature section must cause failure."""
        bundle = _make_minimal_bundle()
        assert _evaluate_passed(
            ci=bundle["ci"],
            metrics=bundle["metrics"],
            migration=bundle["migration"],
            e2e=bundle["e2e"],
            load=bundle["load"],
            signature=None,  # type: ignore[arg-type]
            rollback=bundle["rollback"],
            approvals=bundle["approvals"],
            artifacts=bundle["artifacts"],
            provenance=bundle["provenance"],
        ) is False

    def test_missing_rollback_fails(self) -> None:
        """Missing rollback section must cause failure."""
        bundle = _make_minimal_bundle()
        assert _evaluate_passed(
            ci=bundle["ci"],
            metrics=bundle["metrics"],
            migration=bundle["migration"],
            e2e=bundle["e2e"],
            load=bundle["load"],
            signature=bundle["signature"],
            rollback=None,  # type: ignore[arg-type]
            approvals=bundle["approvals"],
            artifacts=bundle["artifacts"],
            provenance=bundle["provenance"],
        ) is False

    def test_missing_approvals_fails(self) -> None:
        """Missing approvals section must cause failure."""
        bundle = _make_minimal_bundle()
        assert _evaluate_passed(
            ci=bundle["ci"],
            metrics=bundle["metrics"],
            migration=bundle["migration"],
            e2e=bundle["e2e"],
            load=bundle["load"],
            signature=bundle["signature"],
            rollback=bundle["rollback"],
            approvals=None,  # type: ignore[arg-type]
            artifacts=bundle["artifacts"],
            provenance=bundle["provenance"],
        ) is False

    def test_uncomputed_signature_fails(self) -> None:
        """Signature with computed=false must cause failure."""
        bundle = _make_minimal_bundle()
        bundle["signature"]["computed"] = False
        assert _evaluate_passed(
            ci=bundle["ci"],
            metrics=bundle["metrics"],
            migration=bundle["migration"],
            e2e=bundle["e2e"],
            load=bundle["load"],
            signature=bundle["signature"],
            rollback=bundle["rollback"],
            approvals=bundle["approvals"],
            artifacts=bundle["artifacts"],
            provenance=bundle["provenance"],
        ) is False

    def test_mypy_failure_fails(self) -> None:
        """Non-zero mypy errors must cause failure."""
        bundle = _make_minimal_bundle()
        bundle["ci"]["mypy"]["errors"] = 5
        bundle["ci"]["mypy"]["passed"] = False
        assert _evaluate_passed(
            ci=bundle["ci"],
            metrics=bundle["metrics"],
            migration=bundle["migration"],
            e2e=bundle["e2e"],
            load=bundle["load"],
            signature=bundle["signature"],
            rollback=bundle["rollback"],
            approvals=bundle["approvals"],
            artifacts=bundle["artifacts"],
            provenance=bundle["provenance"],
        ) is False

    def test_ruff_failure_fails(self) -> None:
        """Non-zero ruff errors must cause failure."""
        bundle = _make_minimal_bundle()
        bundle["ci"]["ruff"]["errors"] = 3
        bundle["ci"]["ruff"]["passed"] = False
        assert _evaluate_passed(
            ci=bundle["ci"],
            metrics=bundle["metrics"],
            migration=bundle["migration"],
            e2e=bundle["e2e"],
            load=bundle["load"],
            signature=bundle["signature"],
            rollback=bundle["rollback"],
            approvals=bundle["approvals"],
            artifacts=bundle["artifacts"],
            provenance=bundle["provenance"],
        ) is False


# ---------------------------------------------------------------------------
# Tests: Signature computation
# ---------------------------------------------------------------------------

class TestSignature:
    """Verify signature computation is deterministic and excludes itself."""

    def test_signature_is_sha256(self) -> None:
        """Signature algorithm must be sha256."""
        bundle = _make_minimal_bundle()
        sig = _compute_signature(bundle)
        assert sig["algorithm"] == "sha256"
        assert sig["computed"] is True
        assert len(sig["hash"]) == 64  # SHA-256 hex digest length

    def test_signature_is_deterministic(self) -> None:
        """Same input must produce same signature."""
        bundle = _make_minimal_bundle()
        sig1 = _compute_signature(bundle)
        sig2 = _compute_signature(bundle)
        assert sig1["hash"] == sig2["hash"]

    def test_signature_excludes_itself(self) -> None:
        """Signature must not include the signature section in computation."""
        bundle = _make_minimal_bundle()
        bundle["signature"] = {"algorithm": "sha256", "computed": False, "hash": "old"}
        sig1 = _compute_signature(bundle)
        bundle["signature"] = {"algorithm": "sha256", "computed": True, "hash": "different"}
        sig2 = _compute_signature(bundle)
        assert sig1["hash"] == sig2["hash"]


# ---------------------------------------------------------------------------
# Tests: Voice status
# ---------------------------------------------------------------------------

class TestVoiceStatus:
    """Verify Voice is correctly marked as NOT_ACCEPTED."""

    def test_voice_not_accepted_by_default(self) -> None:
        """Voice must be NOT_ACCEPTED when no LiveKit evidence exists."""
        with patch("build_v12_acceptance_bundle.get_git_sha", return_value="a" * 40), \
             patch("build_v12_acceptance_bundle.get_mypy_result", return_value={"errors": 0, "passed": True, "output_snippet": ""}), \
             patch("build_v12_acceptance_bundle.get_ruff_result", return_value={"errors": 0, "passed": True}):
            bundle = build_bundle(candidate_sha="a" * 40)
        assert bundle["voice"]["status"] == "NOT_ACCEPTED"
        assert bundle["voice"]["default_enabled"] is False

    def test_voice_mentions_missing_livekit_evidence(self) -> None:
        """Voice reason must mention missing LiveKit evidence."""
        with patch("build_v12_acceptance_bundle.get_git_sha", return_value="a" * 40), \
             patch("build_v12_acceptance_bundle.get_mypy_result", return_value={"errors": 0, "passed": True, "output_snippet": ""}), \
             patch("build_v12_acceptance_bundle.get_ruff_result", return_value={"errors": 0, "passed": True}):
            bundle = build_bundle(candidate_sha="a" * 40)
        assert "LiveKit" in bundle["voice"]["reason"]
        assert "NOT" in bundle["voice"]["status"]


# ---------------------------------------------------------------------------
# Tests: Markdown generation
# ---------------------------------------------------------------------------

class TestMarkdownGeneration:
    """Verify Markdown is correctly generated from JSON bundle."""

    def test_final_acceptance_md_contains_passed_status(self) -> None:
        """Generated Markdown must contain the passed status."""
        bundle = _make_minimal_bundle()
        md = generate_final_acceptance_md(bundle)
        assert "Passed" in md
        assert "YES" in md or "NO" in md

    def test_final_acceptance_md_contains_voice_status(self) -> None:
        """Generated Markdown must mention Voice status."""
        bundle = _make_minimal_bundle()
        md = generate_final_acceptance_md(bundle)
        assert "Voice" in md
        assert "NOT_ACCEPTED" in md

    def test_acceptance_summary_md_contains_metrics(self) -> None:
        """Generated summary must contain test metrics."""
        bundle = _make_minimal_bundle()
        md = generate_acceptance_summary_md(bundle)
        assert "2103+" in md
        assert "106" in md

    def test_markdown_is_auto_generated(self) -> None:
        """Generated Markdown must indicate it's auto-generated."""
        bundle = _make_minimal_bundle()
        md = generate_final_acceptance_md(bundle)
        assert "auto-generated" in md
        md2 = generate_acceptance_summary_md(bundle)
        assert "auto-generated" in md2


# ---------------------------------------------------------------------------
# Tests: Required sections constant
# ---------------------------------------------------------------------------

class TestRequiredSections:
    """Verify the REQUIRED_SECTIONS constant includes all mandatory sections."""

    def test_all_required_sections_defined(self) -> None:
        """All 9 required sections must be in the constant."""
        expected = {"metrics", "artifacts", "provenance", "migration", "e2e", "load", "signature", "rollback", "approvals"}
        assert REQUIRED_SECTIONS == expected

    def test_required_sections_count(self) -> None:
        """There must be exactly 9 required sections."""
        assert len(REQUIRED_SECTIONS) == 9
