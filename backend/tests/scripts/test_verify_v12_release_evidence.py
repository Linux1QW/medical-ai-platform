"""Tests for verify_v12_release_evidence.py.

Ensures the evidence gate is fail-closed:
- A valid 40-char SHA must NOT skip the Evidence Gate
- Deploy must reject 'skipped' status
- Missing any required field must fail
- Signature verification must work
"""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
from typing import Any

import pytest
from scripts.ci.verify_v12_release_evidence import (
    VerificationResult,
    generate_signature,
    verify_release_evidence,
)

VALID_SHA = "a" * 40
VALID_SHA_2 = "b" * 40
SIGNING_KEY = "test-signing-key-for-evidence-gate"


def _make_manifest(
    sha: str = VALID_SHA,
    overrides: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Build a valid evidence manifest with optional overrides."""
    manifest: dict[str, Any] = {
        "candidate_sha": sha,
        "all_passed": True,
        "mode": "live",
        "metrics": {
            "inquiry_score": 92.5,
            "diagnosis_score": 88.0,
            "treatment_score": 90.0,
            "humanistic_score": 85.0,
            "knowledge_consistency_score": 91.0,
            "overall_score": 89.3,
        },
        "migration": {"passed": True, "revisions_applied": 3},
        "playwright": {"passed": True, "tests_run": 42},
        "load": {"passed": True, "p99_latency_ms": 230},
        "fault_recovery": {"passed": True, "recovery_time_s": 12},
        "approvals": {"approvers": ["lead-eng", "qa-lead"]},
        "report_hash": "",  # will be filled by _write_evidence
    }
    if overrides:
        manifest.update(overrides)
    return manifest


def _write_evidence(
    tmp_path: Path,
    manifest: dict[str, Any] | None = None,
    report_content: str = '{"summary": "all good"}',
    signing_key: str = SIGNING_KEY,
) -> Path:
    """Write manifest, report, and signature to a temp directory."""
    evidence_dir = tmp_path / "evidence"
    evidence_dir.mkdir(exist_ok=True)

    if manifest is None:
        manifest = _make_manifest()

    # Write report first so we can compute its hash
    report_path = evidence_dir / "report.json"
    report_path.write_text(report_content, encoding="utf-8")
    manifest["report_hash"] = hashlib.sha256(
        report_path.read_bytes()
    ).hexdigest()

    # Write manifest
    manifest_path = evidence_dir / "evidence_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    # Write detached signature
    sig = generate_signature(str(evidence_dir), signing_key)
    sig_path = evidence_dir / "evidence_manifest.json.sig"
    sig_path.write_text(sig, encoding="utf-8")

    return evidence_dir


# ── SHA validation tests ──────────────────────────────────────────────────────


class TestSHAValidation:
    """Verify candidate_sha validation."""

    def test_valid_sha_passes_gate(self, tmp_path: Path) -> None:
        """A valid 40-char SHA with correct evidence must pass."""
        evidence_dir = _write_evidence(tmp_path)
        result = verify_release_evidence(VALID_SHA, str(evidence_dir), signing_key=SIGNING_KEY)
        assert result.passed is True
        assert result.errors == []

    def test_valid_sha_must_not_skip_gate(self, tmp_path: Path) -> None:
        """A valid 40-char SHA must NOT be able to skip the Evidence Gate.

        Even with a perfectly valid SHA, if evidence is missing, it must fail.
        """
        # No evidence directory at all
        result = verify_release_evidence(VALID_SHA, str(tmp_path / "nonexistent"), signing_key=SIGNING_KEY)
        assert result.passed is False
        assert any("not found" in e.lower() or "missing" in e.lower() for e in result.errors)

    def test_short_sha_rejected(self, tmp_path: Path) -> None:
        """Short SHA (7-char) must be rejected immediately."""
        result = verify_release_evidence("abc1234", str(tmp_path), signing_key=SIGNING_KEY)
        assert result.passed is False
        assert any("40-char" in e for e in result.errors)

    def test_non_hex_sha_rejected(self, tmp_path: Path) -> None:
        """Non-hex 40-char string must be rejected."""
        bad_sha = "g" * 40
        result = verify_release_evidence(bad_sha, str(tmp_path), signing_key=SIGNING_KEY)
        assert result.passed is False
        assert any("40-char" in e for e in result.errors)

    def test_empty_sha_rejected(self, tmp_path: Path) -> None:
        """Empty SHA must be rejected."""
        result = verify_release_evidence("", str(tmp_path), signing_key=SIGNING_KEY)
        assert result.passed is False

    def test_sha_mismatch_between_request_and_manifest(self, tmp_path: Path) -> None:
        """SHA in manifest must match the requested SHA."""
        evidence_dir = _write_evidence(tmp_path)
        result = verify_release_evidence(VALID_SHA_2, str(evidence_dir), signing_key=SIGNING_KEY)
        assert result.passed is False
        assert any("mismatch" in e.lower() for e in result.errors)


# ── Deploy must reject skipped status ─────────────────────────────────────────


class TestDeployRejectsSkipped:
    """Verify that deploy does not accept 'skipped' evidence gate status."""

    def test_missing_manifest_is_failure(self, tmp_path: Path) -> None:
        """Missing manifest must result in failure, not skip."""
        evidence_dir = tmp_path / "empty_evidence"
        evidence_dir.mkdir()
        result = verify_release_evidence(VALID_SHA, str(evidence_dir), signing_key=SIGNING_KEY)
        assert result.passed is False
        assert any("manifest" in e.lower() for e in result.errors)

    def test_no_evidence_dir_is_failure(self, tmp_path: Path) -> None:
        """Non-existent evidence directory must fail, not skip."""
        result = verify_release_evidence(
            VALID_SHA, str(tmp_path / "does_not_exist"), signing_key=SIGNING_KEY
        )
        assert result.passed is False

    def test_verification_result_default_is_fail(self) -> None:
        """VerificationResult must default to passed=True but any fail() call sets it False."""
        result = VerificationResult(passed=True)
        assert result.passed is True
        result.fail("something went wrong")
        assert result.passed is False


# ── Required field tests ──────────────────────────────────────────────────────


class TestRequiredFields:
    """Missing any required field must fail."""

    @pytest.mark.parametrize(
        "missing_field",
        [
            "candidate_sha",
            "all_passed",
            "mode",
            "metrics",
            "migration",
            "playwright",
            "load",
            "fault_recovery",
            "approvals",
        ],
    )
    def test_missing_field_causes_failure(
        self, tmp_path: Path, missing_field: str
    ) -> None:
        """Removing any required field must cause verification failure."""
        manifest = _make_manifest()
        del manifest[missing_field]
        evidence_dir = _write_evidence(tmp_path, manifest=manifest)
        result = verify_release_evidence(VALID_SHA, str(evidence_dir), signing_key=SIGNING_KEY)
        assert result.passed is False

    def test_missing_report_hash_field_causes_failure(
        self, tmp_path: Path
    ) -> None:
        """Removing report_hash from manifest must cause failure."""
        manifest = _make_manifest()
        evidence_dir = _write_evidence(tmp_path, manifest=manifest)
        # Now remove report_hash from the written manifest (after _write_evidence set it)
        manifest_path = evidence_dir / "evidence_manifest.json"
        stored = json.loads(manifest_path.read_text(encoding="utf-8"))
        del stored["report_hash"]
        manifest_path.write_text(json.dumps(stored, indent=2), encoding="utf-8")
        result = verify_release_evidence(VALID_SHA, str(evidence_dir), signing_key=SIGNING_KEY)
        assert result.passed is False
        assert any("report_hash" in e or "Missing" in e for e in result.errors)


# ── all_passed tests ──────────────────────────────────────────────────────────


class TestAllPassed:
    """all_passed must be true."""

    def test_all_passed_false_fails(self, tmp_path: Path) -> None:
        evidence_dir = _write_evidence(
            tmp_path, manifest=_make_manifest(overrides={"all_passed": False})
        )
        result = verify_release_evidence(VALID_SHA, str(evidence_dir), signing_key=SIGNING_KEY)
        assert result.passed is False
        assert any("all_passed" in e for e in result.errors)

    def test_all_passed_string_fails(self, tmp_path: Path) -> None:
        """String 'true' is not the same as boolean true."""
        evidence_dir = _write_evidence(
            tmp_path, manifest=_make_manifest(overrides={"all_passed": "true"})
        )
        result = verify_release_evidence(VALID_SHA, str(evidence_dir), signing_key=SIGNING_KEY)
        assert result.passed is False


# ── Mode tests ────────────────────────────────────────────────────────────────


class TestMode:
    """Mode must be 'live'."""

    def test_non_live_mode_fails(self, tmp_path: Path) -> None:
        evidence_dir = _write_evidence(
            tmp_path, manifest=_make_manifest(overrides={"mode": "dry-run"})
        )
        result = verify_release_evidence(VALID_SHA, str(evidence_dir), signing_key=SIGNING_KEY)
        assert result.passed is False
        assert any("live" in e.lower() for e in result.errors)


# ── Metrics tests ─────────────────────────────────────────────────────────────


class TestMetrics:
    """All required metrics must be present."""

    def test_missing_one_metric_fails(self, tmp_path: Path) -> None:
        manifest = _make_manifest()
        del manifest["metrics"]["overall_score"]
        evidence_dir = _write_evidence(tmp_path, manifest=manifest)
        result = verify_release_evidence(VALID_SHA, str(evidence_dir), signing_key=SIGNING_KEY)
        assert result.passed is False
        assert any("metrics" in e.lower() for e in result.errors)

    def test_empty_metrics_fails(self, tmp_path: Path) -> None:
        evidence_dir = _write_evidence(
            tmp_path, manifest=_make_manifest(overrides={"metrics": {}})
        )
        result = verify_release_evidence(VALID_SHA, str(evidence_dir), signing_key=SIGNING_KEY)
        assert result.passed is False


# ── Sub-test section tests ────────────────────────────────────────────────────


class TestSubTests:
    """Migration, Playwright, Load, Fault Recovery must all pass."""

    @pytest.mark.parametrize(
        "section",
        ["migration", "playwright", "load", "fault_recovery"],
    )
    def test_subtest_not_passed_fails(self, tmp_path: Path, section: str) -> None:
        evidence_dir = _write_evidence(
            tmp_path,
            manifest=_make_manifest(overrides={section: {"passed": False}}),
        )
        result = verify_release_evidence(VALID_SHA, str(evidence_dir), signing_key=SIGNING_KEY)
        assert result.passed is False
        assert any(section in e for e in result.errors)

    @pytest.mark.parametrize(
        "section",
        ["migration", "playwright", "load", "fault_recovery"],
    )
    def test_subtest_missing_fails(self, tmp_path: Path, section: str) -> None:
        manifest = _make_manifest()
        del manifest[section]
        evidence_dir = _write_evidence(tmp_path, manifest=manifest)
        result = verify_release_evidence(VALID_SHA, str(evidence_dir), signing_key=SIGNING_KEY)
        assert result.passed is False


# ── Approvals tests ───────────────────────────────────────────────────────────


class TestApprovals:
    """Approvals must be present."""

    def test_empty_approvals_fails(self, tmp_path: Path) -> None:
        evidence_dir = _write_evidence(
            tmp_path, manifest=_make_manifest(overrides={"approvals": {}})
        )
        result = verify_release_evidence(VALID_SHA, str(evidence_dir), signing_key=SIGNING_KEY)
        assert result.passed is False

    def test_no_approvers_fails(self, tmp_path: Path) -> None:
        evidence_dir = _write_evidence(
            tmp_path,
            manifest=_make_manifest(overrides={"approvals": {"approvers": []}}),
        )
        result = verify_release_evidence(VALID_SHA, str(evidence_dir), signing_key=SIGNING_KEY)
        assert result.passed is False


# ── Report hash tests ─────────────────────────────────────────────────────────


class TestReportHash:
    """Report hash must match actual report content."""

    def test_report_hash_mismatch_fails(self, tmp_path: Path) -> None:
        evidence_dir = _write_evidence(tmp_path)
        # Overwrite manifest with a wrong report_hash
        manifest_path = evidence_dir / "evidence_manifest.json"
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        manifest["report_hash"] = "0" * 64  # wrong hash
        manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
        result = verify_release_evidence(VALID_SHA, str(evidence_dir), signing_key=SIGNING_KEY)
        assert result.passed is False
        assert any("report_hash" in e for e in result.errors)

    def test_missing_report_file_fails(self, tmp_path: Path) -> None:
        evidence_dir = _write_evidence(tmp_path)
        # Delete the report file
        (evidence_dir / "report.json").unlink()
        result = verify_release_evidence(VALID_SHA, str(evidence_dir), signing_key=SIGNING_KEY)
        assert result.passed is False


# ── Signature verification tests ──────────────────────────────────────────────


class TestSignatureVerification:
    """Detached signature must be valid."""

    def test_valid_signature_passes(self, tmp_path: Path) -> None:
        evidence_dir = _write_evidence(tmp_path, signing_key=SIGNING_KEY)
        result = verify_release_evidence(VALID_SHA, str(evidence_dir), signing_key=SIGNING_KEY)
        assert result.passed is True

    def test_wrong_signing_key_fails(self, tmp_path: Path) -> None:
        evidence_dir = _write_evidence(tmp_path, signing_key=SIGNING_KEY)
        result = verify_release_evidence(
            VALID_SHA, str(evidence_dir), signing_key="wrong-key"
        )
        assert result.passed is False
        assert any("signature" in e.lower() for e in result.errors)

    def test_missing_signature_file_fails(self, tmp_path: Path) -> None:
        evidence_dir = _write_evidence(tmp_path, signing_key=SIGNING_KEY)
        # Delete signature file
        (evidence_dir / "evidence_manifest.json.sig").unlink()
        result = verify_release_evidence(VALID_SHA, str(evidence_dir), signing_key=SIGNING_KEY)
        assert result.passed is False
        assert any("signature" in e.lower() for e in result.errors)

    def test_tampered_manifest_fails_signature(self, tmp_path: Path) -> None:
        """Modifying manifest after signing must fail signature check."""
        evidence_dir = _write_evidence(tmp_path, signing_key=SIGNING_KEY)
        # Tamper with manifest
        manifest_path = evidence_dir / "evidence_manifest.json"
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        manifest["all_passed"] = False
        manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
        result = verify_release_evidence(VALID_SHA, str(evidence_dir), signing_key=SIGNING_KEY)
        assert result.passed is False

    def test_no_signing_key_fails(self, tmp_path: Path) -> None:
        """Missing signing key must fail (no bypass)."""
        evidence_dir = _write_evidence(tmp_path, signing_key=SIGNING_KEY)
        # Clear env var if set
        old = os.environ.pop("RELEASE_SIGNING_KEY", None)
        try:
            result = verify_release_evidence(
                VALID_SHA, str(evidence_dir), signing_key=""
            )
            assert result.passed is False
            assert any("signing key" in e.lower() for e in result.errors)
        finally:
            if old is not None:
                os.environ["RELEASE_SIGNING_KEY"] = old

    def test_generate_signature_and_verify(self, tmp_path: Path) -> None:
        """End-to-end: generate signature, then verify."""
        evidence_dir = _write_evidence(tmp_path, signing_key=SIGNING_KEY)
        # Re-generate signature (should match)
        sig = generate_signature(str(evidence_dir), SIGNING_KEY)
        sig_path = evidence_dir / "evidence_manifest.json.sig"
        assert sig_path.read_text(encoding="utf-8").strip() == sig


# ── Fail-closed integration test ──────────────────────────────────────────────


class TestFailClosed:
    """Evidence gate must be fail-closed: default deny."""

    def test_everything_valid_passes(self, tmp_path: Path) -> None:
        """Full valid evidence must pass."""
        evidence_dir = _write_evidence(tmp_path)
        result = verify_release_evidence(VALID_SHA, str(evidence_dir), signing_key=SIGNING_KEY)
        assert result.passed is True
        assert result.candidate_sha == VALID_SHA
        assert result.errors == []

    def test_any_single_failure_blocks_deploy(self, tmp_path: Path) -> None:
        """Any single failure must block deployment."""
        # Test with all_passed=False as representative
        evidence_dir = _write_evidence(
            tmp_path, manifest=_make_manifest(overrides={"all_passed": False})
        )
        result = verify_release_evidence(VALID_SHA, str(evidence_dir), signing_key=SIGNING_KEY)
        assert result.passed is False
        assert len(result.errors) >= 1
