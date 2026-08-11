"""Verify V1.2 release evidence before deployment.

This script enforces a fail-closed evidence gate: every deployment MUST pass
all checks or be rejected. No bypass, no skip, no default-allow.

Checks performed:
- candidate_sha exists and is a full 40-character hex SHA
- all_passed is true
- mode is "live"
- All required metrics exist
- Migration tests passed
- Playwright tests passed
- Load tests passed
- Fault Recovery tests passed
- Approvals present
- Report hash matches actual report content
- Detached cryptographic signature is valid

Usage:
    python verify_v12_release_evidence.py \\
        --candidate-sha <40-char-sha> \\
        --evidence-dir <path-to-evidence-dir>
"""
from __future__ import annotations

import argparse
import hashlib
import hmac
import json
import os
import re
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

REQUIRED_METRICS = frozenset({
    "inquiry_score",
    "diagnosis_score",
    "treatment_score",
    "humanistic_score",
    "knowledge_consistency_score",
    "overall_score",
})

REQUIRED_FIELDS = frozenset({
    "candidate_sha",
    "all_passed",
    "mode",
    "metrics",
    "migration",
    "playwright",
    "load",
    "fault_recovery",
    "approvals",
    "report_hash",
})

SHA_PATTERN = re.compile(r"^[0-9a-f]{40}$")


@dataclass
class VerificationResult:
    """Result of evidence verification."""

    passed: bool
    candidate_sha: str = ""
    errors: list[str] = field(default_factory=list)

    def fail(self, reason: str) -> None:
        """Record a failure reason."""
        self.errors.append(reason)
        self.passed = False


def _load_manifest(evidence_dir: Path) -> dict[str, Any] | None:
    """Load the evidence manifest JSON file."""
    manifest_path = evidence_dir / "evidence_manifest.json"
    if not manifest_path.is_file():
        return None
    with open(manifest_path, encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, dict):
        return None
    return data


def _load_signature(evidence_dir: Path) -> str | None:
    """Load the detached signature file."""
    sig_path = evidence_dir / "evidence_manifest.json.sig"
    if not sig_path.is_file():
        return None
    return sig_path.read_text(encoding="utf-8").strip()


def _compute_manifest_hash(evidence_dir: Path) -> str | None:
    """Compute SHA-256 hash of the manifest file content."""
    manifest_path = evidence_dir / "evidence_manifest.json"
    if not manifest_path.is_file():
        return None
    content = manifest_path.read_bytes()
    return hashlib.sha256(content).hexdigest()


def _verify_signature(
    evidence_dir: Path,
    signing_key: str,
) -> tuple[bool, str]:
    """Verify the detached HMAC-SHA256 signature of the manifest.

    Returns (is_valid, message).
    """
    sig = _load_signature(evidence_dir)
    if sig is None:
        return False, "Detached signature file (evidence_manifest.json.sig) not found"

    manifest_hash = _compute_manifest_hash(evidence_dir)
    if manifest_hash is None:
        return False, "Cannot compute manifest hash for signature verification"

    expected_sig = hmac.new(
        signing_key.encode("utf-8"),
        manifest_hash.encode("utf-8"),
        hashlib.sha256,
    ).hexdigest()

    if not hmac.compare_digest(sig, expected_sig):
        return False, "Signature verification failed: HMAC mismatch"

    return True, "Signature valid"


def _verify_sha(manifest: dict[str, Any], candidate_sha: str, result: VerificationResult) -> None:
    """Verify candidate_sha field."""
    manifest_sha = manifest.get("candidate_sha", "")
    if not manifest_sha:
        result.fail("candidate_sha is missing from manifest")
        return
    if not SHA_PATTERN.match(manifest_sha):
        result.fail(f"candidate_sha in manifest is not a valid 40-char SHA: {manifest_sha!r}")
        return
    if manifest_sha != candidate_sha:
        result.fail(
            f"candidate_sha mismatch: manifest={manifest_sha}, requested={candidate_sha}"
        )


def _verify_all_passed(manifest: dict[str, Any], result: VerificationResult) -> None:
    """Verify all_passed is true."""
    if manifest.get("all_passed") is not True:
        result.fail(f"all_passed is not true: {manifest.get('all_passed')!r}")


def _verify_mode(manifest: dict[str, Any], result: VerificationResult) -> None:
    """Verify mode is live."""
    if manifest.get("mode") != "live":
        result.fail(f"mode is not 'live': {manifest.get('mode')!r}")


def _verify_metrics(manifest: dict[str, Any], result: VerificationResult) -> None:
    """Verify all required metrics exist."""
    metrics = manifest.get("metrics")
    if not isinstance(metrics, dict):
        result.fail("metrics field is missing or not a dict")
        return
    missing = REQUIRED_METRICS - set(metrics.keys())
    if missing:
        result.fail(f"Missing required metrics: {sorted(missing)}")


def _verify_sub_test(
    manifest: dict[str, Any],
    field_name: str,
    result: VerificationResult,
) -> None:
    """Verify a sub-test section (migration, playwright, load, fault_recovery)."""
    section = manifest.get(field_name)
    if not isinstance(section, dict):
        result.fail(f"{field_name} field is missing or not a dict")
        return
    if section.get("passed") is not True:
        result.fail(f"{field_name} did not pass: {section.get('passed')!r}")


def _verify_approvals(manifest: dict[str, Any], result: VerificationResult) -> None:
    """Verify approvals exist."""
    approvals = manifest.get("approvals")
    if not approvals:
        result.fail("approvals field is missing or empty")
        return
    if not isinstance(approvals, dict):
        result.fail("approvals field is not a dict")
        return
    if not approvals.get("approvers"):
        result.fail("approvals.approvers is empty or missing")


def _verify_report_hash(manifest: dict[str, Any], evidence_dir: Path, result: VerificationResult) -> None:
    """Verify report hash matches actual report content."""
    expected_hash = manifest.get("report_hash")
    if not expected_hash:
        result.fail("report_hash is missing from manifest")
        return

    report_path = evidence_dir / "report.json"
    if not report_path.is_file():
        result.fail("report.json not found in evidence directory")
        return

    actual_hash = hashlib.sha256(report_path.read_bytes()).hexdigest()
    if actual_hash != expected_hash:
        result.fail(
            f"report_hash mismatch: manifest={expected_hash}, actual={actual_hash}"
        )


def verify_release_evidence(
    candidate_sha: str,
    evidence_dir: str,
    signing_key: str | None = None,
) -> VerificationResult:
    """Verify all release evidence for a candidate SHA.

    This function is fail-closed: any missing or invalid evidence results
    in a failed verification.

    Args:
        candidate_sha: The 40-character commit SHA to verify.
        evidence_dir: Path to the directory containing evidence files.
        signing_key: Optional signing key for signature verification.
                     If None, reads from RELEASE_SIGNING_KEY env var.

    Returns:
        VerificationResult with passed=True only if ALL checks pass.
    """
    result = VerificationResult(passed=True, candidate_sha=candidate_sha)
    evidence_path = Path(evidence_dir)

    # Validate candidate_sha format
    if not SHA_PATTERN.match(candidate_sha):
        result.fail(f"candidate_sha is not a valid 40-char hex SHA: {candidate_sha!r}")
        return result

    # Load manifest
    manifest = _load_manifest(evidence_path)
    if manifest is None:
        result.fail("Evidence manifest not found: evidence_manifest.json")
        return result

    # Check all required top-level fields exist
    missing_fields = REQUIRED_FIELDS - set(manifest.keys())
    if missing_fields:
        result.fail(f"Missing required manifest fields: {sorted(missing_fields)}")
        return result

    # Run all verifications
    _verify_sha(manifest, candidate_sha, result)
    _verify_all_passed(manifest, result)
    _verify_mode(manifest, result)
    _verify_metrics(manifest, result)
    _verify_sub_test(manifest, "migration", result)
    _verify_sub_test(manifest, "playwright", result)
    _verify_sub_test(manifest, "load", result)
    _verify_sub_test(manifest, "fault_recovery", result)
    _verify_approvals(manifest, result)
    _verify_report_hash(manifest, evidence_path, result)

    # Signature verification
    effective_key = signing_key or os.environ.get("RELEASE_SIGNING_KEY", "")
    if not effective_key:
        result.fail(
            "No signing key provided. Set RELEASE_SIGNING_KEY env var "
            "or pass signing_key parameter"
        )
    else:
        sig_valid, sig_msg = _verify_signature(evidence_path, effective_key)
        if not sig_valid:
            result.fail(sig_msg)

    return result


def generate_signature(evidence_dir: str, signing_key: str) -> str:
    """Generate a detached HMAC-SHA256 signature for the manifest.

    Args:
        evidence_dir: Path to the directory containing the manifest.
        signing_key: The signing key to use.

    Returns:
        The hex-encoded signature string.
    """
    evidence_path = Path(evidence_dir)
    manifest_hash = _compute_manifest_hash(evidence_path)
    if manifest_hash is None:
        raise FileNotFoundError(
            f"Manifest not found in {evidence_dir}"
        )
    return hmac.new(
        signing_key.encode("utf-8"),
        manifest_hash.encode("utf-8"),
        hashlib.sha256,
    ).hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Verify V1.2 release evidence (fail-closed)"
    )
    parser.add_argument(
        "--candidate-sha",
        required=True,
        help="Full 40-character commit SHA to verify",
    )
    parser.add_argument(
        "--evidence-dir",
        required=True,
        help="Path to directory containing evidence manifest and report",
    )
    parser.add_argument(
        "--signing-key",
        default=None,
        help="Signing key (or set RELEASE_SIGNING_KEY env var)",
    )
    args = parser.parse_args()

    result = verify_release_evidence(
        candidate_sha=args.candidate_sha,
        evidence_dir=args.evidence_dir,
        signing_key=args.signing_key,
    )

    if result.passed:
        print(f"✓ Evidence gate PASSED for {result.candidate_sha}")
        return 0

    print(f"✗ Evidence gate FAILED for {result.candidate_sha}")
    for error in result.errors:
        print(f"  - {error}")
    return 1


if __name__ == "__main__":
    sys.exit(main())
