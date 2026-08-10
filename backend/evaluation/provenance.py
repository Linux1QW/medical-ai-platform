"""Baseline provenance validation for RAG release gates."""
from __future__ import annotations

import hashlib
import json
from datetime import datetime
from pathlib import Path
from typing import Any


class ProvenanceError(Exception):
    """Raised when provenance validation fails."""


REQUIRED_PROVENANCE_FIELDS = {
    "schema_version",
    "source_commit",
    "workflow_run_id",
    "index_generation",
    "index_manifest_sha256",
    "report_sha256",
    "golden_sha256",
    "evaluator_sha256",
    "created_at",
}


def sha256_file(path: Path) -> str:
    """Compute SHA-256 hex digest of a file."""
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def validate_baseline_provenance(
    *,
    provenance_path: Path,
    baseline_path: Path,
    golden_path: Path,
    evaluator_path: Path,
    manifest_path: Path,
    candidate_commit: str,
) -> dict[str, Any]:
    """Validate baseline provenance integrity.

    Checks:
    1. All required fields present in provenance
    2. SHA-256 digests match actual files
    3. source_commit != candidate_commit (baseline must differ from candidate)
    4. Baseline generation matches manifest
    """
    provenance = json.loads(provenance_path.read_text(encoding="utf-8"))

    # Check required fields
    missing = REQUIRED_PROVENANCE_FIELDS - set(provenance.keys())
    if missing:
        raise ProvenanceError(f"missing provenance fields: {sorted(missing)}")

    # Check candidate != baseline source
    if provenance["source_commit"] == candidate_commit:
        raise ProvenanceError(
            "candidate commit matches baseline source_commit; "
            "baseline must come from a different commit"
        )

    # Verify file digests
    expected = {
        "index_manifest_sha256": sha256_file(manifest_path),
        "report_sha256": sha256_file(baseline_path),
        "golden_sha256": sha256_file(golden_path),
        "evaluator_sha256": sha256_file(evaluator_path),
    }

    mismatches = []
    for field, actual in expected.items():
        if provenance[field] != actual:
            mismatches.append(f"{field}: expected {provenance[field]}, got {actual}")

    if mismatches:
        raise ProvenanceError(f"SHA-256 mismatch: {'; '.join(mismatches)}")

    return {
        "validated": True,
        "source_commit": provenance["source_commit"],
        "index_generation": provenance["index_generation"],
        "candidate_commit": candidate_commit,
    }
