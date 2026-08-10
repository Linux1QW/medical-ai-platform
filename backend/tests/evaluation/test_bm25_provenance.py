"""Tests for BM25 provenance validation."""
import json
import pytest
from pathlib import Path

from evaluation.provenance import (
    ProvenanceError,
    sha256_file,
    validate_baseline_provenance,
)


@pytest.fixture
def provenance_bundle(tmp_path):
    """Create a valid provenance bundle for testing."""
    # Create dummy files
    baseline = tmp_path / "baseline-report.json"
    baseline.write_text('{"index_generation": "rag-20260101-abc", "metrics": {}}')

    golden = tmp_path / "golden-set.json"
    golden.write_text('[]')

    evaluator = tmp_path / "evaluator.py"
    evaluator.write_text("# evaluator")

    manifest = tmp_path / "index-manifest.json"
    manifest.write_text('{"schema_version": 1}')

    provenance = tmp_path / "provenance.json"
    prov_data = {
        "schema_version": 1,
        "source_commit": "a" * 40,
        "workflow_run_id": "run-123",
        "index_generation": "rag-20260101-abc",
        "index_manifest_sha256": sha256_file(manifest),
        "report_sha256": sha256_file(baseline),
        "golden_sha256": sha256_file(golden),
        "evaluator_sha256": sha256_file(evaluator),
        "created_at": "2026-01-01T00:00:00Z",
    }
    provenance.write_text(json.dumps(prov_data))

    return {
        "provenance_path": provenance,
        "baseline_path": baseline,
        "golden_path": golden,
        "evaluator_path": evaluator,
        "manifest_path": manifest,
        "candidate_commit": "b" * 40,
    }


def test_valid_provenance_passes(provenance_bundle):
    result = validate_baseline_provenance(**provenance_bundle)
    assert result["validated"] is True


def test_rejects_candidate_as_own_baseline(provenance_bundle):
    provenance_bundle["candidate_commit"] = "a" * 40  # same as source_commit
    with pytest.raises(ProvenanceError, match="candidate commit"):
        validate_baseline_provenance(**provenance_bundle)


def test_rejects_missing_field(provenance_bundle):
    prov = json.loads(provenance_bundle["provenance_path"].read_text())
    del prov["schema_version"]
    provenance_bundle["provenance_path"].write_text(json.dumps(prov))
    with pytest.raises(ProvenanceError, match="missing"):
        validate_baseline_provenance(**provenance_bundle)


def test_rejects_sha_mismatch(provenance_bundle):
    # Modify baseline file after provenance was computed
    provenance_bundle["baseline_path"].write_text('{"modified": true}')
    with pytest.raises(ProvenanceError, match="SHA-256"):
        validate_baseline_provenance(**provenance_bundle)
