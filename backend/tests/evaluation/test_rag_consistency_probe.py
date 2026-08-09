"""Tests for probe_rag_consistency.py – generation switch validation."""

import json
from pathlib import Path

import pytest

from scripts.eval import probe_rag_consistency


def _write_manifest(root: Path, generation: str, sha: str = "a" * 64) -> Path:
    """Write a minimal artifact manifest for a given generation."""
    manifest = {
        "generation": generation,
        "index_manifest_sha256": sha,
        "components": {
            "bm25": {"path": f"indexes/{generation}/bm25"},
            "chroma": {"path": f"indexes/{generation}/chroma"},
        },
    }
    manifest_path = root / "manifest.json"
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    return manifest_path


def test_probe_output_schema_release_generations(tmp_path):
    """When baseline and candidate generations differ, probe_mode=release_generations."""
    artifact_root = tmp_path / "artifacts"
    artifact_root.mkdir()
    baseline_dir = artifact_root / "baseline"
    baseline_dir.mkdir()
    candidate_dir = artifact_root / "candidate"
    candidate_dir.mkdir()
    _write_manifest(baseline_dir, "gen-v1")
    _write_manifest(candidate_dir, "gen-v2")

    golden = tmp_path / "golden.json"
    golden.write_text(json.dumps({"queries": ["q1", "q2"]}), encoding="utf-8")

    output = tmp_path / "report.json"

    result = probe_rag_consistency.run_probe(
        baseline_generation="gen-v1",
        candidate_generation="gen-v2",
        artifact_root=artifact_root,
        golden_path=golden,
        output_path=output,
    )

    assert result["probe_mode"] == "release_generations"
    assert result["switch_baseline_generation"] == "gen-v1"
    assert result["switch_candidate_generation"] == "gen-v2"
    assert result["validated_candidate_generation"] == "gen-v2"
    assert result["query_count"] == 2
    assert result["generation_mismatch_count"] == 0
    assert result["stale_cache_hit_count"] == 0
    assert "manifest_digest" in result
    assert "timestamp" in result
    assert output.exists()


def test_probe_synthetic_switch_when_same_generation(tmp_path):
    """When baseline == candidate, probe_mode=synthetic_switch."""
    artifact_root = tmp_path / "artifacts"
    artifact_root.mkdir()
    candidate_dir = artifact_root / "candidate"
    candidate_dir.mkdir()
    _write_manifest(candidate_dir, "gen-v1")

    golden = tmp_path / "golden.json"
    golden.write_text(json.dumps({"queries": ["q1"]}), encoding="utf-8")

    output = tmp_path / "report.json"

    result = probe_rag_consistency.run_probe(
        baseline_generation="gen-v1",
        candidate_generation="gen-v1",
        artifact_root=artifact_root,
        golden_path=golden,
        output_path=output,
    )

    assert result["probe_mode"] == "synthetic_switch"
    assert result["validated_candidate_generation"] == "gen-v1"


def test_probe_fails_on_generation_mismatch_count(tmp_path):
    """Non-zero generation_mismatch_count causes non-zero exit."""
    artifact_root = tmp_path / "artifacts"
    artifact_root.mkdir()
    baseline_dir = artifact_root / "baseline"
    baseline_dir.mkdir()
    candidate_dir = artifact_root / "candidate"
    candidate_dir.mkdir()
    _write_manifest(baseline_dir, "gen-v1")
    _write_manifest(candidate_dir, "gen-v2")

    golden = tmp_path / "golden.json"
    golden.write_text(json.dumps({"queries": ["q1"]}), encoding="utf-8")

    output = tmp_path / "report.json"

    # Simulate a probe where the candidate returns stale baseline data
    result = probe_rag_consistency.run_probe(
        baseline_generation="gen-v1",
        candidate_generation="gen-v2",
        artifact_root=artifact_root,
        golden_path=golden,
        output_path=output,
        _force_mismatch=True,
    )

    assert result["generation_mismatch_count"] > 0
    with pytest.raises(SystemExit) as exc_info:
        probe_rag_consistency.main_from_result(result)
    assert exc_info.value.code != 0


def test_probe_fails_on_stale_cache_hit(tmp_path):
    """Non-zero stale_cache_hit_count causes non-zero exit."""
    artifact_root = tmp_path / "artifacts"
    artifact_root.mkdir()
    baseline_dir = artifact_root / "baseline"
    baseline_dir.mkdir()
    candidate_dir = artifact_root / "candidate"
    candidate_dir.mkdir()
    _write_manifest(baseline_dir, "gen-v1")
    _write_manifest(candidate_dir, "gen-v2")

    golden = tmp_path / "golden.json"
    golden.write_text(json.dumps({"queries": ["q1"]}), encoding="utf-8")

    output = tmp_path / "report.json"

    result = probe_rag_consistency.run_probe(
        baseline_generation="gen-v1",
        candidate_generation="gen-v2",
        artifact_root=artifact_root,
        golden_path=golden,
        output_path=output,
        _force_stale_cache=True,
    )

    assert result["stale_cache_hit_count"] > 0
    with pytest.raises(SystemExit) as exc_info:
        probe_rag_consistency.main_from_result(result)
    assert exc_info.value.code != 0


def test_probe_restores_pointer_on_failure(tmp_path):
    """Even on failure, the initial pointer is restored in finally block."""
    artifact_root = tmp_path / "artifacts"
    artifact_root.mkdir()
    baseline_dir = artifact_root / "baseline"
    baseline_dir.mkdir()
    candidate_dir = artifact_root / "candidate"
    candidate_dir.mkdir()
    _write_manifest(baseline_dir, "gen-v1")
    _write_manifest(candidate_dir, "gen-v2")

    golden = tmp_path / "golden.json"
    golden.write_text(json.dumps({"queries": ["q1"]}), encoding="utf-8")

    output = tmp_path / "report.json"
    pointer_file = artifact_root / "active_pointer.json"
    pointer_file.write_text(json.dumps({"generation": "gen-v1"}), encoding="utf-8")

    result = probe_rag_consistency.run_probe(
        baseline_generation="gen-v1",
        candidate_generation="gen-v2",
        artifact_root=artifact_root,
        golden_path=golden,
        output_path=output,
        _force_mismatch=True,
    )

    # Pointer should be restored to original
    restored = json.loads(pointer_file.read_text(encoding="utf-8"))
    assert restored["generation"] == "gen-v1"


def test_probe_validates_manifest_identity(tmp_path):
    """Probe validates manifest/component identity from artifact."""
    artifact_root = tmp_path / "artifacts"
    artifact_root.mkdir()
    candidate_dir = artifact_root / "candidate"
    candidate_dir.mkdir()
    _write_manifest(candidate_dir, "gen-v2", sha="b" * 64)

    golden = tmp_path / "golden.json"
    golden.write_text(json.dumps({"queries": ["q1"]}), encoding="utf-8")

    output = tmp_path / "report.json"

    result = probe_rag_consistency.run_probe(
        baseline_generation="gen-v1",
        candidate_generation="gen-v2",
        artifact_root=artifact_root,
        golden_path=golden,
        output_path=output,
    )

    assert result["manifest_digest"] is not None
    assert len(result["manifest_digest"]) == 64
