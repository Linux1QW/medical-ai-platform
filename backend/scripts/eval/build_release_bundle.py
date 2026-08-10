"""Build an auditable RAG measured release bundle."""
from __future__ import annotations

import hashlib
import subprocess
import tarfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


def get_source_commit() -> str:
    """Get current HEAD commit SHA from git. Must not accept CLI overrides."""
    return subprocess.check_output(
        ["git", "rev-parse", "HEAD"], text=True
    ).strip()


def generate_index_generation() -> str:
    """Generate a unique index generation identifier."""
    now = datetime.now(timezone.utc)
    suffix = hashlib.sha256(now.isoformat().encode()).hexdigest()[:8]
    return f"rag-{now.strftime('%Y%m%d%H%M%S')}-{suffix}"


def sha256_file(path: Path) -> str:
    """Compute SHA-256 hex digest of a file."""
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def build_manifest(
    *,
    source_commit: str,
    index_generation: str,
    document_count: int,
    chunk_count: int,
    index_sha256: str,
) -> dict[str, Any]:
    """Build the index manifest."""
    return {
        "schema_version": 1,
        "source_commit": source_commit,
        "index_generation": index_generation,
        "document_count": document_count,
        "chunk_count": chunk_count,
        "index_sha256": index_sha256,
        "created_at": datetime.now(timezone.utc).isoformat(),
    }


def build_bundle(
    *,
    output_dir: Path,
    candidate_report: Path,
    baseline_report: Path,
    baseline_provenance: Path,
    index_manifest: Path,
    consistency_report: Path,
    rag_agent_report: Path,
) -> Path:
    """Build a release bundle tarball.

    source_commit is always read from git rev-parse HEAD.
    """
    source_commit = get_source_commit()
    bundle_name = f"rag-release-bundle-{source_commit[:12]}"
    bundle_dir = output_dir / bundle_name
    bundle_dir.mkdir(parents=True, exist_ok=True)

    members = {
        "candidate-report.json": candidate_report,
        "baseline-report.json": baseline_report,
        "baseline-provenance.json": baseline_provenance,
        "index-manifest.json": index_manifest,
        "consistency-report.json": consistency_report,
        "rag-agent-report.json": rag_agent_report,
    }

    for dest_name, src_path in members.items():
        content = Path(src_path).read_bytes()
        (bundle_dir / dest_name).write_bytes(content)

    tarball = output_dir / f"{bundle_name}.tar.gz"
    with tarfile.open(tarball, "w:gz") as tar:
        tar.add(bundle_dir, arcname=bundle_name)

    return tarball


if __name__ == "__main__":
    print("Usage: Import and call build_bundle() with appropriate paths")
    print("This script is designed to be called from CI workflow")
