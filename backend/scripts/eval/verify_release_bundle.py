"""Verify RAG release bundle completeness and integrity."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

REQUIRED_MEMBERS = [
    "candidate-report.json",
    "baseline-report.json",
    "baseline-provenance.json",
    "index-manifest.json",
    "consistency-report.json",
    "rag-agent-report.json",
]


def verify_bundle(bundle_dir: Path) -> dict[str, Any]:
    """Verify that a release bundle contains all required members.

    Returns a dict with 'passed' (bool) and 'errors' (list[str]).
    """
    bundle = Path(bundle_dir)
    errors: list[str] = []

    for member in REQUIRED_MEMBERS:
        member_path = bundle / member
        if not member_path.exists():
            errors.append(member)
        else:
            try:
                json.loads(member_path.read_text(encoding="utf-8"))
            except json.JSONDecodeError as e:
                errors.append(f"{member}: invalid JSON ({e})")

    return {
        "passed": len(errors) == 0,
        "errors": errors,
        "members_checked": list(REQUIRED_MEMBERS),
    }


if __name__ == "__main__":
    import sys

    if len(sys.argv) < 2:
        print("Usage: verify_release_bundle.py <bundle_dir>")
        sys.exit(1)

    result = verify_bundle(Path(sys.argv[1]))
    if result["passed"]:
        print(f"Bundle valid: {len(result['members_checked'])} members checked")
        sys.exit(0)
    else:
        print(f"Bundle invalid: missing/invalid members: {result['errors']}")
        sys.exit(1)
