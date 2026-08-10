"""Verify that all GitHub Actions workflow steps pin actions to immutable 40-char SHAs."""
from __future__ import annotations

import re
import sys
from pathlib import Path

USES_RE = re.compile(r"^\s*-\s+uses:\s*([^\s#]+)", re.MULTILINE)
SHA_RE = re.compile(r"^[0-9a-f]{40}$")


def verify_workflow_pins(workflow_root: Path) -> list[str]:
    """Return a list of error strings for every non-pinned or non-SHA action reference."""
    errors: list[str] = []
    root = Path(workflow_root)
    for path in sorted(root.glob("*.y*ml")):
        for ref in USES_RE.findall(path.read_text(encoding="utf-8")):
            if ref.startswith("./") or ref.startswith("docker://"):
                continue
            action, separator, revision = ref.rpartition("@")
            if not separator or not action or not SHA_RE.fullmatch(revision):
                errors.append(f"{path}:{ref}")
    return errors


if __name__ == "__main__":
    target = Path(sys.argv[1]) if len(sys.argv) > 1 else Path(".github/workflows")
    findings = verify_workflow_pins(target)
    if findings:
        for err in findings:
            print(f"ERROR: {err}")
        sys.exit(1)
    print("all workflow actions are pinned")
