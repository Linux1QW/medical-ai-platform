"""Tests for RAG release bundle verification."""
import json
import sys
from pathlib import Path

import pytest

# Add scripts/eval to path for import
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "scripts" / "eval"))

from verify_release_bundle import REQUIRED_MEMBERS, verify_bundle


@pytest.fixture
def bundle_dir(tmp_path):
    """Create a valid bundle directory for testing."""
    for member in REQUIRED_MEMBERS:
        (tmp_path / member).write_text(json.dumps({"test": True}))
    return tmp_path


def test_valid_bundle_passes(bundle_dir):
    result = verify_bundle(bundle_dir)
    assert result["passed"] is True


@pytest.mark.parametrize("missing", REQUIRED_MEMBERS)
def test_bundle_rejects_missing_member(bundle_dir, missing):
    (bundle_dir / missing).unlink()
    result = verify_bundle(bundle_dir)
    assert result["passed"] is False
    assert missing in result["errors"]
