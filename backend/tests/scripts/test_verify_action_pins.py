"""Tests for verify_action_pins."""
import pytest
from pathlib import Path

from scripts.ci.verify_action_pins import verify_workflow_pins


def test_rejects_mutable_and_nonexistent_shape(tmp_path):
    workflow = tmp_path / "bad.yml"
    workflow.write_text(
        "jobs:\n  test:\n    steps:\n      - uses: actions/checkout@v4\n"
        "      - uses: actions/setup-node@abc123\n",
        encoding="utf-8",
    )
    errors = verify_workflow_pins(tmp_path)
    assert len(errors) == 2


def test_accepts_local_action_and_full_sha(tmp_path):
    workflow = tmp_path / "good.yml"
    workflow.write_text(
        "jobs:\n  test:\n    steps:\n"
        "      - uses: ./github/actions/local\n"
        "      - uses: actions/checkout@fbc6f3992d24b796d5a048ff273f7fcc4a7b6c09\n",
        encoding="utf-8",
    )
    assert verify_workflow_pins(tmp_path) == []
