"""Tests for V1.1 evaluation job schema contracts."""
import pytest
from uuid import UUID, uuid4
from datetime import datetime, timezone


def test_evaluation_job_status_allows_valid_values():
    from app.schemas.evaluation import EvaluationJobStatus
    for s in ["queued", "running", "retrying", "completed",
              "needs_review", "reviewed", "failed", "cancelled"]:
        assert s in EvaluationJobStatus.__args__


def test_evaluation_submit_out_valid():
    from app.schemas.evaluation import EvaluationSubmitOut
    run_id = uuid4()
    obj = EvaluationSubmitOut(
        run_id=run_id, consultation_id=42, status="queued",
        status_url=f"/api/v1/evaluations/runs/{run_id}/status",
        websocket_url=f"/api/v1/evaluations/ws/runs/{run_id}",
    )
    assert obj.run_id == run_id
    assert obj.status == "queued"


def test_evaluation_submit_out_rejects_non_queued():
    from app.schemas.evaluation import EvaluationSubmitOut
    from pydantic import ValidationError
    with pytest.raises(ValidationError):
        EvaluationSubmitOut(
            run_id=uuid4(), consultation_id=1, status="running",
            status_url="/s", websocket_url="/w",
        )


def test_evaluation_submit_out_uuid_is_string_in_json():
    from app.schemas.evaluation import EvaluationSubmitOut
    run_id = uuid4()
    obj = EvaluationSubmitOut(
        run_id=run_id, consultation_id=1, status="queued",
        status_url="/s", websocket_url="/w",
    )
    json_data = obj.model_dump_json()
    assert str(run_id) in json_data


def test_run_status_out_valid():
    from app.schemas.evaluation import EvaluationRunStatusOut
    obj = EvaluationRunStatusOut(
        run_id=uuid4(), consultation_id=42, status="running",
        progress=45, message="评估中", attempt=1,
        submitted_at=datetime.now(timezone.utc),
    )
    assert obj.progress == 45
    assert obj.evaluation_id is None
    assert obj.cancel_requested is False


def test_run_status_out_rejects_progress_over_100():
    from app.schemas.evaluation import EvaluationRunStatusOut
    from pydantic import ValidationError
    with pytest.raises(ValidationError):
        EvaluationRunStatusOut(
            run_id=uuid4(), consultation_id=1, status="running",
            progress=101, attempt=0,
            submitted_at=datetime.now(timezone.utc),
        )


def test_run_status_out_rejects_invalid_status():
    from app.schemas.evaluation import EvaluationRunStatusOut
    from pydantic import ValidationError
    with pytest.raises(ValidationError):
        EvaluationRunStatusOut(
            run_id=uuid4(), consultation_id=1, status="pending_review",
            attempt=0, submitted_at=datetime.now(timezone.utc),
        )


def test_cancel_requested_not_a_status():
    from app.schemas.evaluation import EvaluationRunStatusOut
    obj = EvaluationRunStatusOut(
        run_id=uuid4(), consultation_id=1, status="running",
        cancel_requested=True, attempt=0,
        submitted_at=datetime.now(timezone.utc),
    )
    assert obj.cancel_requested is True
    assert obj.status == "running"


def test_datetime_normalized_to_utc():
    from app.schemas.evaluation import EvaluationRunStatusOut
    naive = datetime(2026, 8, 9, 10, 0, 0)
    obj = EvaluationRunStatusOut(
        run_id=uuid4(), consultation_id=1, status="queued",
        attempt=0, submitted_at=naive,
    )
    json_str = obj.model_dump_json()
    assert "Z" in json_str or "+00:00" in json_str


def test_cancel_out_valid():
    from app.schemas.evaluation import EvaluationCancelOut
    obj = EvaluationCancelOut(
        run_id=uuid4(), status="running", cancel_requested=True,
    )
    assert obj.cancel_requested is True
    assert obj.status == "running"
