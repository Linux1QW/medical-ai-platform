"""Tests for V1.1 error response contract."""
from app.schemas.common import ErrorOut


def test_error_out_preserves_context():
    error = ErrorOut(
        error_code="EVALUATION_IN_PROGRESS",
        message="评估正在进行中",
        detail="评估正在进行中，请勿重复提交",
        request_id="abc123",
        context={"run_id": "6e8d1a67-4a31-4b33-8507-9c8ac1bd3160", "status": "running"},
    )
    data = error.model_dump()
    assert data["context"]["run_id"] == "6e8d1a67-4a31-4b33-8507-9c8ac1bd3160"
    assert data["error_code"] == "EVALUATION_IN_PROGRESS"


def test_error_out_no_context_for_plain():
    error = ErrorOut(
        error_code="INTERNAL_ERROR", message="服务器内部错误",
        detail="some error", request_id="xyz789",
    )
    data = error.model_dump()
    assert data["context"] is None


def test_error_out_full_serialization():
    error = ErrorOut(
        error_code="NOT_FOUND", message="资源不存在",
        detail="评估报告不存在", request_id="req001",
        error_type="NotFoundError", context={"resource": "evaluation"},
    )
    json_str = error.model_dump_json()
    assert "NOT_FOUND" in json_str
    assert "resource" in json_str
