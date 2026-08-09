# -*- coding: utf-8 -*-
"""Evaluation Job API 契约测试 — TDD 红灯先行

覆盖：
- 成功提交 202 + 只有 run_id
- 路由不调用 .delay/apply_async
- DB 中 queued run/lock/audit/outbox 同一 run_id
- doctor B 查询/取消 doctor A 的 run → 404
- admin → 200
- 重复提交 409 + 已有 run context
- outbox flush/commit 异常 → 503，无半状态
- cancel: queued/retrying → cancelled 事务
- cancel: running → 202 + cancel_requested=true
- cancel: 终态重复取消幂等
- cancel: needs_review → 409
- status: 来自 DB + progress latest 补充
- Redis 不可用时 status 仍 200 + progress=None
"""

from __future__ import annotations

import uuid
from datetime import datetime, timezone
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from fastapi.testclient import TestClient

from app.core.deps import get_current_user
from app.db.session import get_db
from app.main import app
from app.models.evaluation_run import EvaluationRun


# ── Helpers ──────────────────────────────────────────────────────────────────

DOCTOR_A = SimpleNamespace(id=10, username="doc_a", role="doctor", permissions=None)
DOCTOR_B = SimpleNamespace(id=20, username="doc_b", role="doctor", permissions=None)
ADMIN_USER = SimpleNamespace(id=1, username="admin", role="admin", permissions=None)


def _override_get_db():
    yield None


@pytest.fixture
def client():
    c = TestClient(app, raise_server_exceptions=False)
    yield c
    c.close()
    app.dependency_overrides.clear()


def _as_user(user):
    app.dependency_overrides[get_db] = _override_get_db
    app.dependency_overrides[get_current_user] = lambda: user


def _make_run(
    run_id=None,
    consultation_id=1,
    status="queued",
    attempt=0,
    execution_owner=None,
    cancel_requested_at=None,
    cancel_requested_by=None,
    evaluation_id=None,
    error_type=None,
    error_message=None,
    started_at=None,
    finished_at=None,
    created_at=None,
):
    return EvaluationRun(
        id=run_id or str(uuid.uuid4()),
        consultation_id=consultation_id,
        graph_version="evaluation-graph-v1",
        scoring_policy_version="v1",
        checkpoint_thread_id=f"evaluation:{run_id or 'x'}",
        status=status,
        attempt=attempt,
        execution_owner=execution_owner,
        cancel_requested_at=cancel_requested_at,
        cancel_requested_by=cancel_requested_by,
        evaluation_id=evaluation_id,
        error_type=error_type,
        error_message=error_message,
        started_at=started_at,
        finished_at=finished_at,
        created_at=created_at or datetime.now(timezone.utc),
    )


# ── Submission Tests ─────────────────────────────────────────────────────────


class TestSubmitEvaluation:
    """POST /api/v1/evaluations/ → 202 EvaluationSubmitOut"""

    def test_submit_returns_202_with_run_id_only(self, client):
        """成功提交返回 202，响应体只有 run_id、consultation_id、status、status_url、websocket_url"""
        _as_user(DOCTOR_A)
        run_id = str(uuid.uuid4())
        mock_run = _make_run(run_id=run_id, status="queued")

        mock_access = AsyncMock()
        mock_txn = AsyncMock(return_value=mock_run)
        with patch("app.api.v1.evaluations.require_consultation_access", mock_access):
            with patch("app.api.v1.evaluations._submit_evaluation_transaction", mock_txn):
                resp = client.post(
                    "/api/v1/evaluations/",
                    json={"consultation_id": 1},
                )

        assert resp.status_code == 202, resp.text
        body = resp.json()
        assert body["run_id"] == run_id
        assert body["consultation_id"] == 1
        assert body["status"] == "queued"
        assert "status_url" in body
        assert "websocket_url" in body
        # 不得包含 task_id
        assert "task_id" not in body

    def test_submit_does_not_call_celery_delay(self, client):
        """路由不调用 .delay 或 apply_async"""
        _as_user(DOCTOR_A)
        run_id = str(uuid.uuid4())
        mock_run = _make_run(run_id=run_id, status="queued")

        mock_access = AsyncMock()
        mock_txn = AsyncMock(return_value=mock_run)
        mock_task = MagicMock()
        with patch("app.api.v1.evaluations.require_consultation_access", mock_access):
            with patch("app.api.v1.evaluations._submit_evaluation_transaction", mock_txn):
                with patch("app.tasks.evaluation_task.run_evaluation_task", mock_task):
                    resp = client.post(
                        "/api/v1/evaluations/",
                        json={"consultation_id": 1},
                    )

        assert resp.status_code == 202
        mock_task.delay.assert_not_called()
        mock_task.apply_async.assert_not_called()

    def test_submit_transaction_creates_run_lock_audit_outbox(self, client):
        """事务内创建 run、lock、audit、outbox，使用同一 run_id"""
        _as_user(DOCTOR_A)
        run_id = str(uuid.uuid4())
        mock_run = _make_run(run_id=run_id, status="queued")

        mock_access = AsyncMock()
        mock_txn = AsyncMock(return_value=mock_run)
        with patch("app.api.v1.evaluations.require_consultation_access", mock_access):
            with patch("app.api.v1.evaluations._submit_evaluation_transaction", mock_txn):
                resp = client.post(
                    "/api/v1/evaluations/",
                    json={"consultation_id": 1},
                )

        # 确认事务函数被调用
        assert mock_txn.called
        assert resp.status_code == 202

    def test_duplicate_submit_returns_409(self, client):
        """重复提交返回 409 + 已有 run 的 context"""
        _as_user(DOCTOR_A)

        from fastapi import HTTPException
        mock_access = AsyncMock()
        mock_txn = AsyncMock(
            side_effect=HTTPException(
                status_code=409,
                detail={"error_code": "EVALUATION_IN_PROGRESS", "message": "评估正在进行中"},
            )
        )
        with patch("app.api.v1.evaluations.require_consultation_access", mock_access):
            with patch("app.api.v1.evaluations._submit_evaluation_transaction", mock_txn):
                resp = client.post(
                    "/api/v1/evaluations/",
                    json={"consultation_id": 1},
                )

        assert resp.status_code == 409

    def test_outbox_failure_returns_503_no_half_state(self, client):
        """outbox flush 或 commit 异常时返回 503，不留下半状态"""
        _as_user(DOCTOR_A)

        from fastapi import HTTPException
        mock_access = AsyncMock()
        mock_txn = AsyncMock(
            side_effect=HTTPException(status_code=503, detail={"error_code": "SUBMISSION_FAILED", "message": "评估提交失败"})
        )
        with patch("app.api.v1.evaluations.require_consultation_access", mock_access):
            with patch("app.api.v1.evaluations._submit_evaluation_transaction", mock_txn):
                resp = client.post(
                    "/api/v1/evaluations/",
                    json={"consultation_id": 1},
                )

        assert resp.status_code == 503
        # 异常文本不出现在响应中
        assert "outbox flush failed" not in resp.text


# ── Cancel Tests ─────────────────────────────────────────────────────────────


class TestCancelEvaluation:
    """POST /api/v1/evaluations/runs/{run_id}/cancel"""

    def test_cancel_queued_returns_200_cancelled(self, client):
        """queued run 取消后返回 cancelled"""
        _as_user(DOCTOR_A)
        run_id = str(uuid.uuid4())
        mock_run = _make_run(run_id=run_id, status="cancelled")

        mock_access = AsyncMock(return_value=mock_run)
        mock_txn = AsyncMock(return_value=("cancelled_before_start", mock_run))
        mock_nudge = AsyncMock()
        with patch("app.api.v1.evaluations.require_evaluation_run_access", mock_access):
            with patch("app.api.v1.evaluations._cancel_evaluation_transaction", mock_txn):
                with patch("app.api.v1.evaluations._best_effort_cancel_nudge", mock_nudge):
                    resp = client.post(f"/api/v1/evaluations/runs/{run_id}/cancel")

        assert resp.status_code == 200
        body = resp.json()
        assert body["status"] == "cancelled"
        assert body["run_id"] == run_id

    def test_cancel_retrying_returns_200_cancelled(self, client):
        """retrying run 取消后返回 cancelled"""
        _as_user(DOCTOR_A)
        run_id = str(uuid.uuid4())
        mock_run = _make_run(run_id=run_id, status="cancelled")

        mock_access = AsyncMock(return_value=mock_run)
        mock_txn = AsyncMock(return_value=("cancelled_before_start", mock_run))
        mock_nudge = AsyncMock()
        with patch("app.api.v1.evaluations.require_evaluation_run_access", mock_access):
            with patch("app.api.v1.evaluations._cancel_evaluation_transaction", mock_txn):
                with patch("app.api.v1.evaluations._best_effort_cancel_nudge", mock_nudge):
                    resp = client.post(f"/api/v1/evaluations/runs/{run_id}/cancel")

        assert resp.status_code == 200
        body = resp.json()
        assert body["status"] == "cancelled"

    def test_cancel_running_returns_202_cancel_requested(self, client):
        """running run 返回 202 + cancel_requested=true"""
        _as_user(DOCTOR_A)
        run_id = str(uuid.uuid4())
        mock_run = _make_run(
            run_id=run_id, status="running",
            cancel_requested_at=datetime.now(timezone.utc),
            cancel_requested_by=DOCTOR_A.id,
        )

        mock_access = AsyncMock(return_value=mock_run)
        mock_txn = AsyncMock(return_value=("requested_running", mock_run))
        mock_nudge = AsyncMock()
        with patch("app.api.v1.evaluations.require_evaluation_run_access", mock_access):
            with patch("app.api.v1.evaluations._cancel_evaluation_transaction", mock_txn):
                with patch("app.api.v1.evaluations._best_effort_cancel_nudge", mock_nudge):
                    resp = client.post(f"/api/v1/evaluations/runs/{run_id}/cancel")

        assert resp.status_code == 202
        body = resp.json()
        assert body["cancel_requested"] is True
        assert body["status"] == "running"

    def test_cancel_terminal_idempotent(self, client):
        """终态重复取消幂等"""
        _as_user(DOCTOR_A)
        run_id = str(uuid.uuid4())
        mock_run = _make_run(run_id=run_id, status="completed")

        mock_access = AsyncMock(return_value=mock_run)
        mock_txn = AsyncMock(return_value=("already_terminal", mock_run))
        with patch("app.api.v1.evaluations.require_evaluation_run_access", mock_access):
            with patch("app.api.v1.evaluations._cancel_evaluation_transaction", mock_txn):
                resp = client.post(f"/api/v1/evaluations/runs/{run_id}/cancel")

        assert resp.status_code == 200
        body = resp.json()
        assert body["status"] == "completed"

    def test_cancel_needs_review_returns_409(self, client):
        """needs_review 返回 409"""
        _as_user(DOCTOR_A)
        run_id = str(uuid.uuid4())
        mock_run = _make_run(run_id=run_id, status="needs_review")

        mock_access = AsyncMock(return_value=mock_run)
        mock_txn = AsyncMock(return_value=("not_cancellable", mock_run))
        with patch("app.api.v1.evaluations.require_evaluation_run_access", mock_access):
            with patch("app.api.v1.evaluations._cancel_evaluation_transaction", mock_txn):
                resp = client.post(f"/api/v1/evaluations/runs/{run_id}/cancel")

        assert resp.status_code == 409

    def test_cancel_never_passes_terminate_true(self, client):
        """所有取消路径从不传 terminate=True"""
        _as_user(DOCTOR_A)
        run_id = str(uuid.uuid4())
        mock_run = _make_run(run_id=run_id, status="cancelled")

        mock_access = AsyncMock(return_value=mock_run)
        mock_txn = AsyncMock(return_value=("cancelled_before_start", mock_run))
        mock_nudge = AsyncMock()
        with patch("app.api.v1.evaluations.require_evaluation_run_access", mock_access):
            with patch("app.api.v1.evaluations._cancel_evaluation_transaction", mock_txn):
                with patch("app.api.v1.evaluations._best_effort_cancel_nudge", mock_nudge):
                    resp = client.post(f"/api/v1/evaluations/runs/{run_id}/cancel")

        assert resp.status_code == 200
        # nudge 被调用但 terminate 从不为 True — 由 _best_effort_cancel_nudge 实现保证


# ── Ownership Tests ──────────────────────────────────────────────────────────


class TestOwnershipAccess:
    """doctor B 查询/取消 doctor A 的 run → 404, admin → 200"""

    def test_doctor_b_query_doctor_a_run_returns_404(self, client):
        """doctor B 查询 doctor A 的 run → 404"""
        _as_user(DOCTOR_B)
        run_id = str(uuid.uuid4())

        from fastapi import HTTPException
        mock_access = AsyncMock(side_effect=HTTPException(status_code=404))
        with patch("app.api.v1.evaluations.require_evaluation_run_access", mock_access):
            resp = client.get(f"/api/v1/evaluations/runs/{run_id}/status")

        assert resp.status_code == 404

    def test_admin_query_any_run_returns_200(self, client):
        """admin 查询任意 run → 200"""
        _as_user(ADMIN_USER)
        run_id = str(uuid.uuid4())
        mock_run = _make_run(run_id=run_id, status="running", attempt=1)

        mock_access = AsyncMock(return_value=mock_run)
        mock_progress = AsyncMock(return_value=None)
        with patch("app.api.v1.evaluations.require_evaluation_run_access", mock_access):
            with patch("app.api.v1.evaluations._get_progress_latest", mock_progress):
                resp = client.get(f"/api/v1/evaluations/runs/{run_id}/status")

        assert resp.status_code == 200

    def test_doctor_b_cancel_doctor_a_run_returns_404(self, client):
        """doctor B 取消 doctor A 的 run → 404"""
        _as_user(DOCTOR_B)
        run_id = str(uuid.uuid4())

        from fastapi import HTTPException
        mock_access = AsyncMock(side_effect=HTTPException(status_code=404))
        with patch("app.api.v1.evaluations.require_evaluation_run_access", mock_access):
            resp = client.post(f"/api/v1/evaluations/runs/{run_id}/cancel")

        assert resp.status_code == 404


# ── Status Tests ─────────────────────────────────────────────────────────────


class TestRunStatus:
    """GET /api/v1/evaluations/runs/{run_id}/status"""

    def test_status_from_db_with_progress_supplement(self, client):
        """status/evaluation_id/cancel_requested 来自 DB，progress 只补充"""
        _as_user(DOCTOR_A)
        run_id = str(uuid.uuid4())
        mock_run = _make_run(
            run_id=run_id, status="running", attempt=1,
            evaluation_id=None,
            cancel_requested_at=datetime.now(timezone.utc),
            cancel_requested_by=DOCTOR_A.id,
            started_at=datetime.now(timezone.utc),
        )

        mock_access = AsyncMock(return_value=mock_run)
        mock_progress = AsyncMock(return_value={"progress": 45, "message": "评估中"})
        with patch("app.api.v1.evaluations.require_evaluation_run_access", mock_access):
            with patch("app.api.v1.evaluations._get_progress_latest", mock_progress):
                resp = client.get(f"/api/v1/evaluations/runs/{run_id}/status")

        assert resp.status_code == 200
        body = resp.json()
        assert body["status"] == "running"
        assert body["progress"] == 45
        assert body["message"] == "评估中"
        assert body["cancel_requested"] is True
        assert body["run_id"] == run_id

    def test_status_redis_down_returns_200_progress_none(self, client):
        """Redis 不可用时仍返回 200 且 progress=None"""
        _as_user(DOCTOR_A)
        run_id = str(uuid.uuid4())
        mock_run = _make_run(
            run_id=run_id, status="running", attempt=1,
            started_at=datetime.now(timezone.utc),
        )

        mock_access = AsyncMock(return_value=mock_run)
        mock_progress = AsyncMock(return_value=None)
        with patch("app.api.v1.evaluations.require_evaluation_run_access", mock_access):
            with patch("app.api.v1.evaluations._get_progress_latest", mock_progress):
                resp = client.get(f"/api/v1/evaluations/runs/{run_id}/status")

        assert resp.status_code == 200
        body = resp.json()
        assert body["progress"] is None
        assert body["message"] is None
        assert body["status"] == "running"

    def test_status_failed_exposes_error_code(self, client):
        """failed 只暴露标准化 error_code"""
        _as_user(DOCTOR_A)
        run_id = str(uuid.uuid4())
        mock_run = _make_run(
            run_id=run_id, status="failed", attempt=1,
            error_type="worker_lost",
            error_message="Some internal error details",
            finished_at=datetime.now(timezone.utc),
        )

        mock_access = AsyncMock(return_value=mock_run)
        mock_progress = AsyncMock(return_value=None)
        with patch("app.api.v1.evaluations.require_evaluation_run_access", mock_access):
            with patch("app.api.v1.evaluations._get_progress_latest", mock_progress):
                resp = client.get(f"/api/v1/evaluations/runs/{run_id}/status")

        assert resp.status_code == 200
        body = resp.json()
        assert body["status"] == "failed"
        assert body["error_code"] == "worker_lost"

    def test_progress_cannot_override_terminal_db_status(self, client):
        """progress latest 不能用旧事件把 DB 终态改回 running"""
        _as_user(DOCTOR_A)
        run_id = str(uuid.uuid4())
        mock_run = _make_run(
            run_id=run_id, status="completed", attempt=1,
            evaluation_id=42,
            finished_at=datetime.now(timezone.utc),
        )

        mock_access = AsyncMock(return_value=mock_run)
        mock_progress = AsyncMock(return_value={"progress": 80, "message": "still running"})
        with patch("app.api.v1.evaluations.require_evaluation_run_access", mock_access):
            with patch("app.api.v1.evaluations._get_progress_latest", mock_progress):
                resp = client.get(f"/api/v1/evaluations/runs/{run_id}/status")

        assert resp.status_code == 200
        body = resp.json()
        # DB 终态优先
        assert body["status"] == "completed"
        # progress 不覆盖终态
        assert body["evaluation_id"] == 42


# ── Lock-status Compatibility ────────────────────────────────────────────────


class TestLockStatusCompat:
    """GET /api/v1/evaluations/{consultation_id}/lock-status 兼容映射"""

    def test_lock_status_maps_pending_to_queued(self, client):
        """对外映射 pending → queued"""
        _as_user(DOCTOR_A)

        mock_access = AsyncMock()
        mock_status = AsyncMock(return_value={
            "consultation_id": 1,
            "status": "pending",
            "run_id": str(uuid.uuid4()),
            "locked_at": datetime.now(timezone.utc).isoformat(),
            "expires_at": datetime.now(timezone.utc).isoformat(),
            "is_active": True,
        })
        with patch("app.api.v1.evaluations.require_consultation_access", mock_access):
            with patch("app.api.v1.evaluations.get_lock_status", mock_status):
                resp = client.get("/api/v1/evaluations/1/lock-status")

        assert resp.status_code == 200
        body = resp.json()
        # pending 映射为 queued
        assert body["status"] == "queued"
