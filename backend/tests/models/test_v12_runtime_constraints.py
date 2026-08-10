"""Tests for V1.2 runtime model constraints."""

from app.models.coach_decision import CoachDecision
from app.models.coach_session import CoachSession
from app.models.coach_stream_event import CoachStreamEvent
from app.models.experiment_assignment import ExperimentAssignment
from app.models.prompt_bundle import PromptBundle
from app.models.trainee_memory import TraineeMemory, TraineeMemoryConsent


def test_coach_session_has_public_id():
    cols = {c.name for c in CoachSession.__table__.columns}
    assert "public_id" in cols


def test_coach_session_public_id_is_unique():
    col = CoachSession.__table__.c.public_id
    assert col.unique


def test_coach_session_mode_check_constraint():
    names = {c.name for c in CoachSession.__table__.constraints}
    assert "ck_coach_session_mode" in names


def test_coach_session_status_check_constraint():
    names = {c.name for c in CoachSession.__table__.constraints}
    assert "ck_coach_session_status" in names


def test_decision_has_suggestion_id():
    cols = {c.name for c in CoachDecision.__table__.columns}
    assert "suggestion_id" in cols


def test_decision_suggestion_id_is_unique():
    col = CoachDecision.__table__.c.suggestion_id
    assert col.unique


def test_decision_idempotency_is_unique():
    names = {c.name for c in CoachDecision.__table__.constraints}
    assert "uq_coach_decision_session_idempotency" in names


def test_decision_risk_level_check_constraint():
    names = {c.name for c in CoachDecision.__table__.constraints}
    assert "ck_coach_decision_risk_level" in names


def test_coach_stream_event_table_exists():
    assert CoachStreamEvent.__tablename__ == "coach_stream_events"


def test_coach_stream_event_has_required_columns():
    cols = {c.name for c in CoachStreamEvent.__table__.columns}
    expected = {"id", "event_id", "session_id", "sequence", "event_type", "data_json", "expires_at"}
    assert expected.issubset(cols)


def test_coach_stream_event_session_sequence_unique():
    names = {c.name for c in CoachStreamEvent.__table__.constraints}
    assert "uq_coach_stream_session_sequence" in names


def test_coach_stream_event_has_fk_to_coach_sessions():
    fks = CoachStreamEvent.__table__.c.session_id.foreign_keys
    assert len(fks) == 1
    fk = next(iter(fks))
    assert fk.column.table.name == "coach_sessions"


def test_prompt_bundle_version_is_unique():
    names = {c.name for c in PromptBundle.__table__.constraints}
    assert "uq_prompt_bundle_name_version" in names


def test_prompt_bundle_has_content_hash():
    cols = {c.name for c in PromptBundle.__table__.columns}
    assert "content_hash" in cols


def test_trainee_memory_doctor_id_has_fk():
    fks = TraineeMemory.__table__.c.doctor_id.foreign_keys
    assert len(fks) == 1
    fk = next(iter(fks))
    assert fk.column.table.name == "users"


def test_trainee_memory_reviewer_id_has_fk():
    fks = TraineeMemory.__table__.c.reviewer_id.foreign_keys
    assert len(fks) == 1
    fk = next(iter(fks))
    assert fk.column.table.name == "users"


def test_trainee_memory_status_check_constraint():
    names = {c.name for c in TraineeMemory.__table__.constraints}
    assert "ck_trainee_memory_status" in names


def test_trainee_memory_has_compound_index():
    idx_names = {idx.name for idx in TraineeMemory.__table__.indexes}
    assert "ix_trainee_memory_doctor_status_expires" in idx_names


def test_trainee_memory_consent_table_exists():
    assert TraineeMemoryConsent.__tablename__ == "trainee_memory_consents"


def test_trainee_memory_consent_has_doctor_fk():
    fks = TraineeMemoryConsent.__table__.c.doctor_id.foreign_keys
    assert len(fks) == 1
    fk = next(iter(fks))
    assert fk.column.table.name == "users"


def test_experiment_assignment_has_stage():
    cols = {c.name for c in ExperimentAssignment.__table__.columns}
    assert "stage" in cols


def test_experiment_assignment_has_rollout_pct():
    cols = {c.name for c in ExperimentAssignment.__table__.columns}
    assert "rollout_pct" in cols


def test_experiment_stage_check_constraint():
    names = {c.name for c in ExperimentAssignment.__table__.constraints}
    assert "ck_experiment_stage" in names


def test_experiment_rollout_pct_check_constraint():
    names = {c.name for c in ExperimentAssignment.__table__.constraints}
    assert "ck_experiment_rollout_pct" in names
