"""Tests for V1.2 agent runtime models."""

from app.models.agent_trace_event import AgentTraceEvent
from app.models.coach_decision import CoachDecision
from app.models.coach_session import CoachSession
from app.models.experiment_assignment import ExperimentAssignment
from app.models.prompt_bundle import PromptBundle
from app.models.trainee_memory import TraineeMemory


def test_coach_session_has_required_columns():
    """CoachSession has consultation_id unique and thread_id unique."""
    col_names = {c.name for c in CoachSession.__table__.columns}
    assert "id" in col_names
    assert "consultation_id" in col_names
    assert "doctor_id" in col_names
    assert "mode" in col_names
    assert "status" in col_names
    assert "thread_id" in col_names


def test_coach_decision_has_unique_turn():
    """CoachDecision has unique constraint on (session_id, turn_no)."""
    constraint_names = {c.name for c in CoachDecision.__table__.constraints}
    assert "uq_coach_decision_session_turn" in constraint_names


def test_trace_event_append_only():
    """AgentTraceEvent has unique (trace_id, sequence)."""
    constraint_names = {c.name for c in AgentTraceEvent.__table__.constraints}
    assert "uq_trace_event_trace_sequence" in constraint_names


def test_trace_payload_capture_defaults_off():
    """AgentTraceEvent input/output payload default to None."""
    event = AgentTraceEvent(
        event_type="node_finished",
        trace_id="trace-1",
        sequence=1,
        session_id="session-1",
    )
    assert event.input_payload is None
    assert event.output_payload is None


def test_trainee_memory_has_status_lifecycle():
    """TraineeMemory has status field with candidate/approved/rejected/expired."""
    col_names = {c.name for c in TraineeMemory.__table__.columns}
    assert "status" in col_names
    assert "skill_dimension" in col_names
    assert "summary" in col_names
    # No patient content column
    assert "patient_content" not in col_names
    assert "patient_name" not in col_names


def test_prompt_bundle_is_immutable():
    """PromptBundle has version and status fields."""
    col_names = {c.name for c in PromptBundle.__table__.columns}
    assert "version" in col_names
    assert "status" in col_names
    assert "source_commit" in col_names


def test_experiment_assignment_has_deterministic_fields():
    """ExperimentAssignment has experiment_id, subject_id, variant."""
    col_names = {c.name for c in ExperimentAssignment.__table__.columns}
    assert "experiment_id" in col_names
    assert "subject_id" in col_names
    assert "variant" in col_names
