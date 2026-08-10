"""Tests for CoachContextBuilder — source-bound context construction."""
from datetime import datetime, timedelta
from unittest.mock import AsyncMock, MagicMock

import pytest

from app.models.consultation import Consultation, ConsultationMessage
from app.models.patient import VirtualPatient
from app.models.trainee_memory import TraineeMemory, TraineeMemoryConsent
from app.services.coach_context_builder import CoachContextBuilder


def _make_mock_db(
    patient: VirtualPatient | None = None,
    messages: list[ConsultationMessage] | None = None,
    consent: TraineeMemoryConsent | None = None,
    memories: list[TraineeMemory] | None = None,
) -> AsyncMock:
    """Build a mock AsyncSession that returns the given ORM objects."""
    db = AsyncMock()

    def _execute_side_effect(query):
        result_mock = MagicMock()
        query_str = str(query)

        if "virtual_patients" in query_str:
            result_mock.scalar_one_or_none.return_value = patient
            return result_mock
        elif "consultation_messages" in query_str:
            scalars_mock = MagicMock()
            scalars_mock.all.return_value = messages or []
            result_mock.scalars.return_value = scalars_mock
            return result_mock
        elif "trainee_memory_consents" in query_str:
            result_mock.scalar_one_or_none.return_value = consent
            return result_mock
        elif "trainee_memories" in query_str:
            scalars_mock = MagicMock()
            scalars_mock.all.return_value = memories or []
            result_mock.scalars.return_value = scalars_mock
            return result_mock

        result_mock.scalar_one_or_none.return_value = None
        scalars_mock = MagicMock()
        scalars_mock.all.return_value = []
        result_mock.scalars.return_value = scalars_mock
        return result_mock

    db.execute = AsyncMock(side_effect=_execute_side_effect)
    return db


def _make_patient(**overrides) -> VirtualPatient:
    defaults = dict(
        name="测试患者",
        age=45,
        gender="male",
        personality_type="配合型",
        chief_complaint="反复胸闷3天",
        medical_history="高血压病史5年",
        symptoms='{"chest_tightness": "mild"}',
        expected_diagnosis="冠心病",
        system_prompt="你是一个45岁男性患者...",
    )
    defaults.update(overrides)
    patient = VirtualPatient(**defaults)
    patient.id = 1
    return patient


def _make_consultation(patient_id: int = 1, doctor_id: int = 1) -> Consultation:
    c = Consultation(doctor_id=doctor_id, patient_id=patient_id, status="in_progress")
    c.id = 100
    return c


def _make_messages(consultation_id: int = 100) -> list[ConsultationMessage]:
    msgs = []
    for i, (role, content) in enumerate(
        [
            ("patient", "我最近三天总是觉得胸闷"),
            ("doctor", "胸闷是什么情况下会出现？"),
            ("patient", "走路快了或者爬楼梯的时候比较明显"),
        ],
        start=1,
    ):
        m = ConsultationMessage(
            consultation_id=consultation_id, role=role, content=content, sequence=i
        )
        m.id = i
        msgs.append(m)
    return msgs


@pytest.mark.asyncio
async def test_builder_contains_visible_messages_but_not_gold():
    """Builder includes doctor/patient messages but never gold labels."""
    patient = _make_patient()
    messages = _make_messages()
    consultation = _make_consultation()

    db = _make_mock_db(patient=patient, messages=messages, consent=None, memories=[])

    view = await CoachContextBuilder().build(db, consultation, consultation.doctor_id)
    payload = view.model_dump_json()

    assert "doctor" in payload
    assert "patient" in payload
    assert "胸闷" in payload

    assert "expected_diagnosis" not in payload
    assert "system_prompt" not in payload
    assert "冠心病" not in payload


@pytest.mark.asyncio
async def test_builder_patient_profile_only_public_fields():
    """Patient profile contains only age, gender, chief_complaint."""
    patient = _make_patient()
    consultation = _make_consultation()
    db = _make_mock_db(patient=patient, messages=[], consent=None)

    view = await CoachContextBuilder().build(db, consultation, consultation.doctor_id)

    assert view.visible_patient.age == 45
    assert view.visible_patient.gender == "male"
    assert "胸闷" in view.visible_patient.chief_complaint

    payload = view.visible_patient.model_dump_json()
    assert "测试患者" not in payload
    assert "冠心病" not in payload


@pytest.mark.asyncio
async def test_builder_messages_ordered_by_sequence():
    """Messages are ordered by sequence number."""
    patient = _make_patient()
    messages = _make_messages()
    consultation = _make_consultation()
    db = _make_mock_db(patient=patient, messages=messages, consent=None)

    view = await CoachContextBuilder().build(db, consultation, consultation.doctor_id)

    sequences = [m.sequence for m in view.messages]
    assert sequences == [1, 2, 3]
    assert view.messages[0].role == "patient"
    assert view.messages[1].role == "doctor"


@pytest.mark.asyncio
async def test_builder_no_memories_without_consent():
    """No memories returned when doctor has not granted consent."""
    patient = _make_patient()
    consultation = _make_consultation()

    mem = TraineeMemory(
        doctor_id=1,
        status="approved",
        skill_dimension="问诊技巧",
        summary="善于引导患者描述症状",
    )
    mem.id = 1

    db = _make_mock_db(patient=patient, messages=[], consent=None, memories=[mem])

    view = await CoachContextBuilder().build(db, consultation, consultation.doctor_id)
    assert view.approved_profile_memories == []


@pytest.mark.asyncio
async def test_builder_approved_memories_with_consent():
    """Approved, consented, non-expired memories are included (max 5).

    Note: The SQL .limit(5) is enforced by the database. The mock simulates
    what the DB query would return — at most 5 memories.
    """
    patient = _make_patient()
    consultation = _make_consultation()

    consent = TraineeMemoryConsent(
        doctor_id=1, granted=1, granted_at=datetime.utcnow()
    )
    consent.id = 1

    # Simulate DB returning at most 5 (the limit is in the query)
    memories = []
    for i in range(5):
        mem = TraineeMemory(
            doctor_id=1,
            status="approved",
            skill_dimension=f"维度{i}",
            summary=f"技能摘要{i}",
            updated_at=datetime.utcnow() - timedelta(hours=i),
        )
        mem.id = i + 1
        memories.append(mem)

    db = _make_mock_db(
        patient=patient, messages=[], consent=consent, memories=memories
    )

    view = await CoachContextBuilder().build(db, consultation, consultation.doctor_id)
    assert len(view.approved_profile_memories) == 5
    for m in view.approved_profile_memories:
        assert m.memory_id is not None


@pytest.mark.asyncio
async def test_builder_expired_memories_excluded():
    """Expired memories are excluded even if approved and consented."""
    patient = _make_patient()
    consultation = _make_consultation()

    consent = TraineeMemoryConsent(
        doctor_id=1, granted=1, granted_at=datetime.utcnow()
    )
    consent.id = 1

    # All memories are expired → mock returns empty list
    db = _make_mock_db(
        patient=patient, messages=[], consent=consent, memories=[]
    )

    view = await CoachContextBuilder().build(db, consultation, consultation.doctor_id)
    assert len(view.approved_profile_memories) == 0


@pytest.mark.asyncio
async def test_builder_no_orm_objects_in_view():
    """CoachContextView contains no ORM objects — all are Pydantic models."""
    patient = _make_patient()
    messages = _make_messages()
    consultation = _make_consultation()
    db = _make_mock_db(patient=patient, messages=messages, consent=None)

    view = await CoachContextBuilder().build(db, consultation, consultation.doctor_id)

    assert not hasattr(view.visible_patient, "_sa_instance_state")
    for msg in view.messages:
        assert not hasattr(msg, "_sa_instance_state")
    for mem in view.approved_profile_memories:
        assert not hasattr(mem, "_sa_instance_state")
