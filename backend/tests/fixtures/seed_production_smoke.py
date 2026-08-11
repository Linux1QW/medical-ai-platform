# -*- coding: utf-8 -*-
"""Persist deterministic V1.1 browser-smoke fixtures in the isolated E2E DB.

Creates idempotently:
- admin_v11 and doctor_v11;
- patient id=2;
- ended consultation id=2 with ordered messages.

The command is intentionally refused unless ``ENVIRONMENT=test``.
"""
from __future__ import annotations

import hashlib
import json
import os
import sys
from datetime import datetime, timezone
from urllib.parse import quote_plus

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

E2E_ADMIN_USER = os.environ.get("E2E_ADMIN_USER", "admin_v11")
E2E_DOCTOR_USER = os.environ.get("E2E_DOCTOR_USER", "doctor_v11")
E2E_PASSWORD = os.environ.get("E2E_PASSWORD", "e2e_test_password_2026")
E2E_INDEX_VERSION = "e2e-low-evidence-v1"
FIXED_PATIENT_ID = 2
FIXED_CONSULTATION_ID = 2


def check_environment() -> None:
    if os.environ.get("ENVIRONMENT") != "test":
        raise RuntimeError("ENVIRONMENT must be 'test' for production-smoke seed")


def _get_sync_engine():
    host = os.environ.get("MYSQL_HOST", "localhost")
    port = os.environ.get("MYSQL_PORT", "3306")
    user = os.environ.get("MYSQL_USER", "root")
    password = os.environ.get("MYSQL_PASSWORD", "")
    database = os.environ.get("MYSQL_DATABASE", "medical_ai_e2e")
    url = (
        f"mysql+pymysql://{quote_plus(user)}:{quote_plus(password)}"
        f"@{host}:{port}/{quote_plus(database)}"
    )
    from sqlalchemy import create_engine

    return create_engine(url, pool_pre_ping=True)


def seed_users(session):
    from passlib.context import CryptContext
    from sqlalchemy import text

    password_hash = CryptContext(schemes=["bcrypt"], deprecated="auto").hash(
        E2E_PASSWORD
    )
    users = [
        {
            "username": E2E_ADMIN_USER,
            "email": f"{E2E_ADMIN_USER}@e2e.test",
            "real_name": "V1.1 E2E 管理员",
            "role": "admin",
            "department": "系统管理",
            "hashed_password": password_hash,
            "permissions": json.dumps(["review:manage", "evaluation:create"]),
        },
        {
            "username": E2E_DOCTOR_USER,
            "email": f"{E2E_DOCTOR_USER}@e2e.test",
            "real_name": "V1.1 E2E 测试医生",
            "role": "doctor",
            "department": "内科",
            "hashed_password": password_hash,
            "permissions": json.dumps(["evaluation:create"]),
        },
    ]
    statement = text(
        """
        INSERT INTO users
            (username, email, hashed_password, real_name, role, department, permissions)
        VALUES
            (:username, :email, :hashed_password, :real_name, :role, :department, :permissions)
        ON DUPLICATE KEY UPDATE
            hashed_password=VALUES(hashed_password), real_name=VALUES(real_name),
            role=VALUES(role), department=VALUES(department), permissions=VALUES(permissions)
        """
    )
    for user in users:
        session.execute(statement, user)

    doctor_id = session.execute(
        text("SELECT id FROM users WHERE username=:username"),
        {"username": E2E_DOCTOR_USER},
    ).scalar_one()
    admin_id = session.execute(
        text("SELECT id FROM users WHERE username=:username"),
        {"username": E2E_ADMIN_USER},
    ).scalar_one()
    return users, doctor_id, admin_id


def seed_patient_and_consultation(session, doctor_id: int):
    from sqlalchemy import text

    patient = {
        "id": FIXED_PATIENT_ID,
        "case_id": "e2e-v11-patient-001",
        "name": "V1.1 合成患者",
        "age": 50,
        "gender": "male",
        "personality_type": "配合型",
        "chief_complaint": "头痛伴低热三天",
        "medical_history": "无特殊病史",
        "symptoms": json.dumps(["头痛", "低热"], ensure_ascii=False),
        "system_prompt": "这是 V1.1 E2E 合成患者，不包含真实患者信息。",
        "expected_diagnosis": "普通感冒",
        "difficulty_level": 1,
    }
    consultation = {
        "id": FIXED_CONSULTATION_ID,
        "doctor_id": doctor_id,
        "patient_id": FIXED_PATIENT_ID,
        "status": "completed",
        "max_rounds": 20,
        "consultation_type": "initial",
    }
    messages = [
        {"role": "doctor", "content": "您好，请描述一下您的症状。", "sequence": 1},
        {"role": "patient", "content": "我头痛三天了，还伴有低热。", "sequence": 2},
        {"role": "doctor", "content": "头痛是持续性还是阵发性？", "sequence": 3},
        {"role": "patient", "content": "持续性的，没有恶心呕吐。", "sequence": 4},
        {"role": "doctor", "content": "建议先完善血常规检查。", "sequence": 5},
    ]
    session.execute(
        text(
            """
            INSERT INTO virtual_patients
                (id, case_id, name, age, gender, personality_type, chief_complaint,
                 medical_history, symptoms, expected_diagnosis, system_prompt, difficulty_level)
            VALUES
                (:id, :case_id, :name, :age, :gender, :personality_type, :chief_complaint,
                 :medical_history, :symptoms, :expected_diagnosis, :system_prompt, :difficulty_level)
            ON DUPLICATE KEY UPDATE
                name=VALUES(name), chief_complaint=VALUES(chief_complaint),
                medical_history=VALUES(medical_history), symptoms=VALUES(symptoms)
            """
        ),
        patient,
    )
    session.execute(
        text(
            """
            INSERT INTO consultations
                (id, doctor_id, patient_id, status, max_rounds, consultation_type)
            VALUES
                (:id, :doctor_id, :patient_id, :status, :max_rounds, :consultation_type)
            ON DUPLICATE KEY UPDATE
                doctor_id=VALUES(doctor_id), patient_id=VALUES(patient_id), status=VALUES(status)
            """
        ),
        consultation,
    )
    message_statement = text(
        """
        INSERT INTO consultation_messages (consultation_id, role, content, sequence)
        VALUES (:consultation_id, :role, :content, :sequence)
        ON DUPLICATE KEY UPDATE content=VALUES(content)
        """
    )
    for message in messages:
        session.execute(
            message_statement,
            {**message, "consultation_id": FIXED_CONSULTATION_ID},
        )
    return patient, consultation, messages


def seed_low_evidence_index() -> tuple[dict, dict]:
    chunk = {
        "id": "e2e-synthetic-chunk-001",
        "content": "V1.1 E2E synthetic low-evidence fixture.",
        "metadata": {
            "source": "e2e-synthetic-test",
            "is_synthetic": True,
            "version": E2E_INDEX_VERSION,
            "created_at": datetime.now(timezone.utc).isoformat(),
        },
    }
    checksum = hashlib.sha256(
        json.dumps(chunk, sort_keys=True).encode("utf-8")
    ).hexdigest()
    manifest = {
        "version": E2E_INDEX_VERSION,
        "schema_version": "1.0",
        "candidate_count": 1,
        "source_count": 1,
        "chunks": [{"id": chunk["id"], "checksum": checksum}],
    }
    return chunk, manifest


def main() -> dict:
    check_environment()
    engine = _get_sync_engine()
    with engine.connect() as connection:
        try:
            users, doctor_id, admin_id = seed_users(connection)
            patient, consultation, messages = seed_patient_and_consultation(
                connection, doctor_id
            )
            connection.commit()
        except Exception:
            connection.rollback()
            raise
    chunk, manifest = seed_low_evidence_index()
    assert len(messages) >= 4
    engine.dispose()
    print(
        "V1.1 production-smoke seed persisted: "
        f"doctor_id={doctor_id}, admin_id={admin_id}, consultation_id={FIXED_CONSULTATION_ID}"
    )
    return {
        "users": users,
        "patient": patient,
        "consultation": consultation,
        "messages": messages,
        "chunk": chunk,
        "manifest": manifest,
        "doctor_id": doctor_id,
        "admin_id": admin_id,
        "patient_id": FIXED_PATIENT_ID,
        "consultation_id": FIXED_CONSULTATION_ID,
    }


if __name__ == "__main__":
    main()
