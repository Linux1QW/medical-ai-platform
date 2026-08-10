# -*- coding: utf-8 -*-
"""E2E seed data for V1.2 Coach flow tests.

Idempotently creates:
- admin_v12 / doctor_v12 users with coach:use and coach:trace:view permissions
- A test patient with realistic chief complaint
- An active consultation owned by doctor_v12 with ≥4 ordered messages
- Low-evidence index fixture (ACTIVE_INDEX_VERSION=e2e-coach-v12)

Usage:
    python -m tests.fixtures.seed_v12_coach_e2e

Only runs in ENVIRONMENT=test against the isolated E2E database.
"""

import hashlib
import json
import os
import sys
from datetime import datetime, timezone

# Ensure app modules are importable
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))


# ──────────────────────────────────────────
# Configuration
# ──────────────────────────────────────────
E2E_ADMIN_USER = os.environ.get("E2E_ADMIN_USER", "admin_v12")
E2E_DOCTOR_USER = os.environ.get("E2E_DOCTOR_USER", "doctor_v12")
E2E_PASSWORD = os.environ.get("E2E_PASSWORD", "e2e_test_password_2026")
E2E_INDEX_VERSION = "e2e-coach-v12"


def check_environment():
    """Ensure running in test environment."""
    env = os.environ.get("ENVIRONMENT", "")
    if env != "test":
        print(f"ERROR: ENVIRONMENT must be 'test', got '{env}'")
        print("Set ENVIRONMENT=test before running this script.")
        sys.exit(1)


# ──────────────────────────────────────────
# Users
# ──────────────────────────────────────────
def seed_users():
    """Idempotently create admin_v12 and doctor_v12 accounts.

    doctor_v12 has coach:use permission.
    admin_v12 has coach:trace:view permission.
    """
    from passlib.context import CryptContext

    pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")
    password_hash = pwd_context.hash(E2E_PASSWORD)

    users = [
        {
            "username": E2E_ADMIN_USER,
            "email": f"{E2E_ADMIN_USER}@e2e.test",
            "real_name": "V1.2 E2E 管理员",
            "role": "admin",
            "department": "系统管理",
            "password_hash": password_hash,
            "permissions": ["coach:trace:view", "coach:use"],
        },
        {
            "username": E2E_DOCTOR_USER,
            "email": f"{E2E_DOCTOR_USER}@e2e.test",
            "real_name": "V1.2 E2E 测试医生",
            "role": "doctor",
            "department": "内科",
            "password_hash": password_hash,
            "permissions": ["coach:use"],
        },
    ]

    print(f"[seed] Users: {E2E_ADMIN_USER} (admin), {E2E_DOCTOR_USER} (doctor)")
    return users


# ──────────────────────────────────────────
# Patient and consultation
# ──────────────────────────────────────────
def seed_patient_and_consultation():
    """Create a test patient and an active consultation with ordered messages.

    The consultation is in 'ongoing' status so Coach can operate on it.
    """
    patient = {
        "name": "V1.2合成患者",
        "age": 45,
        "gender": "男",
        "personality_type": "配合型",
        "chief_complaint": "反复头痛伴乏力一周",
        "medical_history": "高血压病史2年，规律服药中",
        "symptoms": json.dumps(["头痛", "乏力", "偶有头晕"]),
        "system_prompt": "这是V1.2 E2E测试合成患者，非真实患者信息。请模拟头痛伴乏力的典型问诊过程。",
        "expected_diagnosis": "高血压控制不佳伴紧张性头痛",
    }

    consultation = {
        "status": "ongoing",
        "max_rounds": 20,
    }

    messages = [
        {"role": "doctor", "content": "您好，请问今天来就诊主要是什么问题？", "sequence": 1},
        {"role": "patient", "content": "我最近一周总是头痛，而且感觉很乏力。", "sequence": 2},
        {"role": "doctor", "content": "头痛是哪个部位？是持续性的还是一阵一阵的？", "sequence": 3},
        {"role": "patient", "content": "主要是后脑勺和两侧，持续性的钝痛，休息后会好一点。", "sequence": 4},
        {"role": "doctor", "content": "您的血压最近有监测吗？降压药还在吃吗？", "sequence": 5},
        {"role": "patient", "content": "药在吃，但是最近血压计测出来偏高一些，大概150/95左右。", "sequence": 6},
    ]

    print(f"[seed] Patient: {patient['name']}")
    print(f"[seed] Consultation: status={consultation['status']}, messages={len(messages)}")
    return patient, consultation, messages


# ──────────────────────────────────────────
# Low-evidence index fixture
# ──────────────────────────────────────────
def seed_low_evidence_index():
    """Idempotently create a low-evidence index fixture for Coach testing.

    Contains a single synthetic chunk clearly marked as test data.
    """
    synthetic_chunk = {
        "id": "e2e-coach-v12-chunk-001",
        "content": (
            "本段落为V1.2 Coach E2E合成测试内容。"
            "高血压患者头痛应首先评估血压控制情况。"
            "此chunk为合成测试数据，不包含真实医学指南内容。"
        ),
        "metadata": {
            "source": "e2e-synthetic-test-v12",
            "is_synthetic": True,
            "version": E2E_INDEX_VERSION,
            "created_at": datetime.now(timezone.utc).isoformat(),
        },
    }

    chunk_json = json.dumps(synthetic_chunk, sort_keys=True, ensure_ascii=False)
    checksum = hashlib.sha256(chunk_json.encode()).hexdigest()

    manifest = {
        "version": E2E_INDEX_VERSION,
        "schema_version": "1.0",
        "candidate_count": 1,
        "source_count": 1,
        "chunks": [
            {
                "id": synthetic_chunk["id"],
                "checksum": checksum,
                "source": synthetic_chunk["metadata"]["source"],
            }
        ],
        "created_at": datetime.now(timezone.utc).isoformat(),
    }

    print(f"[seed] Low-evidence index: {E2E_INDEX_VERSION}")
    print(f"[seed]   candidate_count={manifest['candidate_count']}")
    return synthetic_chunk, manifest


# ──────────────────────────────────────────
# Main
# ──────────────────────────────────────────
def main():
    """Main entry: idempotently seed all V1.2 Coach E2E test data."""
    check_environment()

    print("=" * 60)
    print("V1.2 Coach E2E Seed")
    print(f"ENVIRONMENT: {os.environ.get('ENVIRONMENT')}")
    print(f"INDEX_VERSION: {E2E_INDEX_VERSION}")
    print("=" * 60)

    # 1. Users
    users = seed_users()

    # 2. Patient and consultation
    patient, consultation, messages = seed_patient_and_consultation()

    # 3. Low-evidence index
    chunk, manifest = seed_low_evidence_index()

    # Assertions
    assert manifest["candidate_count"] == 1
    assert manifest["source_count"] == 1
    assert len(messages) >= 4
    assert consultation["status"] == "ongoing"

    print("=" * 60)
    print("V1.2 Coach E2E Seed complete. All assertions passed.")
    print("=" * 60)

    return {
        "users": users,
        "patient": patient,
        "consultation": consultation,
        "messages": messages,
        "manifest": manifest,
    }


if __name__ == "__main__":
    main()
