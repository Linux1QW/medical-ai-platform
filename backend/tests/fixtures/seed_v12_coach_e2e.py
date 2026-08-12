# -*- coding: utf-8 -*-
"""E2E seed data for V1.2 Coach flow tests.

Idempotently creates (with real DB persistence):
- admin_v12 / doctor_v12 users with coach:use and coach:trace:view permissions
- A test patient with realistic chief complaint (fixed id=1)
- An active consultation owned by doctor_v12 with ≥6 ordered messages (fixed id=1)
- Low-evidence index fixture (ACTIVE_INDEX_VERSION=e2e-coach-v12)

Usage:
    python -m tests.fixtures.seed_v12_coach_e2e

Only runs in ENVIRONMENT=test against the isolated E2E database.
"""

import hashlib
import json
import math
import os
import sys
from datetime import datetime, timezone
from urllib.parse import quote_plus

# Ensure app modules are importable
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))


# ──────────────────────────────────────────
# Configuration
# ──────────────────────────────────────────
E2E_ADMIN_USER = os.environ.get("E2E_ADMIN_USER", "admin_v12")
E2E_DOCTOR_USER = os.environ.get("E2E_DOCTOR_USER", "doctor_v12")
E2E_PASSWORD = os.environ.get("E2E_PASSWORD", "e2e_test_password_2026")
E2E_INDEX_VERSION = "rag-20260812000000-d4465500"

# Fixed IDs for deterministic E2E tests
FIXED_PATIENT_ID = 1
FIXED_CONSULTATION_ID = 1
FIXED_DOCTOR_ID = None  # resolved after user creation
FIXED_ADMIN_ID = None   # resolved after user creation


def check_environment():
    """Ensure running in test environment."""
    env = os.environ.get("ENVIRONMENT", "")
    if env != "test":
        print(f"ERROR: ENVIRONMENT must be 'test', got '{env}'")
        print("Set ENVIRONMENT=test before running this script.")
        sys.exit(1)


def _get_sync_engine():
    """Build a synchronous SQLAlchemy engine from env vars."""
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


# ──────────────────────────────────────────
# Users
# ──────────────────────────────────────────
def seed_users(session):
    """Idempotently create admin_v12 and doctor_v12 accounts.

    doctor_v12 has coach:use permission.
    admin_v12 has coach:trace:view permission.
    """
    from passlib.context import CryptContext
    from sqlalchemy import text

    pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")
    password_hash = pwd_context.hash(E2E_PASSWORD)

    users = [
        {
            "username": E2E_ADMIN_USER,
            "email": f"{E2E_ADMIN_USER}@e2e.test",
            "real_name": "V1.2 E2E 管理员",
            "role": "admin",
            "department": "系统管理",
            "hashed_password": password_hash,
            "permissions": json.dumps(["coach:trace:view", "coach:use"]),
        },
        {
            "username": E2E_DOCTOR_USER,
            "email": f"{E2E_DOCTOR_USER}@e2e.test",
            "real_name": "V1.2 E2E 测试医生",
            "role": "doctor",
            "department": "内科",
            "hashed_password": password_hash,
            "permissions": json.dumps(["coach:use"]),
        },
    ]

    for u in users:
        session.execute(text(
            """
            INSERT INTO users (username, email, hashed_password, real_name, role, department, permissions)
            VALUES (:username, :email, :hashed_password, :real_name, :role, :department, :permissions)
            ON DUPLICATE KEY UPDATE
                real_name=VALUES(real_name),
                role=VALUES(role),
                department=VALUES(department),
                permissions=VALUES(permissions)
            """
        ), u)

    # Resolve fixed IDs
    global FIXED_DOCTOR_ID, FIXED_ADMIN_ID
    row = session.execute(
        text("SELECT id FROM users WHERE username = :u"), {"u": E2E_DOCTOR_USER}
    ).fetchone()
    FIXED_DOCTOR_ID = row[0]

    row = session.execute(
        text("SELECT id FROM users WHERE username = :u"), {"u": E2E_ADMIN_USER}
    ).fetchone()
    FIXED_ADMIN_ID = row[0]

    print(f"[seed] Users: {E2E_ADMIN_USER} (id={FIXED_ADMIN_ID}, admin), "
          f"{E2E_DOCTOR_USER} (id={FIXED_DOCTOR_ID}, doctor)")
    return users


# ──────────────────────────────────────────
# Patient and consultation
# ──────────────────────────────────────────
def seed_patient_and_consultation(session):
    """Create a test patient and an active consultation with ordered messages.

    The consultation is in 'in_progress' status so Coach can operate on it.
    Uses fixed IDs for deterministic E2E tests.
    """
    from sqlalchemy import text

    patient = {
        "id": FIXED_PATIENT_ID,
        "case_id": "e2e-v12-patient-001",
        "name": "V1.2合成患者",
        "age": 45,
        "gender": "male",
        "personality_type": "配合型",
        "chief_complaint": "反复头痛伴乏力一周",
        "medical_history": "高血压病史2年，规律服药中",
        "symptoms": json.dumps(["头痛", "乏力", "偶有头晕"], ensure_ascii=False),
        "system_prompt": "这是V1.2 E2E测试合成患者，非真实患者信息。请模拟头痛伴乏力的典型问诊过程。",
        "expected_diagnosis": "高血压控制不佳伴紧张性头痛",
        "difficulty_level": 2,
    }

    session.execute(text(
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
    ), patient)

    consultation = {
        "id": FIXED_CONSULTATION_ID,
        "doctor_id": FIXED_DOCTOR_ID,
        "patient_id": FIXED_PATIENT_ID,
        "status": "in_progress",
        "max_rounds": 20,
        "consultation_type": "initial",
    }

    session.execute(text(
        """
        INSERT INTO consultations (id, doctor_id, patient_id, status, max_rounds, consultation_type)
        VALUES (:id, :doctor_id, :patient_id, :status, :max_rounds, :consultation_type)
        ON DUPLICATE KEY UPDATE
            status=VALUES(status), doctor_id=VALUES(doctor_id)
        """
    ), consultation)

    messages = [
        {"role": "doctor", "content": "您好，请问今天来就诊主要是什么问题？", "sequence": 1},
        {"role": "patient", "content": "我最近一周总是头痛，而且感觉很乏力。", "sequence": 2},
        {"role": "doctor", "content": "头痛是哪个部位？是持续性的还是一阵一阵的？", "sequence": 3},
        {"role": "patient", "content": "主要是后脑勺和两侧，持续性的钝痛，休息后会好一点。", "sequence": 4},
        {"role": "doctor", "content": "您的血压最近有监测吗？降压药还在吃吗？", "sequence": 5},
        {"role": "patient", "content": "药在吃，但是最近血压计测出来偏高一些，大概150/95左右。", "sequence": 6},
    ]

    for msg in messages:
        msg_with_fk = {**msg, "consultation_id": FIXED_CONSULTATION_ID}
        session.execute(text(
            """
            INSERT INTO consultation_messages (consultation_id, role, content, sequence)
            VALUES (:consultation_id, :role, :content, :sequence)
            ON DUPLICATE KEY UPDATE content=VALUES(content)
            """
        ), msg_with_fk)

    print(f"[seed] Patient: {patient['name']} (id={FIXED_PATIENT_ID})")
    print(f"[seed] Consultation: id={FIXED_CONSULTATION_ID}, status=in_progress, "
          f"messages={len(messages)}, doctor_id={FIXED_DOCTOR_ID}")
    return patient, consultation, messages


# ──────────────────────────────────────────
# Low-evidence index fixture
# ──────────────────────────────────────────
def seed_low_evidence_index():
    """Persist one synthetic dense/BM25 generation and activate it in Redis.

    This uses the same Chroma, BM25 artifact, manifest, and active-generation
    contracts as production. The only synthetic part is the document content.
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
            "version": "e2e-v12",
            "created_at": "2026-08-12T00:00:00+00:00",
        },
    }

    from app.services.rag.embeddings import EMBEDDING_DIM, EMBEDDING_MODEL
    from app.services.rag.indexing.builder import _publish_chroma_candidate
    from app.services.rag.indexing.chunking import CHUNK_SIZE
    from app.services.rag.indexing.manifest import (
        RAGIndexManifest,
        compute_corpus_sha256,
        load_rag_index_manifest,
        manifest_path,
        write_rag_index_manifest,
    )
    from app.services.rag.lexical.artifacts import build_bm25_artifact
    from app.services.rag.lexical.tokenizer import TOKENIZER_VERSION

    # Stable non-zero vector; query embeddings still come from mock-openai.
    digest = hashlib.sha256(synthetic_chunk["content"].encode("utf-8")).digest()
    raw = [float(digest[index % len(digest)] + 1) for index in range(EMBEDDING_DIM)]
    norm = math.sqrt(sum(value * value for value in raw))
    embedding = [value / norm for value in raw]
    metadata = {
        **synthetic_chunk["metadata"],
        "page": 1,
        "heading_path": "E2E synthetic evidence",
        "content_type": "text",
        "chunk_seq": 0,
    }
    record = {
        "id": synthetic_chunk["id"],
        "text": synthetic_chunk["content"],
        "metadata": metadata,
        "embedding": embedding,
    }
    bm25_documents = [{"id": record["id"], "text": record["text"], **metadata}]
    corpus_sha256 = compute_corpus_sha256(bm25_documents)

    if manifest_path(E2E_INDEX_VERSION).exists():
        manifest = load_rag_index_manifest(E2E_INDEX_VERSION)
    else:
        collection_name = _publish_chroma_candidate(E2E_INDEX_VERSION, [record])
        build_bm25_artifact(E2E_INDEX_VERSION, bm25_documents)
        manifest = RAGIndexManifest(
            index_generation=E2E_INDEX_VERSION,
            corpus_sha256=corpus_sha256,
            source_count=1,
            chunk_count=1,
            parser_version="e2e-synthetic-v1",
            chunker_version=f"e2e-fixed-{CHUNK_SIZE}",
            tokenizer_version=TOKENIZER_VERSION,
            embedding_model=EMBEDDING_MODEL,
            embedding_dimension=EMBEDDING_DIM,
            chroma_collection=collection_name,
            bm25_artifact=f"{E2E_INDEX_VERSION}/bm25",
            sparse_artifact=None,
            created_at=datetime(2026, 8, 12, tzinfo=timezone.utc),
        )
        write_rag_index_manifest(manifest)

    import redis

    from app.core.config import settings
    from app.services.rag.indexing.versioning import ACTIVE_GENERATION_KEY

    client = redis.Redis.from_url(settings.REDIS_CHECKPOINT_URL, decode_responses=True)
    try:
        client.set(ACTIVE_GENERATION_KEY, E2E_INDEX_VERSION)
        assert client.get(ACTIVE_GENERATION_KEY) == E2E_INDEX_VERSION
    finally:
        client.close()

    print(f"[seed] Low-evidence index: {E2E_INDEX_VERSION}")
    print(f"[seed]   candidate_count={manifest.chunk_count}")
    return synthetic_chunk, manifest.model_dump(mode="json")


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

    engine = _get_sync_engine()

    with engine.connect() as conn:
        try:
            # 1. Users
            seed_users(conn)

            # 2. Patient and consultation
            patient, consultation, messages = seed_patient_and_consultation(conn)

            # 3. Persist and activate the low-evidence RAG generation
            chunk, manifest = seed_low_evidence_index()

            conn.commit()
        except Exception:
            conn.rollback()
            raise

    # Assertions
    assert manifest["chunk_count"] == 1
    assert manifest["source_count"] == 1
    assert len(messages) >= 4
    assert consultation["status"] == "in_progress"
    assert FIXED_DOCTOR_ID is not None
    assert FIXED_ADMIN_ID is not None

    print("=" * 60)
    print("V1.2 Coach E2E Seed complete. All assertions passed.")
    print(f"  doctor_id={FIXED_DOCTOR_ID}, admin_id={FIXED_ADMIN_ID}")
    print(f"  patient_id={FIXED_PATIENT_ID}, consultation_id={FIXED_CONSULTATION_ID}")
    print("=" * 60)

    engine.dispose()

    return {
        "doctor_id": FIXED_DOCTOR_ID,
        "admin_id": FIXED_ADMIN_ID,
        "patient_id": FIXED_PATIENT_ID,
        "consultation_id": FIXED_CONSULTATION_ID,
        "manifest": manifest,
    }


if __name__ == "__main__":
    main()
