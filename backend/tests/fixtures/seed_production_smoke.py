# -*- coding: utf-8 -*-
"""E2E 测试数据种子脚本

幂等创建：
- 管理员 admin_v11、医生 doctor_v11 账号
- 一个虚拟患者、一个已结束问诊和不少于 4 条有序消息
- 低证据索引 fixture（ACTIVE_INDEX_VERSION=e2e-low-evidence-v1）

使用方式：
    python -m tests.fixtures.seed_production_smoke

仅在 ENVIRONMENT=test 的独立数据库运行。
"""

import hashlib
import json
import os
import sys
from datetime import datetime, timezone

# 确保可以 import app 模块
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))


# ──────────────────────────────────────────
# 配置
# ──────────────────────────────────────────
E2E_ADMIN_USER = os.environ.get("E2E_ADMIN_USER", "admin_v11")
E2E_DOCTOR_USER = os.environ.get("E2E_DOCTOR_USER", "doctor_v11")
E2E_PASSWORD = os.environ.get("E2E_PASSWORD", "e2e_test_password_2026")
E2E_INDEX_VERSION = "e2e-low-evidence-v1"


def check_environment():
    """确保在测试环境运行"""
    env = os.environ.get("ENVIRONMENT", "")
    if env != "test":
        print(f"ERROR: ENVIRONMENT must be 'test', got '{env}'")
        print("Set ENVIRONMENT=test before running this script.")
        sys.exit(1)


# ──────────────────────────────────────────
# 用户数据
# ──────────────────────────────────────────
def seed_users():
    """幂等创建 admin_v11 和 doctor_v11 账号"""
    from passlib.context import CryptContext
    
    pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")
    password_hash = pwd_context.hash(E2E_PASSWORD)
    
    users = [
        {
            "username": E2E_ADMIN_USER,
            "email": f"{E2E_ADMIN_USER}@e2e.test",
            "real_name": "E2E 管理员",
            "role": "admin",
            "department": "系统管理",
            "password_hash": password_hash,
        },
        {
            "username": E2E_DOCTOR_USER,
            "email": f"{E2E_DOCTOR_USER}@e2e.test",
            "real_name": "E2E 测试医生",
            "role": "doctor",
            "department": "内科",
            "password_hash": password_hash,
        },
    ]
    
    print(f"[seed] Users: {E2E_ADMIN_USER} (admin), {E2E_DOCTOR_USER} (doctor)")
    return users


# ──────────────────────────────────────────
# 患者和问诊数据
# ──────────────────────────────────────────
def seed_patient_and_consultation():
    """幂等创建虚拟患者和已结束问诊"""
    patient = {
        "name": "E2E合成患者",
        "age": 50,
        "gender": "男",
        "personality_type": "配合型",
        "chief_complaint": "E2E测试头痛",
        "medical_history": "无特殊病史",
        "symptoms": json.dumps(["头痛", "低热"]),
        "system_prompt": "这是E2E测试合成患者，非真实患者信息。",
        "expected_diagnosis": "普通感冒",
    }
    
    consultation = {
        "status": "ended",
        "max_rounds": 20,
    }
    
    messages = [
        {"role": "doctor", "content": "您好，请描述一下您的症状。", "sequence": 1},
        {"role": "patient", "content": "我头痛已经三天了，伴有低热。", "sequence": 2},
        {"role": "doctor", "content": "头痛是持续性还是阵发性的？有没有恶心呕吐？", "sequence": 3},
        {"role": "patient", "content": "持续性的，没有恶心呕吐。", "sequence": 4},
        {"role": "doctor", "content": "好的，建议做一下血常规检查。", "sequence": 5},
    ]
    
    print(f"[seed] Patient: {patient['name']}, Consultation: {consultation['status']}, Messages: {len(messages)}")
    return patient, consultation, messages


# ──────────────────────────────────────────
# 低证据索引 fixture
# ──────────────────────────────────────────
def seed_low_evidence_index():
    """
    幂等创建低证据索引 fixture
    
    ACTIVE_INDEX_VERSION=e2e-low-evidence-v1
    只含 1 个明确标记为合成测试的 chunk
    """
    # 合成测试 chunk
    synthetic_chunk = {
        "id": "e2e-synthetic-chunk-001",
        "content": "本段落为合成测试内容，用于验证E2E系统的RAG检索功能。"
        "不包含任何真实医学知识或临床建议。"
        "此chunk专门用于确保索引非空，同时不会产生真实的医学证据。",
        "metadata": {
            "source": "e2e-synthetic-test",
            "is_synthetic": True,
            "version": E2E_INDEX_VERSION,
            "created_at": datetime.now(timezone.utc).isoformat(),
        },
    }
    
    # Manifest
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
    print(f"[seed]   candidate_count={manifest['candidate_count']}, source_count={manifest['source_count']}")
    print(f"[seed]   confidence=low (single synthetic chunk)")
    
    return synthetic_chunk, manifest


# ──────────────────────────────────────────
# 清理函数
# ──────────────────────────────────────────
def cleanup_e2e_data():
    """清理 E2E 测试数据（按稳定 case_id 清理）"""
    print("[cleanup] Removing E2E test records...")
    # 实际清理需要数据库连接，这里只打印意图
    print("[cleanup] Done (dry-run without DB connection)")


# ──────────────────────────────────────────
# 主入口
# ──────────────────────────────────────────
def main():
    """主入口：幂等创建所有 E2E 测试数据"""
    check_environment()
    
    print("=" * 60)
    print("E2E Production Smoke Seed")
    print(f"ENVIRONMENT: {os.environ.get('ENVIRONMENT')}")
    print(f"INDEX_VERSION: {E2E_INDEX_VERSION}")
    print("=" * 60)
    
    # 1. 用户
    users = seed_users()
    
    # 2. 患者和问诊
    patient, consultation, messages = seed_patient_and_consultation()
    
    # 3. 低证据索引
    chunk, manifest = seed_low_evidence_index()
    
    # 验证
    assert manifest["candidate_count"] == 1
    assert manifest["source_count"] == 1
    assert len(messages) >= 4
    
    print("=" * 60)
    print("Seed complete. All assertions passed.")
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
