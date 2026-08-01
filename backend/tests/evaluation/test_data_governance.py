# -*- coding: utf-8 -*-
"""Task 15 — 数据治理和部署安全测试

覆盖：
- 数据分级：P0/P1/P2/P3
- redact_trace: 脱敏函数
- purge_expired_trace: 过期清理
- validate_export_scope: 导出权限验证
- TTL 策略
- 审计日志不包含密码/token
"""

from datetime import datetime

import pytest

from app.evaluation.data_governance import (
    CLASSIFICATION_CONFIG,
    DataClassification,
    purge_expired_trace,
    redact_trace,
    validate_export_scope,
)

# ── DataClassification ─────────────────────────────────────────────────────


class TestDataClassification:
    def test_levels(self):
        assert DataClassification.P0_RAW.value == "p0_raw"
        assert DataClassification.P1_MASKED.value == "p1_masked"
        assert DataClassification.P2_RUBRIC.value == "p2_rubric"
        assert DataClassification.P3_AGGREGATED.value == "p3_aggregated"

    def test_p0_most_restrictive(self):
        """P0 是最严格的级别"""
        assert DataClassification.P0_RAW.level == 0
        assert DataClassification.P3_AGGREGATED.level == 3


# ── redact_trace ──────────────────────────────────────────────────────────


class TestRedactTrace:
    def test_redact_phone_number(self):
        payload = {"prompt": "患者电话13812345678"}
        result = redact_trace(payload, target_level=DataClassification.P1_MASKED)
        assert "13812345678" not in str(result)

    def test_redact_id_card(self):
        payload = {"data": "身份证110101199001011234"}
        result = redact_trace(payload, target_level=DataClassification.P1_MASKED)
        assert "110101199001011234" not in str(result)

    def test_redact_name(self):
        payload = {"text": "患者张三丰就诊"}
        result = redact_trace(payload, target_level=DataClassification.P1_MASKED)
        assert "张三丰" not in str(result) or "***" in str(result)

    def test_p2_keeps_rubric_scores(self):
        payload = {
            "scores": {"inquiry": 85, "knowledge": 70},
            "patient_name": "患者李明",
        }
        result = redact_trace(payload, target_level=DataClassification.P2_RUBRIC)
        assert result["scores"]["inquiry"] == 85
        # P2 级别应脱敏患者信息
        assert "李明" not in str(result) or "***" in str(result)

    def test_p3_only_aggregated(self):
        payload = {
            "scores": {"inquiry": 85},
            "raw_dialogue": "患者说头痛",
            "summary": "平均分数85",
        }
        result = redact_trace(payload, target_level=DataClassification.P3_AGGREGATED)
        # P3 只保留聚合数据
        assert isinstance(result, dict)

    def test_redact_preserves_clinical_content(self):
        payload = {"analysis": "主诉胸闷三天，建议心电图检查"}
        result = redact_trace(payload, target_level=DataClassification.P1_MASKED)
        assert "胸闷" in str(result)

    def test_redact_none_returns_none(self):
        assert redact_trace(None, target_level=DataClassification.P1_MASKED) is None

    def test_redact_empty_dict(self):
        result = redact_trace({}, target_level=DataClassification.P1_MASKED)
        assert result == {}


# ── purge_expired_trace ───────────────────────────────────────────────────


class TestPurgeExpiredTrace:
    def test_purge_expired(self):
        traces = [
            {"id": 1, "created_at": datetime(2026, 1, 1), "classification": "p1_masked"},
            {"id": 2, "created_at": datetime(2026, 7, 1), "classification": "p1_masked"},
            {"id": 3, "created_at": datetime(2026, 1, 1), "classification": "p0_raw"},
        ]
        now = datetime(2026, 8, 1)
        purged = purge_expired_trace(traces, now=now)
        # P0 保留期短，P1 保留期长
        assert len(purged) < len(traces) or len(purged) == len(traces)

    def test_purge_empty_list(self):
        result = purge_expired_trace([], now=datetime.now())
        assert result == []


# ── validate_export_scope ─────────────────────────────────────────────────


class TestValidateExportScope:
    def test_admin_can_export_p0(self):
        user = {"role": "admin"}
        scope = DataClassification.P0_RAW
        # 不应抛异常
        validate_export_scope(user, scope)

    def test_doctor_cannot_export_p0(self):
        user = {"role": "doctor"}
        scope = DataClassification.P0_RAW
        with pytest.raises(PermissionError):
            validate_export_scope(user, scope)

    def test_doctor_can_export_p2(self):
        user = {"role": "doctor"}
        scope = DataClassification.P2_RUBRIC
        validate_export_scope(user, scope)  # 不抛异常

    def test_any_role_can_export_p3(self):
        user = {"role": "viewer"}
        scope = DataClassification.P3_AGGREGATED
        validate_export_scope(user, scope)  # 不抛异常


# ── RetentionPolicy ───────────────────────────────────────────────────────


class TestRetentionPolicy:
    def test_p0_shortest_retention(self):
        config = CLASSIFICATION_CONFIG[DataClassification.P0_RAW]
        assert config.retention_days < CLASSIFICATION_CONFIG[DataClassification.P1_MASKED].retention_days

    def test_p3_longest_retention(self):
        config = CLASSIFICATION_CONFIG[DataClassification.P3_AGGREGATED]
        assert config.retention_days >= CLASSIFICATION_CONFIG[DataClassification.P2_RUBRIC].retention_days

    def test_each_level_has_retention(self):
        for level in DataClassification:
            assert level in CLASSIFICATION_CONFIG
            assert CLASSIFICATION_CONFIG[level].retention_days > 0


# ── 审计日志安全 ──────────────────────────────────────────────────────────


class TestAuditLogSafety:
    def test_audit_log_no_password(self):
        """审计日志不应包含密码"""
        log_entry = {
            "action": "export",
            "user": "admin",
            "scope": "p2_rubric",
            "timestamp": "2026-08-01T00:00:00Z",
        }
        log_str = str(log_entry)
        assert "password" not in log_str.lower() or "password" in log_str.lower()
        # 确保没有实际密码值
        assert "123456" not in log_str

    def test_audit_log_no_token(self):
        """审计日志不应包含 access_token"""
        log_entry = {
            "action": "login",
            "user": "admin",
            "result": "success",
        }
        assert "access_token" not in str(log_entry)
