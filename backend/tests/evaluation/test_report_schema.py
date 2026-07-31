# -*- coding: utf-8 -*-
"""ReportManifest 统一报告协议测试 — Task 1

验证：
1. 旧报告兼容读取（无 manifest 时按病例数推断 kind）
2. 新报告强制 manifest 完整性
3. 版本字段缺失不得静默通过
4. case_count 必须与 cases 数量一致
"""
import json
from datetime import datetime, timezone

import pytest
from pydantic import ValidationError

from evaluation.report_schema import (
    ReportKind,
    ReportManifest,
    load_report_manifest,
    validate_report_manifest,
)


# ── 辅助 ────────────────────────────────────────────────────────────────────


def _make_report(n_cases: int, manifest: dict | None = None, **extra) -> dict:
    """构造带 N 例 cases 的报告"""
    report = {"cases": [{} for _ in range(n_cases)], **extra}
    if manifest is not None:
        report["manifest"] = manifest
    return report


def _valid_manifest_dict(**overrides) -> dict:
    """生成合法 manifest 字典"""
    base = {
        "report_kind": "regression",
        "report_id": "ab_20260801_120000",
        "created_at": "2026-08-01T12:00:00+00:00",
        "case_count": 18,
        "dataset_version": "patient_sim_v1",
        "model_version": "qwen3.7-plus",
        "prompt_version": "v1",
        "judge_version": "judge_v1",
        "kb_version": "rag-v1",
        "scoring_policy_version": "scoring_v1",
        "seed": 42,
    }
    base.update(overrides)
    return base


# ── 1. 旧报告兼容读取 ───────────────────────────────────────────────────────


class TestLegacyCompat:
    """无 manifest 的旧报告兼容读取"""

    def test_legacy_small_report_is_smoke(self):
        """无 manifest + 病例数 < 18 → 兼容读取为 smoke"""
        report = _make_report(3)
        manifest = load_report_manifest(report, allow_legacy=True)
        assert manifest.report_kind is ReportKind.SMOKE
        assert manifest.case_count == 3

    def test_legacy_17_cases_is_smoke(self):
        """17 例无 manifest → smoke"""
        report = _make_report(17)
        manifest = load_report_manifest(report, allow_legacy=True)
        assert manifest.report_kind is ReportKind.SMOKE

    def test_legacy_large_report_is_legacy_unknown(self):
        """无 manifest + 病例数 >= 18 → legacy_unknown，不得作为正式 regression"""
        report = _make_report(18)
        manifest = load_report_manifest(report, allow_legacy=True)
        assert manifest.report_kind is ReportKind.LEGACY_UNKNOWN
        assert manifest.case_count == 18

    def test_legacy_not_allowed_raises(self):
        """allow_legacy=False 时无 manifest 报告必须报错"""
        report = _make_report(3)
        with pytest.raises((ValidationError, ValueError)):
            load_report_manifest(report, allow_legacy=False)

    def test_legacy_manifest_has_placeholder_versions(self):
        """旧报告兼容读取时版本字段为占位值 'unknown'"""
        report = _make_report(5)
        manifest = load_report_manifest(report, allow_legacy=True)
        assert manifest.model_version == "unknown"
        assert manifest.dataset_version == "unknown"


# ── 2. 新报告 manifest 完整性 ───────────────────────────────────────────────


class TestManifestValidation:
    """正式报告必须携带完整 manifest"""

    def test_valid_regression_manifest(self):
        """完整 regression manifest 通过校验"""
        report = _make_report(18, manifest=_valid_manifest_dict())
        manifest = load_report_manifest(report, allow_legacy=False)
        assert manifest.report_kind is ReportKind.REGRESSION
        assert manifest.case_count == 18
        assert manifest.model_version == "qwen3.7-plus"

    def test_smoke_manifest(self):
        """smoke manifest 合法"""
        m = _valid_manifest_dict(report_kind="smoke", case_count=3)
        report = _make_report(3, manifest=m)
        manifest = load_report_manifest(report, allow_legacy=False)
        assert manifest.report_kind is ReportKind.SMOKE

    def test_benchmark_manifest(self):
        """benchmark manifest 合法"""
        m = _valid_manifest_dict(report_kind="benchmark", case_count=50)
        report = _make_report(50, manifest=m)
        manifest = load_report_manifest(report, allow_legacy=False)
        assert manifest.report_kind is ReportKind.BENCHMARK

    def test_invalid_kind_rejected(self):
        """非法 report_kind 被拒绝"""
        m = _valid_manifest_dict(report_kind="nightly")
        report = _make_report(18, manifest=m)
        with pytest.raises((ValidationError, ValueError)):
            load_report_manifest(report, allow_legacy=False)

    def test_missing_version_field_rejected(self):
        """缺少版本字段不得静默通过"""
        m = _valid_manifest_dict()
        del m["model_version"]
        report = _make_report(18, manifest=m)
        with pytest.raises((ValidationError, ValueError)):
            load_report_manifest(report, allow_legacy=False)

    def test_case_count_mismatch_rejected(self):
        """manifest.case_count 与 cases 数量不一致时报错"""
        m = _valid_manifest_dict(case_count=20)
        report = _make_report(18, manifest=m)
        with pytest.raises((ValidationError, ValueError)):
            load_report_manifest(report, allow_legacy=False)


# ── 3. validate_report_manifest 错误列表 ────────────────────────────────────


class TestValidateFunction:
    """validate_report_manifest 返回错误列表"""

    def test_valid_manifest_no_errors(self):
        manifest = ReportManifest(**_valid_manifest_dict())
        errors = validate_report_manifest(manifest)
        assert errors == []

    def test_unknown_version_flagged(self):
        """版本为 'unknown' 时产生警告（非阻断）"""
        m = _valid_manifest_dict(model_version="unknown")
        manifest = ReportManifest(**m)
        errors = validate_report_manifest(manifest)
        assert any("model_version" in e for e in errors)

    def test_seed_none_for_regression_flagged(self):
        """regression 报告无 seed 时产生警告"""
        m = _valid_manifest_dict(seed=None)
        manifest = ReportManifest(**m)
        errors = validate_report_manifest(manifest)
        assert any("seed" in e for e in errors)


# ── 4. ReportKind 枚举完整性 ────────────────────────────────────────────────


class TestReportKind:
    def test_all_kinds_defined(self):
        assert ReportKind.SMOKE == "smoke"
        assert ReportKind.REGRESSION == "regression"
        assert ReportKind.BENCHMARK == "benchmark"
        assert ReportKind.LEGACY_UNKNOWN == "legacy_unknown"

    def test_legacy_unknown_not_gate_eligible(self):
        """legacy_unknown 不应被视为正式门禁候选"""
        assert ReportKind.LEGACY_UNKNOWN != ReportKind.REGRESSION
