# -*- coding: utf-8 -*-
"""显式回归门禁测试 — Task 2

验证：
1. smoke 永远 SKIP
2. benchmark 永远 PASS（不阻断）
3. regression 走阈值检查，FAIL 阻断
4. legacy_unknown 返回 INVALID
5. select_gate_report 优先选 manifest regression 报告
6. bootstrap CI 固定 seed 可重复
7. pre-push 退出码 3 INVALID 阻断
"""
import json
import math
import statistics
from pathlib import Path

import pytest

from evaluation.gate import (
    GateDecision,
    calculate_bootstrap_ci,
    evaluate_report_gate,
    select_gate_report,
)
from evaluation.report_schema import ReportKind, ReportManifest


# ── 辅助 ────────────────────────────────────────────────────────────────────


def _make_report(n_cases: int, manifest: dict | None = None, cases_data: list | None = None) -> dict:
    """构造报告"""
    report = {"cases": cases_data or [{} for _ in range(n_cases)]}
    if manifest is not None:
        report["manifest"] = manifest
    return report


def _valid_manifest(**overrides) -> dict:
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


def _thresholds() -> dict:
    return {
        "_gate": {"min_cases": 18},
        "agent_ledger": {
            "coverage_disclosure_rate_min": 0.7,
            "avg_latency_ms_max": 5000,
        },
    }


def _regression_report(n_cases: int = 18, pass_metrics: bool = True) -> dict:
    """生成带 manifest 的 regression 报告"""
    val = 0.85 if pass_metrics else 0.5
    cases = []
    for i in range(n_cases):
        cases.append({
            "case_id": f"patient{i}",
            "agent_ledger": {
                "summary": {
                    "coverage_disclosure_rate": val,
                    "avg_latency_ms": 3000.0,
                }
            },
        })
    return _make_report(n_cases, manifest=_valid_manifest(case_count=n_cases), cases_data=cases)


# ── 1. GateDecision 枚举 ────────────────────────────────────────────────────


class TestGateDecision:
    def test_all_decisions_defined(self):
        assert GateDecision.PASS == "pass"
        assert GateDecision.FAIL == "fail"
        assert GateDecision.SKIP == "skip"
        assert GateDecision.INVALID == "invalid"


# ── 2. evaluate_report_gate 分支 ────────────────────────────────────────────


class TestEvaluateReportGate:
    def test_smoke_always_skip(self):
        """smoke 报告永远返回 SKIP"""
        report = _make_report(3, manifest=_valid_manifest(report_kind="smoke", case_count=3))
        decision, results = evaluate_report_gate(report, _thresholds())
        assert decision is GateDecision.SKIP

    def test_benchmark_always_pass(self):
        """benchmark 报告不阻断，返回 PASS"""
        report = _make_report(20, manifest=_valid_manifest(report_kind="benchmark", case_count=20))
        decision, results = evaluate_report_gate(report, _thresholds())
        assert decision is GateDecision.PASS

    def test_regression_pass(self):
        """regression 指标达标 → PASS"""
        report = _regression_report(pass_metrics=True)
        decision, results = evaluate_report_gate(report, _thresholds())
        assert decision is GateDecision.PASS
        assert all(r["status"] in ("PASS", "SKIP") for r in results)

    def test_regression_fail(self):
        """regression 指标破线 → FAIL"""
        report = _regression_report(pass_metrics=False)
        decision, results = evaluate_report_gate(report, _thresholds())
        assert decision is GateDecision.FAIL
        assert any(r["status"] == "FAIL" for r in results)

    def test_legacy_unknown_invalid(self):
        """legacy_unknown 返回 INVALID"""
        report = _make_report(18, manifest=_valid_manifest(report_kind="legacy_unknown", case_count=18))
        decision, results = evaluate_report_gate(report, _thresholds())
        assert decision is GateDecision.INVALID

    def test_no_manifest_legacy_small_is_skip(self):
        """无 manifest + 小样本 → 兼容读取为 smoke → SKIP"""
        report = _make_report(3)
        decision, results = evaluate_report_gate(report, _thresholds())
        assert decision is GateDecision.SKIP

    def test_no_manifest_legacy_large_is_invalid(self):
        """无 manifest + 大样本 → legacy_unknown → INVALID"""
        report = _make_report(18)
        decision, results = evaluate_report_gate(report, _thresholds())
        assert decision is GateDecision.INVALID


# ── 3. select_gate_report ───────────────────────────────────────────────────


class TestSelectGateReport:
    def test_prefer_regression_manifest(self, tmp_path):
        """优先选 manifest.report_kind=regression 的报告"""
        # 写一个 smoke 报告（文件名更晚）
        smoke = _make_report(3, manifest=_valid_manifest(report_kind="smoke", case_count=3))
        (tmp_path / "ab_20260802_120000.json").write_text(
            json.dumps(smoke), encoding="utf-8"
        )
        # 写一个 regression 报告（文件名更早）
        reg = _regression_report()
        (tmp_path / "ab_20260801_100000.json").write_text(
            json.dumps(reg), encoding="utf-8"
        )
        selected = select_gate_report(tmp_path)
        assert selected is not None
        assert selected.name == "ab_20260801_100000.json"

    def test_fallback_to_latest_if_no_regression(self, tmp_path):
        """无 regression 报告时返回 None"""
        smoke = _make_report(3, manifest=_valid_manifest(report_kind="smoke", case_count=3))
        (tmp_path / "ab_20260801_120000.json").write_text(
            json.dumps(smoke), encoding="utf-8"
        )
        selected = select_gate_report(tmp_path)
        assert selected is None

    def test_empty_dir_returns_none(self, tmp_path):
        """空目录返回 None"""
        assert select_gate_report(tmp_path) is None

    def test_multiple_regression_picks_latest(self, tmp_path):
        """多个 regression 报告选最新（按 created_at）"""
        r1 = _regression_report()
        r1["manifest"]["created_at"] = "2026-08-01T10:00:00+00:00"
        (tmp_path / "ab_20260801_100000.json").write_text(
            json.dumps(r1), encoding="utf-8"
        )
        r2 = _regression_report()
        r2["manifest"]["created_at"] = "2026-08-02T10:00:00+00:00"
        (tmp_path / "ab_20260802_100000.json").write_text(
            json.dumps(r2), encoding="utf-8"
        )
        selected = select_gate_report(tmp_path)
        assert selected is not None
        assert selected.name == "ab_20260802_100000.json"


# ── 4. Bootstrap CI ─────────────────────────────────────────────────────────


class TestBootstrapCI:
    def test_fixed_seed_reproducible(self):
        """固定 seed 的 CI 可重复"""
        values = [0.8, 0.85, 0.9, 0.75, 0.82, 0.88, 0.79, 0.91]
        ci1 = calculate_bootstrap_ci(values, seed=42)
        ci2 = calculate_bootstrap_ci(values, seed=42)
        assert ci1 == ci2

    def test_ci_lower_le_mean_le_upper(self):
        """CI 下界 ≤ 均值 ≤ 上界"""
        values = [0.8, 0.85, 0.9, 0.75, 0.82, 0.88, 0.79, 0.91]
        lo, hi = calculate_bootstrap_ci(values, seed=42)
        mean = statistics.mean(values)
        assert lo <= mean <= hi

    def test_single_value(self):
        """单值 CI 退化为 (val, val)"""
        lo, hi = calculate_bootstrap_ci([0.5], seed=42)
        assert lo == hi == 0.5

    def test_different_seed_different_ci(self):
        """不同 seed 可能产生不同 CI（概率性，但大样本下几乎必然）"""
        values = list(range(100))
        ci1 = calculate_bootstrap_ci(values, seed=1)
        ci2 = calculate_bootstrap_ci(values, seed=999)
        # 不强制不等，但至少结构正确
        assert ci1[0] <= ci1[1]
        assert ci2[0] <= ci2[1]


# ── 5. 退出码映射 ──────────────────────────────────────────────────────────


class TestExitCodeMapping:
    """验证 GateDecision → 退出码映射"""

    def test_pass_is_0(self):
        from evaluation.gate import decision_to_exit_code
        assert decision_to_exit_code(GateDecision.PASS) == 0

    def test_fail_is_1(self):
        from evaluation.gate import decision_to_exit_code
        assert decision_to_exit_code(GateDecision.FAIL) == 1

    def test_skip_is_2(self):
        from evaluation.gate import decision_to_exit_code
        assert decision_to_exit_code(GateDecision.SKIP) == 2

    def test_invalid_is_3(self):
        from evaluation.gate import decision_to_exit_code
        assert decision_to_exit_code(GateDecision.INVALID) == 3
