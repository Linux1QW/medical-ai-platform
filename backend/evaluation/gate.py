# -*- coding: utf-8 -*-
"""显式回归门禁 — 隔离 smoke / benchmark / regression / legacy_unknown。

核心职责：
1. 根据 ReportManifest.report_kind 决定门禁行为
2. 不再依赖文件名时间排序选择报告
3. 提供 bootstrap CI 统计
4. 输出明确的退出码供 pre-push 消费

退出码协议：
  0 = PASS（regression 阈值全过 / benchmark 仅报告）
  1 = FAIL（regression 破线）
  2 = SKIP（smoke / 无报告）
  3 = INVALID（legacy_unknown / manifest 不合法）

用法：
    from evaluation.gate import evaluate_report_gate, select_gate_report

    report_path = select_gate_report(report_dir)
    decision, results = evaluate_report_gate(report, thresholds)
"""
from __future__ import annotations

import json
import random
import statistics
from enum import Enum
from pathlib import Path

from .patient_regression import check_thresholds, extract_arm_metrics
from .report_schema import ReportKind, load_report_manifest

# ── 门禁决策枚举 ─────────────────────────────────────────────────────────────


class GateDecision(str, Enum):
    """门禁决策。

    PASS: 阈值全过或 benchmark 仅报告。
    FAIL: regression 破线，阻断 push。
    SKIP: smoke 或无报告，放行。
    INVALID: legacy_unknown 或 manifest 不合法，阻断 push。
    """

    PASS = "pass"
    FAIL = "fail"
    SKIP = "skip"
    INVALID = "invalid"


def decision_to_exit_code(decision: GateDecision) -> int:
    """GateDecision → pre-push 退出码。"""
    return {
        GateDecision.PASS: 0,
        GateDecision.FAIL: 1,
        GateDecision.SKIP: 2,
        GateDecision.INVALID: 3,
    }[decision]


# ── 门禁评估 ─────────────────────────────────────────────────────────────────


def evaluate_report_gate(
    report: dict,
    thresholds: dict,
) -> tuple[GateDecision, list[dict]]:
    """根据报告 manifest 和阈值决定门禁行为。

    Args:
        report: 完整报告字典。
        thresholds: 阈值配置字典。

    Returns:
        (decision, results) — results 为 check_thresholds 的输出。
    """
    # 加载 manifest（兼容旧报告）
    manifest = load_report_manifest(report, allow_legacy=True)
    kind = manifest.report_kind

    # smoke → SKIP
    if kind is ReportKind.SMOKE:
        return GateDecision.SKIP, []

    # benchmark → PASS（仅报告，不阻断）
    if kind is ReportKind.BENCHMARK:
        return GateDecision.PASS, []

    # legacy_unknown → INVALID
    if kind is ReportKind.LEGACY_UNKNOWN:
        return GateDecision.INVALID, []

    # regression → 走阈值检查
    results, all_ok = check_thresholds(report, thresholds)
    decision = GateDecision.PASS if all_ok else GateDecision.FAIL
    return decision, results


# ── 报告选择 ─────────────────────────────────────────────────────────────────


def select_gate_report(report_dir: Path) -> Path | None:
    """从报告目录选择用于门禁的报告。

    优先选 manifest.report_kind=regression 中 created_at 最新的；
    无 regression 报告时返回 None（不降级到 smoke）。

    Args:
        report_dir: 报告目录路径。

    Returns:
        选中的报告路径，或 None。
    """
    if not report_dir.exists():
        return None

    candidates: list[tuple[str, Path]] = []
    for p in report_dir.glob("ab_*.json"):
        if p.suffix != ".json":
            continue
        try:
            data = json.loads(p.read_text(encoding="utf-8"))
            manifest = load_report_manifest(data, allow_legacy=False)
            if manifest.report_kind is ReportKind.REGRESSION:
                candidates.append((manifest.created_at.isoformat(), p))
        except Exception:
            # 无 manifest 或解析失败 → 跳过
            continue

    if not candidates:
        return None

    # 按 created_at 降序取最新
    candidates.sort(key=lambda x: x[0], reverse=True)
    return candidates[0][1]


# ── Bootstrap CI ─────────────────────────────────────────────────────────────


def calculate_bootstrap_ci(
    values: list[float],
    n_bootstrap: int = 1000,
    confidence: float = 0.95,
    seed: int = 42,
) -> tuple[float, float]:
    """非参数 bootstrap 置信区间。

    Args:
        values: 观测值列表。
        n_bootstrap: 重采样次数。
        confidence: 置信水平。
        seed: 随机种子（保证可重复）。

    Returns:
        (lower, upper) 置信区间。
    """
    if not values:
        return (0.0, 0.0)
    if len(values) == 1:
        return (values[0], values[0])

    rng = random.Random(seed)
    n = len(values)
    means: list[float] = []
    for _ in range(n_bootstrap):
        sample = [rng.choice(values) for _ in range(n)]
        means.append(statistics.mean(sample))

    means.sort()
    alpha = (1 - confidence) / 2
    lo_idx = int(alpha * n_bootstrap)
    hi_idx = int((1 - alpha) * n_bootstrap) - 1
    lo_idx = max(0, min(lo_idx, n_bootstrap - 1))
    hi_idx = max(0, min(hi_idx, n_bootstrap - 1))
    return (means[lo_idx], means[hi_idx])


# ── 统计输出 ─────────────────────────────────────────────────────────────────


def compute_gate_statistics(report: dict) -> dict:
    """为门禁报告计算统计摘要：n、mean、std、CI95。

    Returns:
        {arm: {metric: {n, mean, std, ci95_lower, ci95_upper}}}
    """
    stats: dict[str, dict] = {}
    for arm in ("legacy", "agent_ledger", "agent_tool"):
        metrics = extract_arm_metrics(report, arm)
        arm_stats: dict[str, dict] = {}
        for metric, val in metrics.items():
            # 收集原始值（跨病例）
            raw_values: list[float] = []
            for case in report.get("cases", []):
                summary = (case.get(arm) or {}).get("summary") or {}
                v = summary.get(metric)
                if isinstance(v, (int, float)) and not isinstance(v, bool):
                    raw_values.append(float(v))
            if not raw_values:
                arm_stats[metric] = {"n": 0, "mean": val, "std": 0.0, "ci95_lower": val, "ci95_upper": val}
                continue
            ci = calculate_bootstrap_ci(raw_values, seed=42)
            arm_stats[metric] = {
                "n": len(raw_values),
                "mean": round(statistics.mean(raw_values), 4),
                "std": round(statistics.stdev(raw_values), 4) if len(raw_values) > 1 else 0.0,
                "ci95_lower": round(ci[0], 4),
                "ci95_upper": round(ci[1], 4),
            }
        stats[arm] = arm_stats
    return stats
