# -*- coding: utf-8 -*-
"""患者模拟回放报告的回归阈值检查与两报告对比。

阈值键约定：<metric>_min / <metric>_max，metric 对应臂 summary 中的字段名。
缺失的 metric 标记 SKIP（不算破线），保证阈值文件可先行占位、后续回填。
"""
import json
import logging

logger = logging.getLogger(__name__)


def load_report(path) -> dict:
    """加载回放报告 JSON。"""
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def extract_arm_metrics(report: dict, arm: str) -> dict:
    """跨病例聚合某一臂的数值型 summary 指标（取均值）。"""
    sums: dict[str, float] = {}
    counts: dict[str, int] = {}
    for case in report.get("cases", []):
        summary = (case.get(arm) or {}).get("summary") or {}
        for key, val in summary.items():
            if isinstance(val, bool) or not isinstance(val, (int, float)):
                continue
            sums[key] = sums.get(key, 0.0) + val
            counts[key] = counts.get(key, 0) + 1
    return {k: round(sums[k] / counts[k], 4) for k in sums}


def _split_bound(threshold_key: str):
    """拆分阈值键 -> (metric, bound_type)；非 _min/_max 后缀返回 (None, None)。"""
    if threshold_key.endswith("_min"):
        return threshold_key[: -len("_min")], "min"
    if threshold_key.endswith("_max"):
        return threshold_key[: -len("_max")], "max"
    return None, None


def check_thresholds(report: dict, thresholds: dict) -> tuple[list[dict], bool]:
    """对照阈值检查报告各臂指标。

    返回 (results, all_ok)。results 每项含 arm/metric/bound_type/bound/actual/status。
    status: PASS / FAIL / SKIP（指标缺失或键格式非法）。all_ok 为无 FAIL。
    """
    results: list[dict] = []
    all_ok = True
    for arm, rules in thresholds.items():
        # 跳过注释/元数据键（如 _comment）与非规则值
        if arm.startswith("_") or not isinstance(rules, dict):
            continue
        metrics = extract_arm_metrics(report, arm)
        for threshold_key, bound in rules.items():
            metric, bound_type = _split_bound(threshold_key)
            entry = {
                "arm": arm,
                "metric": metric or threshold_key,
                "bound_type": bound_type,
                "bound": bound,
                "actual": None,
                "status": "SKIP",
            }
            if metric is None or metric not in metrics:
                results.append(entry)
                continue
            actual = metrics[metric]
            entry["actual"] = actual
            if bound_type == "min":
                ok = actual >= bound
            else:  # max
                ok = actual <= bound
            entry["status"] = "PASS" if ok else "FAIL"
            if not ok:
                all_ok = False
            results.append(entry)
    return results, all_ok


def compare_reports(old: dict, new: dict, arms=None) -> dict:
    """对比两份报告每臂关键指标的差值（new - old）。"""
    arms = arms or ["legacy", "agent_ledger", "agent_tool"]
    deltas: dict[str, dict] = {}
    for arm in arms:
        om = extract_arm_metrics(old, arm)
        nm = extract_arm_metrics(new, arm)
        keys = set(om) | set(nm)
        deltas[arm] = {
            k: round(nm.get(k, 0.0) - om.get(k, 0.0), 4)
            for k in sorted(keys)
        }
    return deltas
