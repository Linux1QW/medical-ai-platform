# -*- coding: utf-8 -*-
"""患者模拟回放报告回归检查：对照红线阈值输出 PASS/FAIL，可选对比两份报告差值。

用法（在 backend 目录下）：
    .\\venv\\Scripts\\python.exe scripts\\eval_regression.py --report evaluation/reports/patient_ab/ab_XXX.json
    .\\venv\\Scripts\\python.exe scripts\\eval_regression.py --report new.json --baseline old.json

退出码：0=PASS；1=FAIL（破线）；2=SKIP（smoke/无报告）；3=INVALID（legacy/manifest 不合法）。
"""
import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from evaluation.gate import (  # noqa: E402
    GateDecision,
    compute_gate_statistics,
    decision_to_exit_code,
    evaluate_report_gate,
    select_gate_report,
)
from evaluation.patient_regression import (  # noqa: E402
    compare_reports,
    load_report,
)

DEFAULT_THRESHOLDS = Path(__file__).parent.parent / "evaluation" / "patient_ab_thresholds.json"
REPORT_DIR = Path(__file__).parent.parent / "evaluation" / "reports" / "patient_ab"


def _resolve_report(args_report: str) -> Path | None:
    """解析报告路径：显式指定 > 门禁选择 > None。"""
    if args_report:
        p = Path(args_report)
        return p if p.exists() else None
    return select_gate_report(REPORT_DIR)


def main() -> int:
    sys.stdout.reconfigure(encoding="utf-8")
    parser = argparse.ArgumentParser(description="患者模拟回放回归检查")
    parser.add_argument("--report", default="", help="待检报告 JSON；缺省由门禁选择 regression 报告")
    parser.add_argument("--baseline", default="", help="对比基线报告 JSON（可选）")
    parser.add_argument("--thresholds", default=str(DEFAULT_THRESHOLDS), help="阈值 JSON 路径")
    args = parser.parse_args()

    report_path = _resolve_report(args.report)
    if report_path is None or not report_path.exists():
        print("未找到 regression 报告，请先跑 ab_patient_replay.py --cases @eval_set")
        return 2

    report = load_report(report_path)
    thresholds = json.loads(Path(args.thresholds).read_text(encoding="utf-8"))

    # ── 显式门禁 ──
    decision, results = evaluate_report_gate(report, thresholds)
    exit_code = decision_to_exit_code(decision)

    # SKIP / INVALID 直接输出并返回
    if decision is GateDecision.SKIP:
        print(f"报告 {report_path.name} 类型为 smoke，不作为 pre-push 门禁，SKIP。")
        return 2
    if decision is GateDecision.INVALID:
        print(f"报告 {report_path.name} 类型为 legacy_unknown 或 manifest 不合法，INVALID。")
        return 3

    # PASS / FAIL：输出阈值检查结果
    print(f"待检报告: {report_path.name}")
    print("\n" + "=" * 78)
    print(f"{'臂':<14}{'指标':<26}{'约束':<8}{'阈值':>8}{'实测':>10}{'结果':>8}")
    for r in results:
        actual = "-" if r["actual"] is None else f"{r['actual']:.4g}"
        print(f"{r['arm']:<14}{r['metric']:<26}{str(r['bound_type']):<8}"
              f"{r['bound']:>8}{actual:>10}{r['status']:>8}")

    # ── 统计摘要：n、mean、std、CI95 ──
    stats = compute_gate_statistics(report)
    print("\n--- 统计摘要 ---")
    for arm, metrics in stats.items():
        parts = []
        for metric, s in metrics.items():
            parts.append(f"{metric}: n={s['n']} mean={s['mean']:.4g}"
                         f" std={s['std']:.4g} CI95=[{s['ci95_lower']:.4g}, {s['ci95_upper']:.4g}]")
        if parts:
            print(f"[{arm}] " + " | ".join(parts))

    if args.baseline and Path(args.baseline).exists():
        deltas = compare_reports(load_report(args.baseline), report)
        print("\n--- 对比基线差值 (new - old) ---")
        for arm, d in deltas.items():
            print(f"[{arm}] " + ", ".join(f"{k}:{v:+.4g}" for k, v in d.items() if v))

    verdict = "PASS" if decision is GateDecision.PASS else "FAIL"
    print(f"\n回归结论: {verdict}")
    return exit_code


if __name__ == "__main__":
    sys.exit(main())
