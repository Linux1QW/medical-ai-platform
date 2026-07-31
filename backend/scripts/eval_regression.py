# -*- coding: utf-8 -*-
"""患者模拟回放报告回归检查：对照红线阈值输出 PASS/FAIL，可选对比两份报告差值。

用法（在 backend 目录下）：
    .\\venv\\Scripts\\python.exe scripts\\eval_regression.py --report evaluation/reports/patient_ab/ab_XXX.json
    .\\venv\\Scripts\\python.exe scripts\\eval_regression.py --report new.json --baseline old.json

退出码：0=全部 PASS/SKIP；1=存在 FAIL（破线）。
"""
import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from evaluation.patient_regression import (  # noqa: E402
    check_thresholds,
    compare_reports,
    load_report,
)

DEFAULT_THRESHOLDS = Path(__file__).parent.parent / "evaluation" / "patient_ab_thresholds.json"
REPORT_DIR = Path(__file__).parent.parent / "evaluation" / "reports" / "patient_ab"


def _latest_report() -> Path | None:
    if not REPORT_DIR.exists():
        return None
    reports = sorted(REPORT_DIR.glob("ab_*.json"))
    return reports[-1] if reports else None


def main() -> int:
    sys.stdout.reconfigure(encoding="utf-8")
    parser = argparse.ArgumentParser(description="患者模拟回放回归检查")
    parser.add_argument("--report", default="", help="待检报告 JSON；缺省取最新一份")
    parser.add_argument("--baseline", default="", help="对比基线报告 JSON（可选）")
    parser.add_argument("--thresholds", default=str(DEFAULT_THRESHOLDS), help="阈值 JSON 路径")
    args = parser.parse_args()

    report_path = Path(args.report) if args.report else _latest_report()
    if report_path is None or not report_path.exists():
        print("未找到待检报告，请先跑 ab_patient_replay.py")
        return 2
    report = load_report(report_path)
    thresholds = json.loads(Path(args.thresholds).read_text(encoding="utf-8"))

    print(f"待检报告: {report_path.name}")
    results, ok = check_thresholds(report, thresholds)
    print("\n" + "=" * 78)
    print(f"{'臂':<14}{'指标':<26}{'约束':<8}{'阈值':>8}{'实测':>10}{'结果':>8}")
    for r in results:
        actual = "-" if r["actual"] is None else f"{r['actual']:.4g}"
        print(f"{r['arm']:<14}{r['metric']:<26}{str(r['bound_type']):<8}"
              f"{r['bound']:>8}{actual:>10}{r['status']:>8}")

    if args.baseline and Path(args.baseline).exists():
        deltas = compare_reports(load_report(args.baseline), report)
        print("\n--- 对比基线差值 (new - old) ---")
        for arm, d in deltas.items():
            print(f"[{arm}] " + ", ".join(f"{k}:{v:+.4g}" for k, v in d.items() if v))

    verdict = "PASS" if ok else "FAIL"
    print(f"\n回归结论: {verdict}")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
