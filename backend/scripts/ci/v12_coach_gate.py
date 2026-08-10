"""V1.2 coach release gate CLI.

Runs the 72-case benchmark and evaluates against release policy.
Fail-closed: any missing metric -> passed=false.

Usage:
    python scripts/ci/v12_coach_gate.py [--json] [--output-dir DIR]
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from datetime import datetime, timezone

# Ensure backend root is on sys.path
_BACKEND = os.path.normpath(os.path.join(os.path.dirname(__file__), "..", ".."))
if _BACKEND not in sys.path:
    sys.path.insert(0, _BACKEND)

from evaluation.coach_cases.coach_dataset import load_cases, validate_dataset
from evaluation.coach_eval import (
    build_report,
    evaluate_all,
    evaluate_release_policy,
)
from evaluation.coach_metrics import collect_provenance, validate_provenance


def run_gate(
    *,
    execution_mode: str = "live",
    output_dir: str | None = None,
    output_json: bool = False,
) -> bool:
    """Run the 72-case benchmark gate. Returns True if passed."""
    print("=" * 60)
    print("V1.2 Coach Release Gate")
    print("=" * 60)

    # Step 1: Load and validate dataset
    print("\n[1/5] Loading dataset...")
    cases = load_cases()
    count = len(cases)
    print(f"  Loaded {count} cases")

    errors = validate_dataset(cases)
    if errors:
        print(f"  Dataset validation FAILED:")
        for e in errors:
            print(f"    - {e}")
        _write_failure_report(output_dir, "dataset_validation_failed", errors)
        return False
    print("  Dataset validation: PASS")

    # Step 2: Collect provenance
    print("\n[2/5] Collecting provenance...")
    provenance = collect_provenance(execution_mode=execution_mode)
    prov_errors = validate_provenance(provenance)
    if prov_errors:
        print(f"  Provenance validation FAILED:")
        for e in prov_errors:
            print(f"    - {e}")
        _write_failure_report(output_dir, "provenance_validation_failed", prov_errors)
        return False
    print(f"  source_commit: {provenance.source_commit[:12]}...")
    print(f"  dataset_sha256: {provenance.dataset_sha256[:16]}...")
    print(f"  execution_mode: {provenance.execution_mode}")
    print("  Provenance validation: PASS")

    # Step 3: Run benchmark
    print(f"\n[3/5] Running {count}-case benchmark...")
    try:
        results = evaluate_all(cases)
    except Exception as e:
        print(f"  Benchmark FAILED with exception: {e}")
        _write_failure_report(output_dir, "benchmark_exception", [str(e)])
        return False

    # Check for structural failures
    unique_ids = len(set(r.case_id for r in results))
    duplicates = sum(1 for r in results if r.duplicate)
    skipped = sum(1 for r in results if r.skipped)
    timeouts = sum(1 for r in results if r.timeout)
    graph_errors = sum(1 for r in results if r.graph_error)

    print(f"  Unique case IDs: {unique_ids}/{count}")
    print(f"  Duplicates: {duplicates}")
    print(f"  Skipped: {skipped}")
    print(f"  Timeouts: {timeouts}")
    print(f"  Graph errors: {graph_errors}")

    # Step 4: Build report
    print("\n[4/5] Building evaluation report...")
    report = build_report(results, cases)
    print(f"  Intent accuracy: {report.intent_accuracy:.4f}")
    print(f"  Intent macro-F1: {report.intent_macro_f1:.4f}")
    print(f"  Hidden-fact leaks: {report.hidden_fact_leaks}/{count}")
    print(f"  Unsafe suggestions: {report.unsafe_suggestions}/{count}")
    print(f"  Forbidden tool calls: {report.forbidden_tool_calls}")
    print(f"  Trace completeness: {report.trace_completeness:.4f}")

    # Step 5: Evaluate against release policy
    print("\n[5/5] Evaluating against release policy...")
    report = evaluate_release_policy(report)

    if report.passed:
        print("  Release gate: PASS")
    else:
        print("  Release gate: FAIL")
        for reason in report.fail_reasons:
            print(f"    - {reason}")

    # Write full report
    if output_dir is None:
        output_dir = os.path.join(_BACKEND, "benchmark_reports")
    os.makedirs(output_dir, exist_ok=True)

    full_report = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "passed": report.passed,
        "fail_reasons": report.fail_reasons,
        "dataset_size": report.dataset_size,
        "unique_case_ids": report.unique_case_ids,
        "all_strata_present": report.all_strata_present,
        "intent_accuracy": report.intent_accuracy,
        "intent_macro_f1": report.intent_macro_f1,
        "per_label_f1": report.per_label_f1,
        "hidden_fact_leaks": report.hidden_fact_leaks,
        "unsafe_suggestions": report.unsafe_suggestions,
        "forbidden_tool_calls": report.forbidden_tool_calls,
        "trace_completeness": report.trace_completeness,
        "duplicate_count": report.duplicate_count,
        "skipped_count": report.skipped_count,
        "timeout_count": report.timeout_count,
        "graph_error_count": report.graph_error_count,
        "provenance": provenance.to_dict(),
    }

    report_path = os.path.join(output_dir, "v12_coach_gate_report.json")
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(full_report, f, indent=2, ensure_ascii=False)
    print(f"\nReport written to: {report_path}")

    if output_json:
        print(json.dumps(full_report, indent=2, ensure_ascii=False))

    print(f"\n{'=' * 60}")
    status = "PASS" if report.passed else "FAIL"
    print(f"V1.2 Coach Release Gate: {status}")
    print(f"{'=' * 60}")

    return report.passed


def _write_failure_report(
    output_dir: str | None,
    reason: str,
    details: list[str],
) -> None:
    """Write a failure report when gate cannot complete."""
    if output_dir is None:
        output_dir = os.path.join(_BACKEND, "benchmark_reports")
    os.makedirs(output_dir, exist_ok=True)

    report = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "passed": False,
        "fail_reason": reason,
        "details": details,
    }
    report_path = os.path.join(output_dir, "v12_coach_gate_report.json")
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)
    print(f"Failure report written to: {report_path}")


def main() -> None:
    parser = argparse.ArgumentParser(description="V1.2 Coach Release Gate")
    parser.add_argument(
        "--json", action="store_true",
        help="Also print full report as JSON to stdout",
    )
    parser.add_argument(
        "--output-dir", type=str, default=None,
        help="Directory to write report JSON",
    )
    parser.add_argument(
        "--execution-mode", type=str, default="live",
        choices=["live", "ci"],
        help="Execution mode (mock/fake are rejected)",
    )
    args = parser.parse_args()

    success = run_gate(
        execution_mode=args.execution_mode,
        output_dir=args.output_dir,
        output_json=args.json,
    )
    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()
