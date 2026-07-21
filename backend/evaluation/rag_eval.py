"""
Main entry point for RAG evaluation.
"""
import argparse
import asyncio
import sys
from pathlib import Path

from .ab_compare import run_ab_comparison, write_ab_report
from .config import DEFAULT_CONFIG
from .datasets import load_gold_cases
from .report import generate_json_report, write_json_report, write_markdown_report
from .runners import create_mock_cases, filter_cases_by_split, run_evaluation


async def main():
    parser = argparse.ArgumentParser(description="RAG/Tool Use Evaluation System")
    parser.add_argument(
        "--cases",
        type=Path,
        default=DEFAULT_CONFIG.cases_path,
        help="Path to gold cases JSONL file"
    )
    parser.add_argument(
        "--mode",
        type=str,
        default=DEFAULT_CONFIG.mode,
        choices=["legacy", "tooluse", "both", "mock"],
        help="Evaluation mode: legacy, tooluse, both, or mock"
    )
    parser.add_argument(
        "--split",
        type=str,
        default=DEFAULT_CONFIG.split,
        choices=["dev", "test", "regression"],
        help="Dataset split to evaluate"
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=DEFAULT_CONFIG.limit,
        help="Limit number of cases to evaluate (None for all)"
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_CONFIG.output_dir,
        help="Directory to write reports"
    )
    parser.add_argument(
        "--fail-on-threshold",
        action="store_true",
        default=DEFAULT_CONFIG.fail_on_threshold,
        help="Exit with non-zero code if thresholds are not met"
    )
    parser.add_argument(
        "--compare-versions",
        nargs=2,
        metavar=("VERSION_A", "VERSION_B"),
        help="对比两个索引版本的检索质量（A/B 测试）"
    )

    args = parser.parse_args()

    # A/B 版本对比模式
    if args.compare_versions:
        version_a, version_b = args.compare_versions
        print(f"Starting A/B comparison: {version_a} vs {version_b}")

        # 加载测试用例
        if args.mode == "mock":
            gold_cases = create_mock_cases(args.limit or 5)
        else:
            gold_cases = load_gold_cases(args.cases)
            gold_cases = filter_cases_by_split(gold_cases, args.split)
            if args.limit:
                gold_cases = gold_cases[:args.limit]

        if not gold_cases:
            print("No cases to evaluate!")
            return 1

        # 运行 A/B 对比
        report = await run_ab_comparison(
            version_a=version_a,
            version_b=version_b,
            gold_cases=gold_cases,
            mode=args.mode if args.mode != "mock" else "legacy",
        )

        # 写入报告
        write_ab_report(report, args.output_dir)
        print(f"\nA/B comparison report written to: {args.output_dir}")
        return 0

    print(f"Starting RAG evaluation in {args.mode} mode...")
    print(f"Dataset: {args.cases}")
    print(f"Split: {args.split}")
    print(f"Output directory: {args.output_dir}")

    # Load gold cases based on mode
    if args.mode == "mock":
        print("Using mock cases for smoke testing...")
        gold_cases = create_mock_cases(args.limit or 5)
    else:
        print(f"Loading gold cases from {args.cases}...")
        gold_cases = load_gold_cases(args.cases)
        print(f"Loaded {len(gold_cases)} gold cases")

        # Filter by split
        gold_cases = filter_cases_by_split(gold_cases, args.split)
        print(f"After split filter ({args.split}): {len(gold_cases)} cases")

        # Apply limit
        if args.limit:
            gold_cases = gold_cases[:args.limit]
            print(f"After limit ({args.limit}): {len(gold_cases)} cases")

    if not gold_cases:
        print("No cases to evaluate!")
        return 1

    # Run evaluation
    print(f"Running evaluation on {len(gold_cases)} cases...")
    results = await run_evaluation(gold_cases, args.mode, args.limit)

    print(f"Evaluation completed! Generated {len(results)} results")

    # Generate report
    print("Generating reports...")
    report = generate_json_report(
        results=results,
        gold_cases=gold_cases,
        mode=args.mode,
        dataset_path=str(args.cases),
        split=args.split
    )

    # Write reports
    json_report_path = args.output_dir / "rag_eval_report.json"
    md_report_path = args.output_dir / "rag_eval_report.md"

    write_json_report(report, json_report_path)
    write_markdown_report(report, md_report_path)

    print(f"JSON report written to: {json_report_path}")
    print(f"Markdown report written to: {md_report_path}")

    # Check thresholds and decide exit code
    if args.fail_on_threshold and not report["thresholds"]["passed"]:
        print("\nThreshold check FAILED:")
        for violation in report["thresholds"]["violations"]:
            print(f"  - {violation['description']}: actual {violation['actual']}, threshold {violation['threshold']}")
        return 1
    elif args.fail_on_threshold:
        print("\nAll thresholds PASSED")

    print("\nEvaluation completed successfully!")
    return 0


if __name__ == "__main__":
    exit_code = asyncio.run(main())
    sys.exit(exit_code)
