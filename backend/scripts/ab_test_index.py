#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""索引版本 A/B 对比测试

用法：
    cd backend
    python scripts/ab_test_index.py --version-a rag-v1 --version-b rag-v2
    python scripts/ab_test_index.py --version-a rag-v1 --version-b rag-v2 --cases evaluation/rag_cases/rag_gold_cases.jsonl
    python scripts/ab_test_index.py --version-a rag-v1 --version-b rag-v2 --limit 10 --output-dir reports/ab
"""

import argparse
import asyncio
import logging
import sys
from pathlib import Path

# 添加项目根目录到 path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


async def run_ab_comparison_standalone(
    version_a: str,
    version_b: str,
    cases_path: str = None,
    split: str = "dev",
    limit: int = None,
    mode: str = "legacy",
    output_dir: str = None,
) -> dict:
    """对比两个索引版本的检索质量（独立脚本入口）。

    Args:
        version_a: 版本 A 标识。
        version_b: 版本 B 标识。
        cases_path: 评测数据路径，为 None 时使用默认路径。
        split: 数据集分割。
        limit: 限制样本数。
        mode: 评估模式 ("legacy" / "tooluse")。
        output_dir: 报告输出目录。

    Returns:
        对比报告字典。
    """
    from evaluation.ab_compare import run_ab_comparison, write_ab_report
    from evaluation.datasets import load_gold_cases
    from evaluation.runners import filter_cases_by_split

    # 加载测试用例
    if cases_path is None:
        cases_path = str(
            Path(__file__).resolve().parent.parent
            / "evaluation"
            / "rag_cases"
            / "rag_gold_cases.jsonl"
        )

    cases_file = Path(cases_path)
    if not cases_file.exists():
        print(f"错误: 评测数据文件不存在: {cases_file}")
        sys.exit(1)

    print(f"加载评测数据: {cases_file}")
    gold_cases = load_gold_cases(cases_file)
    print(f"共加载 {len(gold_cases)} 条用例")

    # 按 split 过滤
    gold_cases = filter_cases_by_split(gold_cases, split)
    print(f"split={split} 过滤后: {len(gold_cases)} 条")

    if limit:
        gold_cases = gold_cases[:limit]
        print(f"limit={limit}: {len(gold_cases)} 条")

    if not gold_cases:
        print("错误: 没有可评估的用例!")
        sys.exit(1)

    # 运行 A/B 对比
    report = await run_ab_comparison(
        version_a=version_a,
        version_b=version_b,
        gold_cases=gold_cases,
        mode=mode,
    )

    # 输出报告
    if output_dir:
        out_path = Path(output_dir)
    else:
        out_path = Path(__file__).resolve().parent.parent / "reports" / "ab"

    write_ab_report(report, out_path)

    # 打印摘要
    _print_summary(report)

    return report


def _print_summary(report: dict) -> None:
    """打印对比摘要到控制台。"""
    print(f"\n{'=' * 60}")
    print(f"A/B 对比报告: {report['version_a']} vs {report['version_b']}")
    print(f"{'=' * 60}")
    print(f"样本数: {report['total_cases']}")
    print(f"模式: {report.get('mode', 'legacy')}")
    print()

    # 汇总
    summary = report.get("summary", {})
    sa = summary.get("version_a", {})
    sb = summary.get("version_b", {})

    print(f"{'指标':<25} {'版本A':>12} {'版本B':>12} {'Delta':>12}")
    print(f"{'-' * 61}")

    def _fmt(val, digits=2):
        if val is None:
            return "N/A"
        return f"{val:.{digits}f}"

    def _delta(a, b, digits=2):
        if a is None or b is None:
            return "N/A"
        d = b - a
        sign = "+" if d >= 0 else ""
        return f"{sign}{d:.{digits}f}"

    print(f"{'avg_knowledge_score':<25} {_fmt(sa.get('avg_knowledge_score')):>12} {_fmt(sb.get('avg_knowledge_score')):>12} {_delta(sa.get('avg_knowledge_score'), sb.get('avg_knowledge_score')):>12}")
    print(f"{'avg_latency_ms':<25} {_fmt(sa.get('avg_latency_ms'), 0):>12} {_fmt(sb.get('avg_latency_ms'), 0):>12} {_delta(sa.get('avg_latency_ms'), sb.get('avg_latency_ms'), 0):>12}")
    print(f"{'total_elapsed_s':<25} {_fmt(sa.get('total_elapsed_seconds'), 1):>12} {_fmt(sb.get('total_elapsed_seconds'), 1):>12} {_delta(sa.get('total_elapsed_seconds'), sb.get('total_elapsed_seconds'), 1):>12}")
    print(f"{'error_count':<25} {sa.get('error_count', 0):>12} {sb.get('error_count', 0):>12}")

    # 详细指标
    mc = report.get("metrics_comparison", {})
    if mc:
        print(f"\n{'详细指标':=^60}")
        print(f"{'指标':<30} {'版本A':>10} {'版本B':>10} {'Delta':>10}")
        print(f"{'-' * 60}")
        for key, vals in sorted(mc.items()):
            va = vals.get("version_a", "N/A")
            vb = vals.get("version_b", "N/A")
            delta = vals.get("delta", "N/A")

            def _fval(v):
                if isinstance(v, float):
                    return f"{v:.4f}"
                return str(v)

            def _fdelta(v):
                if isinstance(v, float):
                    sign = "+" if v >= 0 else ""
                    return f"{sign}{v:.4f}"
                return str(v)

            print(f"{key:<30} {_fval(va):>10} {_fval(vb):>10} {_fdelta(delta):>10}")

    # 推荐操作
    rec = report.get("recommendation", "manual_review")
    rec_desc = {
        "promote": "版本 B 显著优于 A，建议切换",
        "reject": "版本 B 劣于 A，不建议切换",
        "manual_review": "差异不显著，需人工判断",
    }
    print(f"\n推荐操作: {rec} — {rec_desc.get(rec, '')}")


def main():
    parser = argparse.ArgumentParser(
        description="索引版本 A/B 对比测试",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例:
    python scripts/ab_test_index.py --version-a rag-v1 --version-b rag-v2
    python scripts/ab_test_index.py --version-a rag-v1 --version-b rag-v2 --limit 10
    python scripts/ab_test_index.py --version-a rag-v1 --version-b rag-v2 --mode tooluse
        """,
    )
    parser.add_argument("--version-a", required=True, help="版本 A（如 rag-v1）")
    parser.add_argument("--version-b", required=True, help="版本 B（如 rag-v2）")
    parser.add_argument("--cases", default=None, help="评测数据路径")
    parser.add_argument("--split", default="dev", choices=["dev", "test", "regression"], help="数据集分割")
    parser.add_argument("--limit", type=int, default=None, help="限制样本数")
    parser.add_argument("--mode", default="legacy", choices=["legacy", "tooluse"], help="评估模式")
    parser.add_argument("--output-dir", default=None, help="报告输出目录")

    args = parser.parse_args()

    report = asyncio.run(run_ab_comparison_standalone(
        version_a=args.version_a,
        version_b=args.version_b,
        cases_path=args.cases,
        split=args.split,
        limit=args.limit,
        mode=args.mode,
        output_dir=args.output_dir,
    ))

    # 根据推荐返回退出码
    if report["recommendation"] == "reject":
        print("\n版本 B 劣于 A，退出码 1")
        sys.exit(1)
    elif report["recommendation"] == "promote":
        print("\n版本 B 优于 A，建议切换！")
        sys.exit(0)
    else:
        print("\n差异不显著，需人工判断")
        sys.exit(0)


if __name__ == "__main__":
    main()
