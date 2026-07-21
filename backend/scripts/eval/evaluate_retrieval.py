#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""RAG 检索质量回归跑分脚本

基于 golden_set.json 对检索管线做 ground-truth 评估，输出 recall@k / precision@k /
ndcg@k / MRR 等指标，用于量化后续优化（如 metadata 预过滤、重排策略调整）的收益。

相关性判定：检索结果的 source 文件名若包含某 case 的 relevant_source_contains 中
任一子串，即判定该结果相关，并映射到对应的"期望来源组"id；不相关结果映射为唯一 token，
从而复用 eval_metrics 中基于 id 集合的排序指标。

用法：
    cd backend
    python scripts/eval/evaluate_retrieval.py
    python scripts/eval/evaluate_retrieval.py --retriever mqe --top-k 10
    python scripts/eval/evaluate_retrieval.py --golden scripts/eval/golden_set.json --output reports/retrieval_eval.json

检索器（--retriever）：
    hybrid  混合检索（BM25 + 向量 + RRF + rerank，默认）
    mqe     多查询扩展 + 分级检索
    base    仅向量检索（最朴素基线）

空库或缺少 API Key 时优雅跳过（退出码 0，不视为失败）。
"""

import argparse
import asyncio
import json
import logging
import sys
from pathlib import Path
from typing import Dict, List

# 添加 backend 根目录到 path：scripts/eval/ → scripts/ → backend/
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

logging.basicConfig(
    level=logging.WARNING,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)

DEFAULT_GOLDEN = Path(__file__).resolve().parent / "golden_set.json"


def _match_group(source: str, groups: List[str]) -> str:
    """返回 source 命中的期望来源组子串；未命中返回空串"""
    if not source:
        return ""
    for sub in groups:
        if sub and sub in source:
            return sub
    return ""


def _build_ranked_and_relevant(evidences: List[Dict], groups: List[str]):
    """将检索结果映射为 (ranked_ids, relevant_ids)

    - 命中期望来源组的结果 → 映射为该组子串（组 id），视为相关
    - 未命中的结果 → 映射为唯一 token（irrelevant#N），不属于 relevant 集
    - relevant_ids 为该 case 期望的全部来源组（groups 去重集合）
    """
    ranked_ids: List[str] = []
    for idx, ev in enumerate(evidences):
        group = _match_group(ev.get("source", ""), groups)
        ranked_ids.append(group if group else f"__irrelevant__#{idx}")
    relevant_ids = {g for g in groups if g}
    return ranked_ids, relevant_ids


async def _run_retriever(retriever: str, query: str, top_k: int) -> List[Dict]:
    """按名称调度检索器"""
    from app.services.rag.retriever import (
        hybrid_retrieve,
        retrieve_medical_evidence,
        retrieve_with_mqe,
    )

    if retriever == "mqe":
        return await retrieve_with_mqe(query, top_k=top_k)
    if retriever == "base":
        return await retrieve_medical_evidence(query, top_k=top_k)
    # 默认 hybrid
    return await hybrid_retrieve(query, top_k=top_k)


async def evaluate(
    golden_path: Path,
    retriever: str,
    top_k: int,
    output_path: Path = None,
) -> int:
    """执行评估，返回进程退出码"""
    from app.services.rag.eval_metrics import aggregate, evaluate_case
    from app.services.rag.medical_store import get_medical_store

    # ── 前置检查：空库优雅跳过 ──
    store = get_medical_store()
    if store.collection is None or store.count() == 0:
        print("⚠️  医学知识库为空，跳过检索质量评估（请先构建索引）。")
        return 0

    with open(golden_path, "r", encoding="utf-8") as f:
        golden = json.load(f)

    k_values = golden.get("k_values", [1, 3, 5, 10])
    cases = golden.get("cases", [])
    if not cases:
        print("⚠️  评估集为空，无可评估的 case。")
        return 0

    # 召回数量取 max(k_values, top_k)，保证 ndcg@10 等有足够候选
    recall_n = max(max(k_values), top_k)

    print(f"{'='*70}")
    print(f"RAG 检索质量评估 | 检索器={retriever} | top_k={recall_n} | 用例数={len(cases)}")
    print(f"知识库文档块={store.count()} | 评估集={golden_path.name}")
    print(f"{'='*70}\n")

    per_case_metrics: List[Dict[str, float]] = []
    per_case_report: List[Dict] = []

    for case in cases:
        cid = case.get("id", "?")
        query = case.get("query", "")
        groups = case.get("relevant_source_contains", [])
        try:
            evidences = await _run_retriever(retriever, query, recall_n)
        except Exception as e:  # 单个 case 失败不影响整体
            logger.warning(f"case={cid} 检索失败: {e}")
            evidences = []

        ranked_ids, relevant_ids = _build_ranked_and_relevant(evidences, groups)
        metrics = evaluate_case(ranked_ids, relevant_ids, k_values=tuple(k_values))
        per_case_metrics.append(metrics)

        hit_sources = [ev.get("source", "") for ev in evidences[:5]]
        per_case_report.append(
            {
                "id": cid,
                "query": query,
                "metrics": metrics,
                "top5_sources": hit_sources,
            }
        )

        max_k = max(k_values)
        print(
            f"[{cid:<18}] hit@{max_k}={metrics.get(f'hit@{max_k}', 0):.2f} "
            f"recall@{max_k}={metrics.get(f'recall@{max_k}', 0):.2f} "
            f"ndcg@{max_k}={metrics.get(f'ndcg@{max_k}', 0):.2f} "
            f"mrr={metrics.get('mrr', 0):.3f}"
        )

    agg = aggregate(per_case_metrics)

    print(f"\n{'='*70}")
    print("宏平均指标（Macro-average）")
    print(f"{'='*70}")
    # 按 k 分组打印
    print(f"{'metric':<12}" + "".join(f"@{k:<8}" for k in k_values))
    for name in ("hit", "recall", "precision", "ndcg"):
        row = f"{name:<12}"
        for k in k_values:
            row += f"{agg.get(f'{name}@{k}', 0):<9.4f}"
        print(row)
    print(f"\nMRR = {agg.get('mrr', 0):.4f}")
    print(f"{'='*70}")

    if output_path:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        report = {
            "retriever": retriever,
            "top_k": recall_n,
            "k_values": k_values,
            "num_cases": len(cases),
            "collection_count": store.count(),
            "aggregate": agg,
            "cases": per_case_report,
        }
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(report, f, ensure_ascii=False, indent=2)
        print(f"\n📄 详细报告已写入：{output_path}")

    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description="RAG 检索质量回归跑分")
    parser.add_argument(
        "--golden",
        type=str,
        default=str(DEFAULT_GOLDEN),
        help="评估集 JSON 路径（默认 scripts/eval/golden_set.json）",
    )
    parser.add_argument(
        "--retriever",
        choices=["hybrid", "mqe", "base"],
        default="hybrid",
        help="检索器类型（默认 hybrid）",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=10,
        help="召回条数（默认 10，最终取 max(k_values, top_k)）",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="可选：详细报告 JSON 输出路径",
    )
    args = parser.parse_args()

    golden_path = Path(args.golden)
    if not golden_path.is_absolute():
        golden_path = Path.cwd() / golden_path
    if not golden_path.exists():
        print(f"❌ 评估集不存在：{golden_path}")
        return 1

    output_path = Path(args.output) if args.output else None

    try:
        return asyncio.run(
            evaluate(golden_path, args.retriever, args.top_k, output_path)
        )
    except KeyboardInterrupt:
        print("\n已中断。")
        return 130


if __name__ == "__main__":
    sys.exit(main())
