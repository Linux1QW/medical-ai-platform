#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""RRF 融合权重网格搜索调参脚本

基于 golden_set.json，对一组候选 RRF 权重组合逐一跑检索并计算指标，
按主指标（默认 ndcg@10）排序输出，帮助锁定当前索引下的最优融合权重。

注意：
- RRF 加权融合走三路 hybrid_recall，因此本脚本默认使用 --retriever mqe
  （tiered_retrieve → hybrid_recall → weighted_rrf）。两路 hybrid 走的是
  非加权 RRF，不受这些权重影响。
- 调参期间自动关闭检索缓存，避免各组合命中同一缓存导致结果无差异。
- 三路权重仅在 settings.BGE_M3_ENABLED=True 时全部生效；否则仅 BM25/Dense
  两项按归一化参与，可仍用于观察 BM25/Dense 配比影响。

用法：
    cd backend
    python scripts/eval/tune_weights.py
    python scripts/eval/tune_weights.py --primary ndcg@5 --top-k 10
"""

import argparse
import asyncio
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent))

# Windows 控制台默认 GBK，无法编码 emoji/部分字符；统一改为 UTF-8 输出避免崩溃
try:
    sys.stdout.reconfigure(encoding="utf-8")
    sys.stderr.reconfigure(encoding="utf-8")
except Exception:
    pass

DEFAULT_GOLDEN = Path(__file__).resolve().parent / "golden_set.json"

# 候选权重组合 [BM25, Dense, Sparse]（和默认约定一致，sum≈1）
WEIGHT_GRID = [
    (0.30, 0.45, 0.25),   # 当前默认
    (0.20, 0.60, 0.20),   # 更偏语义
    (0.40, 0.40, 0.20),   # BM25/Dense 均衡
    (0.25, 0.55, 0.20),
    (0.35, 0.50, 0.15),
    (0.50, 0.35, 0.15),   # 更偏关键词
]


async def _score_combo(weights, golden, retriever, top_k):
    """在给定权重下跑完整评估集，返回聚合指标"""
    from evaluate_retrieval import _build_ranked_and_relevant, _run_retriever

    from app.core.config import settings
    from app.services.rag.eval_metrics import aggregate, evaluate_case

    settings.RRF_WEIGHT_BM25 = weights[0]
    settings.RRF_WEIGHT_DENSE = weights[1]
    settings.RRF_WEIGHT_SPARSE = weights[2]

    k_values = golden.get("k_values", [1, 3, 5, 10])
    recall_n = max(max(k_values), top_k)
    per_case = []
    for case in golden.get("cases", []):
        groups = case.get("relevant_source_contains", [])
        try:
            evidences = await _run_retriever(retriever, case.get("query", ""), recall_n)
        except Exception:
            evidences = []
        ranked_ids, relevant_ids = _build_ranked_and_relevant(evidences, groups)
        per_case.append(evaluate_case(ranked_ids, relevant_ids, k_values=tuple(k_values)))
    return aggregate(per_case)


async def tune(golden_path, retriever, top_k, primary):
    from app.core.config import settings
    from app.services.rag.medical_store import get_medical_store

    store = get_medical_store()
    if store.collection is None or store.count() == 0:
        print("⚠️  医学知识库为空，跳过调参（请先构建索引）。")
        return 0

    # 关闭缓存，保证各组合独立评估
    cache_was_enabled = settings.RETRIEVAL_CACHE_ENABLED
    settings.RETRIEVAL_CACHE_ENABLED = False

    with open(golden_path, "r", encoding="utf-8") as f:
        golden = json.load(f)

    print(f"{'='*70}")
    print(f"RRF 权重网格搜索 | 检索器={retriever} | 主指标={primary} | 组合数={len(WEIGHT_GRID)}")
    print(f"知识库文档块={store.count()} | BGE_M3_ENABLED={settings.BGE_M3_ENABLED}")
    print(f"{'='*70}\n")

    results = []
    try:
        for w in WEIGHT_GRID:
            agg = await _score_combo(w, golden, retriever, top_k)
            score = agg.get(primary, 0.0)
            results.append((w, score, agg))
            print(
                f"weights={w}  {primary}={score:.4f}  "
                f"ndcg@10={agg.get('ndcg@10', 0):.4f}  "
                f"recall@10={agg.get('recall@10', 0):.4f}  "
                f"mrr={agg.get('mrr', 0):.4f}"
            )
    finally:
        settings.RETRIEVAL_CACHE_ENABLED = cache_was_enabled

    results.sort(key=lambda x: x[1], reverse=True)
    best_w, best_score, _ = results[0]
    print(f"\n{'='*70}")
    print(f"🏆 最优权重（按 {primary}）：BM25/Dense/Sparse = {best_w}，{primary}={best_score:.4f}")
    print("   如需固定，请在 config.py 设置 RRF_WEIGHT_BM25/DENSE/SPARSE。")
    print(f"{'='*70}")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description="RRF 融合权重网格搜索")
    parser.add_argument("--golden", type=str, default=str(DEFAULT_GOLDEN))
    parser.add_argument("--retriever", choices=["hybrid", "mqe", "base"], default="mqe")
    parser.add_argument("--top-k", type=int, default=10)
    parser.add_argument("--primary", type=str, default="ndcg@10", help="主排序指标")
    args = parser.parse_args()

    golden_path = Path(args.golden)
    if not golden_path.is_absolute():
        golden_path = Path.cwd() / golden_path
    if not golden_path.exists():
        print(f"❌ 评估集不存在：{golden_path}")
        return 1

    try:
        return asyncio.run(tune(golden_path, args.retriever, args.top_k, args.primary))
    except KeyboardInterrupt:
        print("\n已中断。")
        return 130


if __name__ == "__main__":
    sys.exit(main())
