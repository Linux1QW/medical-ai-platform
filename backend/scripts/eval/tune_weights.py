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
from contextlib import contextmanager
from itertools import product
from pathlib import Path
from typing import Any, Awaitable, Callable, Dict, Iterable, List, Mapping, Optional

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent))

# Windows 控制台默认 GBK，无法编码 emoji/部分字符；统一改为 UTF-8 输出避免崩溃
try:
    sys.stdout.reconfigure(encoding="utf-8")
    sys.stderr.reconfigure(encoding="utf-8")
except Exception:
    pass

DEFAULT_GOLDEN = Path(__file__).resolve().parent / "golden_set.json"

BM25_GRID = {
    "k1": [0.9, 1.2, 1.5],
    "b": [0.5, 0.7, 0.8],
    "heading_boost": [1, 2],
    "entity_boost": [1, 2, 3],
}
RRF_K_GRID = [30, 35, 60]

# 候选权重组合 [BM25, Dense, Sparse]（和默认约定一致，sum≈1）
WEIGHT_GRID = [
    (0.30, 0.45, 0.25),   # 当前默认
    (0.20, 0.60, 0.20),   # 更偏语义
    (0.40, 0.40, 0.20),   # BM25/Dense 均衡
    (0.25, 0.55, 0.20),
    (0.35, 0.50, 0.15),
    (0.50, 0.35, 0.15),   # 更偏关键词
]


def iter_parameter_grid() -> Iterable[Dict[str, Any]]:
    """Yield the complete, deterministic joint BM25/RRF search space."""
    keys = ("k1", "b", "heading_boost", "entity_boost")
    values = [BM25_GRID[key] for key in keys]
    for bm25_values in product(*values):
        for rrf_k in RRF_K_GRID:
            yield {**dict(zip(keys, bm25_values, strict=False)), "rrf_k": rrf_k}


def _primary_score(metrics: Mapping[str, Any], primary: str) -> float:
    value: Any = metrics.get(primary)
    if value is None and isinstance(metrics.get("metrics"), Mapping):
        value = metrics["metrics"].get(primary)
    return float(value) if isinstance(value, (int, float)) else float("-inf")


async def tune_splits(
    dev_cases: List[Mapping[str, Any]],
    test_cases: Optional[List[Mapping[str, Any]]] = None,
    *,
    evaluate_combo: Callable[
        [Mapping[str, Any], List[Mapping[str, Any]], str], Awaitable[Mapping[str, Any]]
    ],
    primary: str = "ndcg@10",
    tuning_split: str = "dev",
) -> Dict[str, Any]:
    """Select parameters on dev and, optionally, evaluate the winner once on test."""
    if tuning_split != "dev":
        raise ValueError("parameter tuning must use the dev split; test is evaluation-only")

    dev_results: List[Dict[str, Any]] = []
    for params in iter_parameter_grid():
        metrics = dict(await evaluate_combo(params, dev_cases, "dev"))
        dev_results.append({"split": "dev", "params": params, "metrics": metrics})

    if not dev_results:
        raise ValueError("dev split produced no tuning candidates")
    best = max(
        enumerate(dev_results),
        key=lambda item: (_primary_score(item[1]["metrics"], primary), -item[0]),
    )[1]
    selected_params = dict(best["params"])

    test_result: Optional[Dict[str, Any]] = None
    if test_cases is not None:
        test_metrics = dict(await evaluate_combo(selected_params, test_cases, "test"))
        test_result = {
            "split": "test",
            "params": selected_params,
            "metrics": test_metrics,
        }

    return {
        "tuning_split": "dev",
        "test_split_evaluated_once": test_result is not None,
        "primary": primary,
        "selected_params": selected_params,
        "selected_dev_result": best,
        "dev_results": dev_results,
        "test_result": test_result,
    }


@contextmanager
def _parameter_context(params: Mapping[str, Any]):
    from app.core.config import settings

    names = {
        "k1": "BM25_K1",
        "b": "BM25_B",
        "heading_boost": "BM25_HEADING_BOOST",
        "entity_boost": "BM25_ENTITY_BOOST",
        "rrf_k": "RRF_K",
    }
    previous = {setting_name: getattr(settings, setting_name) for setting_name in names.values()}
    try:
        for param_name, setting_name in names.items():
            setattr(settings, setting_name, params[param_name])
        yield
    finally:
        for setting_name, value in previous.items():
            setattr(settings, setting_name, value)


@contextmanager
def _candidate_bm25_context(candidate: Any):
    """Route the existing fusion entry point to one parameterized candidate."""
    from app.services.rag.retriever import fusion

    original = fusion.get_bm25_index
    fusion.get_bm25_index = lambda _generation=None: candidate
    try:
        yield
    finally:
        fusion.get_bm25_index = original


def _build_parameterized_index(active_index: Any, params: Mapping[str, Any]) -> Any:
    from app.services.rag.bm25_search import BM25Index

    documents = list(getattr(active_index, "documents", []) or [])
    if not documents:
        raise RuntimeError("active BM25 index has no documents for parameterized tuning")
    candidate = BM25Index()
    with _parameter_context(params):
        candidate.build(documents)
    return candidate


async def _evaluate_parameter_combo(
    params: Mapping[str, Any],
    cases: List[Mapping[str, Any]],
    split: str,
    *,
    active_index: Any,
    retriever: str,
    top_k: int,
) -> Mapping[str, Any]:
    if split not in {"dev", "test"}:
        raise ValueError(f"unsupported evaluation split: {split}")
    from app.core.config import settings

    candidate = _build_parameterized_index(active_index, params)
    golden = {"cases": list(cases), "k_values": [1, 3, 5, 10]}
    cache_was_enabled = settings.RETRIEVAL_CACHE_ENABLED
    settings.RETRIEVAL_CACHE_ENABLED = False
    try:
        with _parameter_context(params), _candidate_bm25_context(candidate):
            aggregate = await _score_combo(
                (
                    settings.RRF_WEIGHT_BM25,
                    settings.RRF_WEIGHT_DENSE,
                    settings.RRF_WEIGHT_SPARSE,
                ),
                golden,
                retriever,
                top_k,
            )
        return aggregate
    finally:
        settings.RETRIEVAL_CACHE_ENABLED = cache_was_enabled


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


def _load_cases(path: Path) -> List[Mapping[str, Any]]:
    with path.open(encoding="utf-8") as golden_file:
        payload = json.load(golden_file)
    cases = payload.get("cases", [])
    if not isinstance(cases, list) or not cases:
        raise ValueError(f"split contains no cases: {path}")
    return cases


async def tune(
    dev_path: Path,
    retriever: str,
    top_k: int,
    primary: str,
    *,
    test_path: Optional[Path] = None,
    output_path: Optional[Path] = None,
) -> int:
    from app.services.rag.bm25_search import get_bm25_index

    active_index = get_bm25_index()
    if not getattr(active_index, "initialized", False) or getattr(active_index, "doc_count", 0) == 0:
        print("BM25 tuning skipped: no initialized active generation (offline verification only).")
        return 0

    dev_cases = _load_cases(dev_path)
    test_cases = _load_cases(test_path) if test_path is not None else None

    async def evaluate_combo(params, cases, split):
        return await _evaluate_parameter_combo(
            params,
            cases,
            split,
            active_index=active_index,
            retriever=retriever,
            top_k=top_k,
        )

    report = await tune_splits(
        dev_cases,
        test_cases,
        evaluate_combo=evaluate_combo,
        primary=primary,
    )
    report["dev_path"] = str(dev_path)
    report["test_path"] = str(test_path) if test_path is not None else None
    report["retriever"] = retriever
    report["top_k"] = top_k
    report["offline_verification"] = False

    if output_path is not None:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(
            json.dumps(report, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        print(f"Tuning report written to {output_path}")
    print(
        "Selected dev parameters: "
        f"{report['selected_params']} ({primary}="
        f"{_primary_score(report['selected_dev_result']['metrics'], primary):.4f})"
    )
    if test_cases is not None:
        print("Test split evaluated exactly once with the selected dev parameters.")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description="Joint BM25/RRF dev tuning with one-shot test evaluation")
    parser.add_argument("--golden", type=Path, help="Backward-compatible alias for --dev-golden")
    parser.add_argument("--dev-golden", type=Path, default=DEFAULT_GOLDEN)
    parser.add_argument("--test-golden", type=Path)
    parser.add_argument("--retriever", choices=["hybrid", "mqe", "base"], default="mqe")
    parser.add_argument("--top-k", type=int, default=10)
    parser.add_argument("--primary", type=str, default="ndcg@10", help="主排序指标")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()

    golden_path = args.golden or args.dev_golden
    if not golden_path.is_absolute():
        golden_path = Path.cwd() / golden_path
    if not golden_path.exists():
        print(f"❌ 评估集不存在：{golden_path}")
        return 1

    test_path = args.test_golden
    if test_path is not None and not test_path.is_absolute():
        test_path = Path.cwd() / test_path
    if test_path is not None and not test_path.exists():
        print(f"❌ test split 不存在：{test_path}")
        return 1
    if args.top_k <= 0:
        parser.error("--top-k must be positive")

    try:
        return asyncio.run(
            tune(
                golden_path,
                args.retriever,
                args.top_k,
                args.primary,
                test_path=test_path,
                output_path=args.output,
            )
        )
    except KeyboardInterrupt:
        print("\n已中断。")
        return 130


if __name__ == "__main__":
    sys.exit(main())
