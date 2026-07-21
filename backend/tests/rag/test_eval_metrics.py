# -*- coding: utf-8 -*-
"""RAG 检索质量指标纯函数单测（eval_metrics）

覆盖 hit@k / recall@k / precision@k / MRR / DCG / nDCG / evaluate_case / aggregate，
均为无 IO 的确定性数学断言。
"""

import math

from app.services.rag.eval_metrics import (
    aggregate,
    dcg_at_k,
    evaluate_case,
    hit_at_k,
    ndcg_at_k,
    precision_at_k,
    recall_at_k,
    reciprocal_rank,
)


class TestHitAtK:
    def test_hit_within_k(self):
        assert hit_at_k(["a", "b", "c"], {"c"}, 3) == 1.0

    def test_miss_beyond_k(self):
        assert hit_at_k(["a", "b", "c"], {"c"}, 2) == 0.0

    def test_empty_relevant(self):
        assert hit_at_k(["a", "b"], set(), 3) == 0.0

    def test_first_position(self):
        assert hit_at_k(["x", "y"], {"x"}, 1) == 1.0


class TestRecallAtK:
    def test_partial_recall(self):
        # 2 个相关，top-3 命中 1 个 → 0.5
        assert recall_at_k(["a", "b", "c"], {"a", "z"}, 3) == 0.5

    def test_full_recall(self):
        assert recall_at_k(["a", "b"], {"a", "b"}, 2) == 1.0

    def test_dedup_ranked(self):
        # 重复项不应把 recall 抬高
        assert recall_at_k(["a", "a", "a"], {"a", "b"}, 3) == 0.5

    def test_empty_relevant(self):
        assert recall_at_k(["a"], set(), 3) == 0.0


class TestPrecisionAtK:
    def test_precision(self):
        # top-4 中命中 2 个 → 0.5
        assert precision_at_k(["a", "x", "b", "y"], {"a", "b"}, 4) == 0.5

    def test_k_zero(self):
        assert precision_at_k(["a"], {"a"}, 0) == 0.0

    def test_empty_relevant(self):
        assert precision_at_k(["a"], set(), 3) == 0.0


class TestReciprocalRank:
    def test_first_hit_rank_2(self):
        assert reciprocal_rank(["x", "a", "b"], {"a"}) == 0.5

    def test_first_position(self):
        assert reciprocal_rank(["a", "b"], {"a"}) == 1.0

    def test_no_hit(self):
        assert reciprocal_rank(["x", "y"], {"a"}) == 0.0


class TestDcgNdcg:
    def test_dcg_first_position(self):
        # rank1 命中：1 / log2(2) = 1.0
        assert dcg_at_k(["a", "x"], {"a"}, 2) == 1.0

    def test_dcg_second_position(self):
        # rank2 命中：1 / log2(3)
        assert math.isclose(dcg_at_k(["x", "a"], {"a"}, 2), 1.0 / math.log2(3))

    def test_ndcg_perfect_order(self):
        # 相关文档全在最前 → nDCG = 1.0
        assert math.isclose(ndcg_at_k(["a", "b", "x"], {"a", "b"}, 3), 1.0)

    def test_ndcg_suboptimal_order(self):
        # 相关项靠后，nDCG 应 < 1
        val = ndcg_at_k(["x", "y", "a"], {"a"}, 3)
        assert 0.0 < val < 1.0

    def test_ndcg_no_hit(self):
        assert ndcg_at_k(["x", "y"], {"a"}, 2) == 0.0

    def test_ndcg_empty_relevant(self):
        assert ndcg_at_k(["a"], set(), 3) == 0.0


class TestEvaluateCase:
    def test_keys_present(self):
        m = evaluate_case(["a", "b", "c"], {"a"}, k_values=(1, 3))
        for key in ("mrr", "hit@1", "recall@1", "precision@1", "ndcg@1",
                    "hit@3", "recall@3", "precision@3", "ndcg@3"):
            assert key in m

    def test_values_consistent(self):
        m = evaluate_case(["a", "x", "y"], {"a"}, k_values=(1, 3))
        assert m["hit@1"] == 1.0
        assert m["mrr"] == 1.0
        assert m["precision@3"] == round(1 / 3, 6)


class TestAggregate:
    def test_macro_average(self):
        cases = [
            {"hit@1": 1.0, "mrr": 1.0},
            {"hit@1": 0.0, "mrr": 0.5},
        ]
        agg = aggregate(cases)
        assert agg["hit@1"] == 0.5
        assert agg["mrr"] == 0.75

    def test_empty(self):
        assert aggregate([]) == {}

    def test_missing_key_skipped(self):
        # 键在部分 case 缺失时，仅对存在的取平均
        cases = [{"hit@1": 1.0}, {"recall@1": 0.0}]
        agg = aggregate(cases)
        assert agg["hit@1"] == 1.0
        assert agg["recall@1"] == 0.0
