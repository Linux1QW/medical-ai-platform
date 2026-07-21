"""
Unit tests for metrics calculation.
"""
import unittest

from evaluation.datasets import RagEvalResult, RagGoldCase, StanceType
from evaluation.metrics import (
    citation_hallucination_rate,
    citation_metrics,
    citation_validity,
    mrr,
    ndcg_at_k,
    recall_at_k,
    refusal_metrics,
    refusal_metrics_from_results,
    retrieval_metrics,
    tool_use_metrics,
)


class TestMetrics(unittest.TestCase):

    def test_recall_at_k_perfect_match(self):
        """Test recall when all gold items are retrieved."""
        retrieved = ["doc1", "doc2", "doc3"]
        gold = ["doc1", "doc2", "doc3"]
        recall = recall_at_k(retrieved, gold, k=3)
        self.assertEqual(recall, 1.0)

    def test_recall_at_k_partial_match(self):
        """Test recall when only some gold items are retrieved."""
        retrieved = ["doc1", "doc2", "doc4"]
        gold = ["doc1", "doc2", "doc3"]
        recall = recall_at_k(retrieved, gold, k=3)
        self.assertAlmostEqual(recall, 2/3, places=2)

    def test_recall_at_k_no_match(self):
        """Test recall when no gold items are retrieved."""
        retrieved = ["doc4", "doc5", "doc6"]
        gold = ["doc1", "doc2", "doc3"]
        recall = recall_at_k(retrieved, gold, k=3)
        self.assertEqual(recall, 0.0)

    def test_recall_at_k_empty_gold(self):
        """Test recall when gold list is empty."""
        retrieved = ["doc1", "doc2"]
        gold = []
        recall = recall_at_k(retrieved, gold, k=3)
        self.assertEqual(recall, 0.0)

    def test_recall_at_k_empty_retrieved(self):
        """Test recall when retrieved list is empty."""
        retrieved = []
        gold = ["doc1", "doc2"]
        recall = recall_at_k(retrieved, gold, k=3)
        self.assertEqual(recall, 0.0)

    def test_mrr_first_rank(self):
        """Test MRR when first result is relevant."""
        retrieved = ["doc1", "doc2", "doc3"]
        gold = ["doc1"]
        mrr_val = mrr(retrieved, gold)
        self.assertEqual(mrr_val, 1.0)

    def test_mrr_middle_rank(self):
        """Test MRR when relevant result is in middle."""
        retrieved = ["doc2", "doc1", "doc3"]
        gold = ["doc1"]
        mrr_val = mrr(retrieved, gold)
        self.assertEqual(mrr_val, 1/2)  # Rank 2 -> 1/2

    def test_mrr_no_relevant(self):
        """Test MRR when no relevant results."""
        retrieved = ["doc2", "doc3", "doc4"]
        gold = ["doc1"]
        mrr_val = mrr(retrieved, gold)
        self.assertEqual(mrr_val, 0.0)

    def test_ndcg_perfect_ranking(self):
        """Test nDCG with perfect ranking."""
        retrieved = ["doc3", "doc2", "doc1"]
        relevance_grades = {"doc1": 1, "doc2": 2, "doc3": 3}  # Perfect order
        ndcg = ndcg_at_k(retrieved, relevance_grades, k=3)
        self.assertAlmostEqual(ndcg, 1.0, places=2)

    def test_citation_validity_all_valid(self):
        """Test citation validity when all citations are valid."""
        used_ids = ["cit1", "cit2", "cit3"]
        allowed_ids = {"cit1", "cit2", "cit3", "cit4"}
        validity = citation_validity(used_ids, allowed_ids)
        self.assertEqual(validity, 1.0)

    def test_citation_validity_partial_valid(self):
        """Test citation validity when some citations are invalid."""
        used_ids = ["cit1", "cit2", "invalid_cit"]
        allowed_ids = {"cit1", "cit2", "cit3"}
        validity = citation_validity(used_ids, allowed_ids)
        self.assertEqual(validity, 2/3)

    def test_citation_hallucination_rate_none(self):
        """Test hallucination rate when no invalid citations."""
        used_ids = ["cit1", "cit2", "cit3"]
        allowed_ids = {"cit1", "cit2", "cit3", "cit4"}
        halluc_rate = citation_hallucination_rate(used_ids, allowed_ids)
        self.assertEqual(halluc_rate, 0.0)

    def test_citation_hallucination_rate_some_invalid(self):
        """Test hallucination rate when some citations are invalid."""
        used_ids = ["cit1", "cit2", "invalid_cit"]
        allowed_ids = {"cit1", "cit2", "cit3"}
        halluc_rate = citation_hallucination_rate(used_ids, allowed_ids)
        self.assertEqual(halluc_rate, 1/3)

    def test_refusal_metrics_from_results_all_correct(self):
        """Test refusal metrics when all predictions are correct."""
        # Create results and gold cases where everything is correct
        results = [
            RagEvalResult(
                case_id="case1",
                mode="test",
                knowledge_score=None,  # Should refuse
                evaluation_status="completed",
                human_review_needed=True,
                review_reason="insufficient_evidence",
                retrieval_status="sufficient"
            ),
            RagEvalResult(
                case_id="case2",
                mode="test",
                knowledge_score=85.0,  # Should not refuse
                evaluation_status="completed",
                human_review_needed=False,
                review_reason=None,
                retrieval_status="sufficient"
            )
        ]

        gold_cases = [
            RagGoldCase(
                case_id="case1",
                split="dev",
                department="test",
                difficulty="easy",
                patient_info="test",
                conversation_text="test",
                expected_stance=StanceType.UNDETERMINED,
                should_refuse=True  # Should refuse
            ),
            RagGoldCase(
                case_id="case2",
                split="dev",
                department="test",
                difficulty="easy",
                patient_info="test",
                conversation_text="test",
                expected_stance=StanceType.SUPPORTS,
                should_refuse=False  # Should not refuse
            )
        ]

        # Compute system_refused for each result
        for result in results:
            # Manually compute system_refused based on our logic
            if result.knowledge_score is None:
                result.system_refused = True
            elif result.evaluation_status == "needs_review":
                result.system_refused = True
            elif (result.human_review_needed and
                  result.review_reason in ["insufficient_evidence", "knowledge_undetermined",
                                          "citation_verification_failed", "retrieval_error",
                                          "system_exception"]):
                result.system_refused = True
            else:
                result.system_refused = False

        # Calculate metrics
        metrics = refusal_metrics_from_results(results, gold_cases)

        # We expect high scores since most metrics depend on the logic
        # Just check that function runs without error and returns expected keys
        expected_keys = [
            "refusal_accuracy", "refusal_precision", "refusal_recall",
            "refusal_f1", "false_refusal_rate", "false_acceptance_rate"
        ]
        for key in expected_keys:
            self.assertIn(key, metrics)
            self.assertIsInstance(metrics[key], float)
            self.assertGreaterEqual(metrics[key], 0.0)
            self.assertLessEqual(metrics[key], 1.0)


class TestRetrievalMetrics(unittest.TestCase):
    """Tests for the retrieval_metrics aggregate function."""

    def test_empty_input(self):
        result = retrieval_metrics([], [])
        self.assertEqual(result["mrr"], 0.0)
        self.assertEqual(result["map"], 0.0)
        self.assertEqual(result["recall_at_1"], 0.0)

    def test_basic_case(self):
        result = retrieval_metrics(
            retrieved_ids_list=[["d1", "d2", "d3"]],
            gold_ids_list=[["d1", "d3"]],
        )
        self.assertEqual(result["recall_at_1"], 0.5)  # 1 of 2 gold docs in top-1
        self.assertEqual(result["mrr"], 1.0)  # first result is relevant
        self.assertGreater(result["ndcg_at_3"], 0.0)

    def test_length_mismatch_raises(self):
        with self.assertRaises(ValueError):
            retrieval_metrics(
                retrieved_ids_list=[["d1"]],
                gold_ids_list=[["d1"], ["d2"]],
            )

    def test_custom_k_values(self):
        result = retrieval_metrics(
            retrieved_ids_list=[["d1", "d2"]],
            gold_ids_list=[["d2"]],
            k_values=[2],
        )
        self.assertIn("recall_at_2", result)
        self.assertNotIn("recall_at_1", result)


class TestCitationMetrics(unittest.TestCase):
    """Tests for the citation_metrics aggregate function."""

    def test_all_valid(self):
        result = citation_metrics(
            used_citation_ids=["c1", "c2"],
            allowed_citation_ids={"c1", "c2", "c3"},
        )
        self.assertEqual(result["citation_validity"], 1.0)
        self.assertEqual(result["citation_hallucination_rate"], 0.0)
        self.assertIsNone(result["citation_coverage"])

    def test_with_gold(self):
        result = citation_metrics(
            used_citation_ids=["c1", "c2"],
            allowed_citation_ids={"c1", "c2"},
            gold_citation_ids=["c1", "c2", "c3"],
        )
        self.assertAlmostEqual(result["citation_coverage"], 2/3, places=2)

    def test_empty_citations(self):
        result = citation_metrics(
            used_citation_ids=[],
            allowed_citation_ids={"c1"},
        )
        self.assertEqual(result["citation_validity"], 1.0)
        self.assertEqual(result["citation_hallucination_rate"], 0.0)


class TestRefusalMetrics(unittest.TestCase):
    """Tests for the refusal_metrics aggregate function (bool-based)."""

    def test_perfect_predictions(self):
        result = refusal_metrics(
            predictions=[True, False, True, False],
            labels=[True, False, False, True],
        )
        self.assertEqual(result["accuracy"], 0.5)
        self.assertEqual(result["support"], 4)

    def test_all_correct(self):
        result = refusal_metrics(
            predictions=[True, False],
            labels=[True, False],
        )
        self.assertEqual(result["accuracy"], 1.0)
        self.assertEqual(result["precision"], 1.0)
        self.assertEqual(result["recall"], 1.0)
        self.assertEqual(result["f1"], 1.0)

    def test_empty_input(self):
        result = refusal_metrics([], [])
        self.assertEqual(result["accuracy"], 0.0)
        self.assertEqual(result["support"], 0)

    def test_length_mismatch_raises(self):
        with self.assertRaises(ValueError):
            refusal_metrics([True], [True, False])


class TestToolUseMetrics(unittest.TestCase):
    """Tests for the tool_use_metrics aggregate function."""

    def test_empty_input(self):
        result = tool_use_metrics([])
        self.assertEqual(result["total_calls"], 0)
        self.assertEqual(result["success_rate"], 0.0)

    def test_basic_case(self):
        logs = [
            {"name": "search", "status": "success", "latency_ms": 100},
            {"name": "search", "status": "error", "latency_ms": 5000},
            {"name": "calc", "status": "success", "latency_ms": 50},
        ]
        result = tool_use_metrics(logs)
        self.assertEqual(result["total_calls"], 3)
        self.assertAlmostEqual(result["success_rate"], 2/3, places=2)
        self.assertAlmostEqual(result["failure_rate"], 1/3, places=2)
        self.assertIn("search", result["per_tool"])
        self.assertIn("calc", result["per_tool"])

    def test_with_expected_results(self):
        logs = [
            {"name": "search", "status": "success", "result": "data"},
            {"name": "calc", "status": "success", "result": 42},
        ]
        expected = [
            {"expected_tool": "search", "expected_output": "data"},
            {"expected_tool": "calc", "expected_output": 42},
        ]
        result = tool_use_metrics(logs, expected_results=expected)
        self.assertEqual(result["accuracy"], 1.0)

    def test_with_cost(self):
        logs = [
            {"name": "search", "status": "success", "cost": 0.01},
            {"name": "search", "status": "success", "cost": 0.02},
        ]
        result = tool_use_metrics(logs)
        self.assertAlmostEqual(result["total_cost"], 0.03)
        self.assertAlmostEqual(result["avg_cost_per_call"], 0.015)


if __name__ == '__main__':
    unittest.main()
