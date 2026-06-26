"""
Unit tests for report generation.
"""
import unittest
import tempfile
from pathlib import Path
from evaluation.report import (
    generate_json_report,
    generate_markdown_report,
    check_thresholds,
    threshold_checker,
    generate_comparison_report,
    write_comparison_report,
    DEFAULT_THRESHOLDS,
)
from evaluation.datasets import RagGoldCase, RagEvalResult, StanceType


def _make_result(case_id="case_001", mode="tooluse", **kwargs):
    """Helper to build a RagEvalResult with sensible defaults."""
    defaults = dict(
        case_id=case_id,
        mode=mode,
        knowledge_score=85.0,
        evaluation_status="completed",
        human_review_needed=False,
        review_reason=None,
        retrieval_status="sufficient",
        evidence_stance=StanceType.SUPPORTS,
        citation_data=[{"id": "cit1", "text": "test"}],
        rag_trace_data={"retrieved_docs": ["doc1"]},
        tool_trace=[{"name": "search", "status": "success", "latency_ms": 100}],
        latency_ms=1200,
        error=None,
        actual_tool_calls=[{"name": "search", "result": "result"}],
        final_answer_text="Final answer text",
    )
    defaults.update(kwargs)
    return RagEvalResult(**defaults)


def _make_gold_case(case_id="case_001", **kwargs):
    """Helper to build a RagGoldCase with sensible defaults."""
    defaults = dict(
        case_id=case_id,
        split="dev",
        department="内科",
        difficulty="easy",
        patient_info="Test patient",
        conversation_text="Test conversation",
        expected_stance=StanceType.SUPPORTS,
        should_refuse=False,
        expected_score_range=[80, 95],
        expected_tool_calls=[{"name": "search", "params": {"query": "test"}}],
        expected_tool_params={"search": {"query": "test"}},
        expected_final_answer_keywords=["answer", "test"],
    )
    defaults.update(kwargs)
    return RagGoldCase(**defaults)


class TestReport(unittest.TestCase):

    def setUp(self):
        self.temp_dir = tempfile.mkdtemp()

    # ── JSON report ──────────────────────────────────────────────

    def test_generate_json_report_basic(self):
        """Test generating a basic JSON report."""
        results = [_make_result()]
        gold_cases = [_make_gold_case()]

        report = generate_json_report(
            results=results,
            gold_cases=gold_cases,
            mode="tooluse",
            dataset_path="test_dataset.jsonl",
            split="dev",
        )

        # Check that report has expected structure
        self.assertIn("timestamp", report)
        self.assertIn("mode", report)
        self.assertIn("dataset", report)
        self.assertIn("metrics", report)
        self.assertIn("tool_breakdown", report)
        self.assertIn("thresholds", report)

        # Check mode is correct
        self.assertEqual(report["mode"], "tooluse")

        # Check dataset info
        dataset = report["dataset"]
        self.assertEqual(dataset["path"], "test_dataset.jsonl")
        self.assertEqual(dataset["split"], "dev")
        self.assertEqual(dataset["total_samples"], 1)

        # Check that metrics exist
        metrics = report["metrics"]
        self.assertIsInstance(metrics, dict)

        # Check that thresholds are evaluated
        thresholds = report["thresholds"]
        self.assertIn("passed", thresholds)
        self.assertIn("violations", thresholds)

    def test_json_report_new_fields(self):
        """Test that JSON report contains new breakdown fields."""
        results = [_make_result()]
        gold_cases = [_make_gold_case()]

        report = generate_json_report(
            results=results, gold_cases=gold_cases,
            mode="tooluse", dataset_path="test.jsonl", split="dev",
        )

        self.assertIn("breakdown_by_department", report)
        self.assertIn("breakdown_by_difficulty", report)

        # score_range_accuracy should be in metrics
        self.assertIn("score_range_accuracy", report["metrics"])

        # thresholds should have compliance_rate and recommendations
        th = report["thresholds"]
        self.assertIn("compliance_rate", th)
        self.assertIn("recommendations", th)
        self.assertIn("summary_by_level", th)

    # ── Markdown report ──────────────────────────────────────────

    def test_generate_markdown_report_basic(self):
        """Test generating a basic Markdown report."""
        report = {
            "timestamp": "2023-01-01T00:00:00+00:00",
            "mode": "tooluse",
            "dataset": {
                "path": "test_cases.jsonl",
                "split": "dev",
                "total_samples": 1,
                "normal_samples": 1,
                "refusal_samples": 0,
            },
            "metrics": {
                "citation_validity": 1.0,
                "citation_hallucination_rate": 0.0,
                "false_acceptance_rate": 0.0,
                "refusal_accuracy": 1.0,
                "avg_latency_ms": 1200,
                "avg_tool_calls": 2.0,
                "tool_success_rate": 1.0,
                "recall_at_5": 0.8,
            },
            "tool_breakdown": {
                "search_medical_kb": {
                    "calls": 1,
                    "success_rate": 1.0,
                    "avg_latency_ms": 100,
                },
            },
            "failed_cases": [],
            "thresholds": {
                "passed": True,
                "violations": [],
                "compliance_rate": 1.0,
                "recommendations": [],
                "summary_by_level": {},
            },
        }

        markdown_content = generate_markdown_report(report)

        # Check that markdown contains expected sections
        self.assertIn("# RAG / Tool Use 评估报告", markdown_content)
        self.assertIn("## 概述", markdown_content)
        self.assertIn("## 核心门槛", markdown_content)
        self.assertIn("## 检索指标", markdown_content)
        self.assertIn("## 拒答指标", markdown_content)
        self.assertIn("## 引用指标", markdown_content)
        self.assertIn("## Tool Use 指标", markdown_content)

        # Check specific content
        self.assertIn("tooluse", markdown_content)
        self.assertIn("100%", markdown_content)
        self.assertIn("0%", markdown_content)
        self.assertIn("R@5", markdown_content)

    def test_markdown_with_comparison(self):
        """Test markdown report with comparison section."""
        report = {
            "timestamp": "2023-01-01T00:00:00+00:00",
            "mode": "legacy",
            "dataset": {
                "path": "test.jsonl", "split": "dev",
                "total_samples": 10, "normal_samples": 8, "refusal_samples": 2,
            },
            "metrics": {
                "citation_validity": 0.95,
                "avg_latency_ms": 1000,
                "avg_tool_calls": 1.5,
                "tool_success_rate": 0.9,
            },
            "tool_breakdown": {},
            "failed_cases": [],
            "thresholds": {"passed": True, "violations": [], "compliance_rate": 0.9, "recommendations": []},
        }

        comparison = {
            "legacy": {
                "avg_knowledge_score": 80.0,
                "avg_latency_ms": 1000,
                "total_elapsed_seconds": 120.0,
                "error_count": 1,
                "review_needed_count": 2,
            },
            "tooluse": {
                "avg_knowledge_score": 85.0,
                "avg_latency_ms": 1500,
                "total_elapsed_seconds": 180.0,
                "error_count": 0,
                "review_needed_count": 1,
            },
            "delta": {
                "avg_knowledge_score": 5.0,
                "avg_latency_ms": 500.0,
            },
            "query_type_breakdown": {
                "information": {
                    "count": 5,
                    "legacy": {"avg_knowledge_score": 82.0},
                    "tooluse": {"avg_knowledge_score": 88.0},
                },
            },
        }

        md = generate_markdown_report(report, comparison_report=comparison)
        self.assertIn("Legacy RAG vs Tool Use 对比", md)
        self.assertIn("按查询类型对比", md)
        self.assertIn("information", md)

    # ── Threshold checking ───────────────────────────────────────

    def test_threshold_check_logic(self):
        """Test that threshold checking works correctly."""
        # Test passing thresholds
        passing_metrics = {
            "citation_validity": 1.0,
            "citation_hallucination_rate": 0.0,
            "false_acceptance_rate": 0.04,
        }
        result = check_thresholds(passing_metrics)
        self.assertTrue(result["passed"])
        self.assertEqual(len(result["violations"]), 0)

        # Test failing thresholds
        failing_metrics = {
            "citation_validity": 0.9,
            "citation_hallucination_rate": 0.1,
            "false_acceptance_rate": 0.08,
        }
        result = check_thresholds(failing_metrics)
        self.assertFalse(result["passed"])
        self.assertGreater(len(result["violations"]), 0)

    def test_threshold_checker_comprehensive(self):
        """Test threshold_checker with compliance report."""
        metrics = {
            "citation_validity": 1.0,
            "citation_hallucination_rate": 0.03,
            "false_acceptance_rate": 0.02,
            "refusal_accuracy": 0.85,
            "stance_accuracy": 0.75,
            "score_range_accuracy": 0.65,
            "recall_at_5": 0.55,
            "tool_success_rate": 0.90,
        }

        result = threshold_checker(metrics)

        self.assertIn("passed", result)
        self.assertIn("summary_by_level", result)
        self.assertIn("compliance_rate", result)
        self.assertIn("recommendations", result)

        # All P0 should pass
        self.assertTrue(result["passed"])
        self.assertEqual(result["summary_by_level"]["P0"]["failed"], 0)

    def test_threshold_checker_with_violations(self):
        """Test threshold_checker generates recommendations for violations."""
        metrics = {
            "citation_validity": 0.8,
            "citation_hallucination_rate": 0.10,
            "false_acceptance_rate": 0.08,
        }

        result = threshold_checker(metrics)

        self.assertFalse(result["passed"])
        self.assertGreater(len(result["violations"]), 0)
        self.assertGreater(len(result["recommendations"]), 0)
        # P0 violations should cause failure
        self.assertEqual(result["summary_by_level"]["P0"]["failed"], 3)

    # ── Comparison report ────────────────────────────────────────

    def test_generate_comparison_report(self):
        """Test comparison report generation."""
        gold_cases = [
            _make_gold_case("c1"),
            _make_gold_case("c2"),
        ]
        legacy_results = [
            _make_result("c1", mode="legacy", knowledge_score=80.0, latency_ms=1000),
            _make_result("c2", mode="legacy", knowledge_score=75.0, latency_ms=1100),
        ]
        tooluse_results = [
            _make_result("c1", mode="tooluse", knowledge_score=85.0, latency_ms=1500),
            _make_result("c2", mode="tooluse", knowledge_score=82.0, latency_ms=1400),
        ]

        comp = generate_comparison_report(
            legacy_results, tooluse_results, gold_cases,
            legacy_elapsed=2.1, tooluse_elapsed=2.9,
        )

        self.assertIn("comparison", comp)
        self.assertIn("metric_comparison", comp)
        self.assertIn("query_type_breakdown", comp)
        self.assertIn("legacy_report", comp)
        self.assertIn("tooluse_report", comp)

        # Delta should be positive (tooluse > legacy)
        delta = comp["comparison"]["delta"]
        self.assertIsNotNone(delta["avg_knowledge_score"])
        self.assertGreater(delta["avg_knowledge_score"], 0)

    def test_write_comparison_report(self):
        """Test writing comparison report to files."""
        gold_cases = [_make_gold_case("c1")]
        legacy_results = [_make_result("c1", mode="legacy")]
        tooluse_results = [_make_result("c1", mode="tooluse")]

        comp = generate_comparison_report(
            legacy_results, tooluse_results, gold_cases,
        )

        json_path = Path(self.temp_dir) / "comp.json"
        md_path = Path(self.temp_dir) / "comp.md"

        write_comparison_report(comp, json_path=json_path, md_path=md_path)

        self.assertTrue(json_path.exists())
        self.assertTrue(md_path.exists())

        # Verify JSON is valid
        import json
        with open(json_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        self.assertIn("comparison", data)

        # Verify Markdown has content
        with open(md_path, 'r', encoding='utf-8') as f:
            md_content = f.read()
        self.assertIn("评估报告", md_content)


if __name__ == '__main__':
    unittest.main()
