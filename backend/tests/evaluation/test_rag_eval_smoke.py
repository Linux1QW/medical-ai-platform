"""
Smoke test for RAG evaluation system.
"""
import asyncio
import tempfile
import unittest
from pathlib import Path


class TestRagEvalSmoke(unittest.TestCase):

    def setUp(self):
        self.temp_dir = tempfile.mkdtemp()
        self.temp_output_dir = Path(self.temp_dir) / "output"
        self.temp_output_dir.mkdir()

    def test_import_modules(self):
        """Test that all evaluation modules can be imported without errors."""
        try:
            from evaluation import config, datasets, metrics, rag_eval, report, runners  # noqa: F401  # noqa: F401
            from evaluation.datasets import RagEvalResult, RagGoldCase  # noqa: F401  # noqa: F401
            from evaluation.metrics import mrr, ndcg_at_k, recall_at_k  # noqa: F401  # noqa: F401
            print("All evaluation modules imported successfully")
        except ImportError as e:
            self.fail(f"Failed to import evaluation modules: {e}")

    def test_create_mock_cases(self):
        """Test creating mock cases for smoke testing."""
        from evaluation.runners import create_mock_cases

        # Test creating mock cases
        mock_cases = create_mock_cases(3)
        self.assertEqual(len(mock_cases), 3)

        # Check that each case has required fields
        for case in mock_cases:
            self.assertIsNotNone(case.case_id)
            self.assertIsNotNone(case.patient_info)
            self.assertIsNotNone(case.conversation_text)
            self.assertIsNotNone(case.expected_stance)

    def test_run_mock_evaluation(self):
        """Test running a mock evaluation."""
        from evaluation.runners import create_mock_cases, run_evaluation

        # Create mock cases
        mock_cases = create_mock_cases(2)

        # Run mock evaluation
        async def run_test():
            results = await run_evaluation(mock_cases, "mock", limit=2)
            return results

        results = asyncio.run(run_test())

        # Check that results were generated
        self.assertEqual(len(results), 2)
        for result in results:
            self.assertIsNotNone(result.case_id)
            self.assertIsNotNone(result.mode)
            self.assertEqual(result.mode, "mock")

    def test_generate_reports(self):
        """Test generating reports from mock results."""

        from evaluation.report import generate_json_report, generate_markdown_report
        from evaluation.runners import create_mock_cases, run_evaluation

        # Create mock cases
        mock_cases = create_mock_cases(2)

        # Run mock evaluation
        async def run_test():
            results = await run_evaluation(mock_cases, "mock", limit=2)
            return results

        results = asyncio.run(run_test())

        # Generate reports
        report = generate_json_report(
            results=results,
            gold_cases=mock_cases,
            mode="mock",
            dataset_path="mock_dataset.jsonl",
            split="dev"
        )

        # Check that report was generated
        self.assertIn("timestamp", report)
        self.assertIn("metrics", report)
        self.assertIn("thresholds", report)

        # Generate markdown report
        markdown_content = generate_markdown_report(report)
        self.assertIn("# RAG / Tool Use 评估报告", markdown_content)
        self.assertIn("## 概述", markdown_content)
        self.assertIn("mock", markdown_content)

    def tearDown(self):
        import shutil
        shutil.rmtree(self.temp_dir, ignore_errors=True)


if __name__ == '__main__':
    unittest.main()
