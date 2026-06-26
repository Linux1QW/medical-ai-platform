"""
Regression tests for RAG evaluation with real gold cases dataset.

Covers:
- Default gold cases dataset loading & format conversion
- Legacy mode evaluation pipeline (data load → evaluate → report)
- Tool Use mode evaluation pipeline
- End-to-end CLI invocation (--limit 3)
- Report generation correctness
"""
import asyncio
import json
import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch, AsyncMock

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parents[3]  # backend/tests/evaluation → project root
DEFAULT_GOLD_CASES_PATH = PROJECT_ROOT / "backend" / "evaluation" / "rag_cases" / "rag_gold_cases.jsonl"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _run_async(coro):
    """Run an async function in a new event loop."""
    return asyncio.get_event_loop().run_until_complete(coro) if False else asyncio.run(coro)


def _mock_knowledge_check_result(mode: str = "legacy") -> dict:
    """Return a realistic mock result matching knowledge_agent output schema."""
    return {
        "score": 78.5,
        "knowledge_score": 78.5,
        "evaluation_status": "completed",
        "human_review_needed": False,
        "review_reason": None,
        "retrieval_status": "sufficient",
        "evidence_stance": "supports",
        "citations": [
            {"id": "cit-001", "text": "Mock citation for regression test", "source": "mock-doc-1"}
        ],
        "rag_trace": {
            "queries": ["mock regression query"],
            "retrieved_docs": ["mock-doc-1", "mock-doc-2"],
            "processed_at": "2026-01-01T00:00:00Z",
        },
        "tool_trace": [
            {
                "name": "search_medical_info",
                "status": "success",
                "input": {"query": "mock"},
                "output": {"result": "mock"},
                "latency_ms": 120,
            }
        ] if mode == "tooluse" else [],
        "actual_tool_calls": [
            {"name": "search_medical_info", "params": {"query": "mock"}, "result": "ok"}
        ] if mode == "tooluse" else [],
        "final_answer": "Mock final answer for regression testing.",
    }


# ===========================================================================
# 1. Dataset loading & format conversion
# ===========================================================================

class TestDefaultGoldCasesLoading(unittest.TestCase):
    """验证默认 gold cases 数据集能够正确加载和解析。"""

    def test_default_dataset_exists(self):
        """默认 gold cases 文件应存在。"""
        self.assertTrue(
            DEFAULT_GOLD_CASES_PATH.exists(),
            f"Default gold cases file not found: {DEFAULT_GOLD_CASES_PATH}",
        )

    def test_load_default_gold_cases(self):
        """应能从默认 JSONL 文件加载至少 10 条 gold cases。"""
        from evaluation.datasets import load_gold_cases

        cases = load_gold_cases(DEFAULT_GOLD_CASES_PATH)
        self.assertGreaterEqual(len(cases), 10, "Expected at least 10 gold cases")

    def test_legacy_format_conversion(self):
        """旧版格式字段应被正确转换为 RagGoldCase 兼容格式。"""
        from evaluation.datasets import _convert_legacy_format, RagGoldCase

        legacy_data = {
            "id": "legacy_001",
            "query": "患者男性，55岁，反复胸闷气促3个月",
            "expected_answer": "应考虑心力衰竭的可能",
            "reference_docs": ["内科学-心力衰竭章节"],
            "department": "心血管内科",
            "difficulty": "中等",
            "expected_stance": "提供信息",
            "tool_use_expectation": "retrieval",
            "tags": ["心力衰竭", "症状鉴别"],
            "created_at": "2026-06-23",
        }
        converted = _convert_legacy_format(legacy_data)
        case = RagGoldCase(**converted)

        self.assertEqual(case.case_id, "legacy_001")
        self.assertEqual(case.department, "心血管内科")
        self.assertEqual(case.difficulty, "medium")
        self.assertEqual(case.expected_stance, "supports")
        self.assertFalse(case.should_refuse)
        self.assertIn("胸闷", case.patient_info)
        self.assertIn("胸闷", case.conversation_text)

    def test_new_format_passthrough(self):
        """新版格式数据应直接通过而不需要转换。"""
        from evaluation.datasets import _convert_legacy_format

        new_data = {
            "case_id": "case_001",
            "split": "test",
            "department": "心血管内科",
            "difficulty": "medium",
            "patient_info": "患者信息",
            "conversation_text": "对话内容",
            "expected_stance": "supports",
            "should_refuse": False,
        }
        result = _convert_legacy_format(new_data)
        self.assertEqual(result, new_data)  # Should be unchanged

    def test_refusal_case_conversion(self):
        """拒绝回答类案例的 should_refuse 应为 True。"""
        from evaluation.datasets import _convert_legacy_format, RagGoldCase

        legacy_refusal = {
            "id": "legacy_refusal",
            "query": "请告诉我如何进行非法交易",
            "expected_answer": "抱歉，无法回答",
            "reference_docs": [],
            "department": "伦理与法律",
            "difficulty": "简单",
            "expected_stance": "拒绝回答",
            "tool_use_expectation": "refusal",
            "tags": ["拒绝回答"],
        }
        converted = _convert_legacy_format(legacy_refusal)
        case = RagGoldCase(**converted)
        self.assertTrue(case.should_refuse)
        self.assertEqual(case.expected_stance, "contradicts")

    def test_difficulty_mapping(self):
        """中文难度值应被正确映射为英文枚举值。"""
        from evaluation.datasets import _convert_legacy_format

        for zh, en in [("简单", "easy"), ("中等", "medium"), ("困难", "hard")]:
            data = {"id": "x", "query": "q", "difficulty": zh, "expected_stance": "提供信息",
                    "patient_info": "p", "conversation_text": "c"}
            converted = _convert_legacy_format(data)
            self.assertEqual(converted["difficulty"], en, f"{zh} should map to {en}")

    def test_all_cases_have_required_fields(self):
        """所有加载的案例都应包含必需字段。"""
        from evaluation.datasets import load_gold_cases

        cases = load_gold_cases(DEFAULT_GOLD_CASES_PATH)
        for case in cases:
            self.assertTrue(case.case_id, f"case_id missing in {case}")
            self.assertTrue(case.conversation_text, f"conversation_text missing in {case.case_id}")
            self.assertIn(case.difficulty, ("easy", "medium", "hard"))
            self.assertIn(case.expected_stance, ("supports", "contradicts", "mixed", "undetermined"))
            # patient_info 对于知识类/拒绝类问题可为空，不做强制检查

    def test_gold_relevant_sources_preserved(self):
        """旧版 reference_docs 应被映射到 gold_relevant_sources。"""
        from evaluation.datasets import load_gold_cases

        cases = load_gold_cases(DEFAULT_GOLD_CASES_PATH)
        case1 = cases[0]
        self.assertGreater(len(case1.gold_relevant_sources), 0)
        self.assertIn("内科学", case1.gold_relevant_sources[0])


# ===========================================================================
# 2. Legacy mode evaluation pipeline (mocked LLM)
# ===========================================================================

class TestLegacyModePipeline(unittest.TestCase):
    """Legacy 模式评估流程回归测试（mock LLM 调用）。"""

    def _get_cases(self, limit: int = 3):
        from evaluation.datasets import load_gold_cases

        cases = load_gold_cases(DEFAULT_GOLD_CASES_PATH)
        return cases[:limit]

    @patch("evaluation.runners.run_knowledge_check", new_callable=AsyncMock)
    def test_legacy_evaluation_runs(self, mock_kc):
        """Legacy 模式应能正常执行评估并返回结果。"""
        from evaluation.runners import run_evaluation

        mock_kc.return_value = _mock_knowledge_check_result("legacy")
        cases = self._get_cases(3)

        results = _run_async(run_evaluation(cases, "legacy"))

        self.assertEqual(len(results), len(cases))
        for r in results:
            self.assertEqual(r.mode, "legacy")
            self.assertIsNotNone(r.knowledge_score)
            self.assertEqual(r.evaluation_status, "completed")
            self.assertIsNone(r.error)

    @patch("evaluation.runners.run_knowledge_check", new_callable=AsyncMock)
    def test_legacy_report_generation(self, mock_kc):
        """Legacy 模式应能生成有效的 JSON 报告。"""
        from evaluation.runners import run_evaluation
        from evaluation.report import generate_json_report

        mock_kc.return_value = _mock_knowledge_check_result("legacy")
        cases = self._get_cases(3)
        results = _run_async(run_evaluation(cases, "legacy"))

        report = generate_json_report(
            results=results,
            gold_cases=cases,
            mode="legacy",
            dataset_path=str(DEFAULT_GOLD_CASES_PATH),
            split="test",
        )

        self.assertIn("timestamp", report)
        self.assertIn("metrics", report)
        self.assertIn("thresholds", report)
        self.assertEqual(report["mode"], "legacy")
        self.assertGreater(report["metrics"]["total_samples"], 0)


# ===========================================================================
# 3. Tool Use mode evaluation pipeline (mocked LLM)
# ===========================================================================

class TestToolUseModePipeline(unittest.TestCase):
    """Tool Use 模式评估流程回归测试（mock LLM 调用）。"""

    def _get_cases(self, limit: int = 3):
        from evaluation.datasets import load_gold_cases

        cases = load_gold_cases(DEFAULT_GOLD_CASES_PATH)
        return cases[:limit]

    @patch("evaluation.runners.run_knowledge_check_with_tools", new_callable=AsyncMock)
    def test_tooluse_evaluation_runs(self, mock_kcwt):
        """Tool Use 模式应能正常执行评估并返回结果。"""
        from evaluation.runners import run_evaluation

        mock_kcwt.return_value = _mock_knowledge_check_result("tooluse")
        cases = self._get_cases(3)

        results = _run_async(run_evaluation(cases, "tooluse"))

        self.assertEqual(len(results), len(cases))
        for r in results:
            self.assertEqual(r.mode, "tooluse")
            self.assertIsNotNone(r.knowledge_score)
            self.assertEqual(r.evaluation_status, "completed")
            self.assertIsNone(r.error)

    @patch("evaluation.runners.run_knowledge_check_with_tools", new_callable=AsyncMock)
    def test_tooluse_report_generation(self, mock_kcwt):
        """Tool Use 模式应能生成有效的 JSON 报告。"""
        from evaluation.runners import run_evaluation
        from evaluation.report import generate_json_report

        mock_kcwt.return_value = _mock_knowledge_check_result("tooluse")
        cases = self._get_cases(3)
        results = _run_async(run_evaluation(cases, "tooluse"))

        report = generate_json_report(
            results=results,
            gold_cases=cases,
            mode="tooluse",
            dataset_path=str(DEFAULT_GOLD_CASES_PATH),
            split="test",
        )

        self.assertIn("timestamp", report)
        self.assertIn("metrics", report)
        self.assertIn("thresholds", report)
        self.assertEqual(report["mode"], "tooluse")

    @patch("evaluation.runners.run_knowledge_check_with_tools", new_callable=AsyncMock)
    def test_tooluse_has_tool_trace(self, mock_kcwt):
        """Tool Use 模式的结果应包含 tool_trace 数据。"""
        from evaluation.runners import run_evaluation

        mock_kcwt.return_value = _mock_knowledge_check_result("tooluse")
        cases = self._get_cases(2)
        results = _run_async(run_evaluation(cases, "tooluse"))

        for r in results:
            self.assertIsInstance(r.tool_trace, list)
            self.assertGreater(len(r.tool_trace), 0)
            self.assertIsInstance(r.actual_tool_calls, list)


# ===========================================================================
# 4. Both modes & high-level runners
# ===========================================================================

class TestBothModesAndHighLevelRunners(unittest.TestCase):
    """测试 both 模式及高层评估接口。"""

    def _get_cases(self, limit: int = 2):
        from evaluation.datasets import load_gold_cases

        cases = load_gold_cases(DEFAULT_GOLD_CASES_PATH)
        return cases[:limit]

    @patch("evaluation.runners.run_knowledge_check", new_callable=AsyncMock)
    @patch("evaluation.runners.run_knowledge_check_with_tools", new_callable=AsyncMock)
    def test_both_mode_produces_double_results(self, mock_kcwt, mock_kc):
        """both 模式应为每个案例生成 legacy + tooluse 两条结果。"""
        from evaluation.runners import run_evaluation

        mock_kc.return_value = _mock_knowledge_check_result("legacy")
        mock_kcwt.return_value = _mock_knowledge_check_result("tooluse")
        cases = self._get_cases(2)

        results = _run_async(run_evaluation(cases, "both"))

        self.assertEqual(len(results), len(cases) * 2)
        modes = [r.mode for r in results]
        self.assertEqual(modes[0], "legacy")
        self.assertEqual(modes[1], "tooluse")

    @patch("evaluation.runners.run_knowledge_check", new_callable=AsyncMock)
    def test_run_legacy_rag_evaluation(self, mock_kc):
        """高层 run_legacy_rag_evaluation 应能正常工作。"""
        from evaluation.runners import run_legacy_rag_evaluation

        mock_kc.return_value = _mock_knowledge_check_result("legacy")

        results = _run_async(
            run_legacy_rag_evaluation(
                cases_path=DEFAULT_GOLD_CASES_PATH,
                split="test",
                limit=2,
            )
        )

        self.assertGreater(len(results), 0)
        for r in results:
            self.assertEqual(r.mode, "legacy")

    @patch("evaluation.runners.run_knowledge_check_with_tools", new_callable=AsyncMock)
    def test_run_tool_use_evaluation(self, mock_kcwt):
        """高层 run_tool_use_evaluation 应能正常工作。"""
        from evaluation.runners import run_tool_use_evaluation

        mock_kcwt.return_value = _mock_knowledge_check_result("tooluse")

        results = _run_async(
            run_tool_use_evaluation(
                cases_path=DEFAULT_GOLD_CASES_PATH,
                split="test",
                limit=2,
            )
        )

        self.assertGreater(len(results), 0)
        for r in results:
            self.assertEqual(r.mode, "tooluse")


# ===========================================================================
# 5. Error handling & edge cases
# ===========================================================================

class TestErrorHandling(unittest.TestCase):
    """评估流程中的异常处理。"""

    @patch("evaluation.runners.run_knowledge_check", new_callable=AsyncMock)
    def test_legacy_error_case_produces_error_result(self, mock_kc):
        """Legacy 模式中 LLM 异常应产生 error 结果而非崩溃。"""
        from evaluation.runners import run_case_legacy
        from evaluation.datasets import load_gold_cases, RagGoldCase

        mock_kc.side_effect = RuntimeError("LLM service unavailable")
        cases = load_gold_cases(DEFAULT_GOLD_CASES_PATH)
        case = cases[0]

        result = _run_async(run_case_legacy(case))

        self.assertEqual(result.mode, "legacy")
        self.assertEqual(result.evaluation_status, "error")
        self.assertEqual(result.retrieval_status, "error")
        self.assertIsNotNone(result.error)
        self.assertIn("LLM service unavailable", result.error)

    @patch("evaluation.runners.run_knowledge_check_with_tools", new_callable=AsyncMock)
    def test_tooluse_error_case_produces_error_result(self, mock_kcwt):
        """Tool Use 模式中 LLM 异常应产生 error 结果而非崩溃。"""
        from evaluation.runners import run_case_tooluse
        from evaluation.datasets import load_gold_cases

        mock_kcwt.side_effect = RuntimeError("Tool service unavailable")
        cases = load_gold_cases(DEFAULT_GOLD_CASES_PATH)
        case = cases[0]

        result = _run_async(run_case_tooluse(case))

        self.assertEqual(result.mode, "tooluse")
        self.assertEqual(result.evaluation_status, "error")
        self.assertIsNotNone(result.error)


# ===========================================================================
# 6. CLI invocation (subprocess, mocked via env var)
# ===========================================================================

class TestCLIInvocation(unittest.TestCase):
    """测试 CLI 命令行入口。"""

    def _run_cli(self, mode: str, limit: int = 3) -> subprocess.CompletedProcess:
        """Run the rag_eval CLI in a subprocess with mocked knowledge agent."""
        env = os.environ.copy()
        env["RAG_EVAL_MOCK_LLM"] = "1"  # Signal to use mock if supported

        cmd = [
            sys.executable, "-m", "evaluation.rag_eval",
            "--mode", mode,
            "--limit", str(limit),
            "--output-dir", tempfile.mkdtemp(),
        ]
        return subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            cwd=str(PROJECT_ROOT),
            env=env,
            timeout=120,
        )

    @unittest.skipUnless(
        os.environ.get("RAG_EVAL_RUN_INTEGRATION"),
        "Set RAG_EVAL_RUN_INTEGRATION=1 to run CLI integration tests (requires LLM API)",
    )
    def test_cli_legacy_mode(self):
        """CLI: python -m evaluation.rag_eval --mode legacy --limit 3"""
        result = self._run_cli("legacy", 3)
        self.assertEqual(result.returncode, 0, f"CLI failed:\nstdout: {result.stdout}\nstderr: {result.stderr}")
        self.assertIn("Evaluation completed", result.stdout)

    @unittest.skipUnless(
        os.environ.get("RAG_EVAL_RUN_INTEGRATION"),
        "Set RAG_EVAL_RUN_INTEGRATION=1 to run CLI integration tests (requires LLM API)",
    )
    def test_cli_tooluse_mode(self):
        """CLI: python -m evaluation.rag_eval --mode tooluse --limit 3"""
        result = self._run_cli("tooluse", 3)
        self.assertEqual(result.returncode, 0, f"CLI failed:\nstdout: {result.stdout}\nstderr: {result.stderr}")
        self.assertIn("Evaluation completed", result.stdout)

    def test_cli_mock_mode(self):
        """CLI mock 模式应始终成功（不依赖 LLM API）。"""
        result = self._run_cli("mock", 3)
        self.assertEqual(result.returncode, 0, f"CLI mock mode failed:\nstdout: {result.stdout}\nstderr: {result.stderr}")
        self.assertIn("Evaluation completed", result.stdout)

    def test_cli_mock_mode_generates_reports(self):
        """CLI mock 模式应生成 JSON 和 Markdown 报告文件。"""
        with tempfile.TemporaryDirectory() as tmpdir:
            cmd = [
                sys.executable, "-m", "evaluation.rag_eval",
                "--mode", "mock",
                "--limit", "2",
                "--output-dir", tmpdir,
            ]
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                cwd=str(PROJECT_ROOT),
                timeout=60,
            )
            self.assertEqual(result.returncode, 0, f"stderr: {result.stderr}")

            json_report = Path(tmpdir) / "rag_eval_report.json"
            md_report = Path(tmpdir) / "rag_eval_report.md"
            self.assertTrue(json_report.exists(), "JSON report not generated")
            self.assertTrue(md_report.exists(), "Markdown report not generated")

            # Validate JSON report structure
            with open(json_report, "r", encoding="utf-8") as f:
                report = json.load(f)
            self.assertIn("timestamp", report)
            self.assertIn("metrics", report)
            self.assertIn("thresholds", report)
            self.assertEqual(report["mode"], "mock")


# ===========================================================================
# 7. Report structure validation
# ===========================================================================

class TestReportStructure(unittest.TestCase):
    """验证评估报告的结构完整性。"""

    @patch("evaluation.runners.run_knowledge_check", new_callable=AsyncMock)
    def test_report_contains_all_metric_sections(self, mock_kc):
        """报告应包含所有关键指标部分。"""
        from evaluation.datasets import load_gold_cases
        from evaluation.runners import run_evaluation, filter_cases_by_split
        from evaluation.report import generate_json_report

        mock_kc.return_value = _mock_knowledge_check_result("legacy")

        cases = load_gold_cases(DEFAULT_GOLD_CASES_PATH)
        cases = cases[:3]
        results = _run_async(run_evaluation(cases, "legacy"))

        report = generate_json_report(
            results=results,
            gold_cases=cases,
            mode="legacy",
            dataset_path=str(DEFAULT_GOLD_CASES_PATH),
            split="test",
        )

        # Top-level keys
        for key in ("timestamp", "mode", "dataset", "metrics", "thresholds"):
            self.assertIn(key, report, f"Missing top-level key: {key}")

        # Dataset info
        self.assertIn("path", report["dataset"])
        self.assertIn("split", report["dataset"])
        self.assertIn("total_samples", report["dataset"])

        # Thresholds
        self.assertIn("passed", report["thresholds"])
        self.assertIn("violations", report["thresholds"])

    @patch("evaluation.runners.run_knowledge_check", new_callable=AsyncMock)
    def test_markdown_report_contains_expected_sections(self, mock_kc):
        """Markdown 报告应包含关键章节。"""
        from evaluation.datasets import load_gold_cases
        from evaluation.runners import run_evaluation, filter_cases_by_split
        from evaluation.report import generate_json_report, generate_markdown_report

        mock_kc.return_value = _mock_knowledge_check_result("legacy")

        cases = load_gold_cases(DEFAULT_GOLD_CASES_PATH)
        cases = cases[:2]
        results = _run_async(run_evaluation(cases, "legacy"))

        json_report = generate_json_report(
            results=results,
            gold_cases=cases,
            mode="legacy",
            dataset_path=str(DEFAULT_GOLD_CASES_PATH),
            split="test",
        )
        md = generate_markdown_report(json_report)

        self.assertIn("# RAG / Tool Use 评估报告", md)
        self.assertIn("## 概述", md)
        self.assertIn("legacy", md.lower())


# ===========================================================================
# 8. Query type classification
# ===========================================================================

class TestQueryTypeClassification(unittest.TestCase):
    """测试查询类型分类逻辑。"""

    def test_classify_query_type(self):
        """查询类型分类应正确反映 gold case 属性。"""
        from evaluation.runners import classify_query_type
        from evaluation.datasets import load_gold_cases

        cases = load_gold_cases(DEFAULT_GOLD_CASES_PATH)

        # case_005 (拒绝回答) → referral
        case_005 = next(c for c in cases if c.case_id == "case_005")
        self.assertEqual(classify_query_type(case_005), "referral")

        # case_001 (普通信息) → information (no gold_citation_ids)
        case_001 = cases[0]
        qt = classify_query_type(case_001)
        self.assertIn(qt, ("information", "citation"))

    def test_group_cases_by_query_type(self):
        """分组函数应返回三个分组。"""
        from evaluation.runners import group_cases_by_query_type
        from evaluation.datasets import load_gold_cases

        cases = load_gold_cases(DEFAULT_GOLD_CASES_PATH)
        groups = group_cases_by_query_type(cases)

        self.assertIn("information", groups)
        self.assertIn("citation", groups)
        self.assertIn("referral", groups)
        total = sum(len(v) for v in groups.values())
        self.assertEqual(total, len(cases))


if __name__ == "__main__":
    unittest.main()
