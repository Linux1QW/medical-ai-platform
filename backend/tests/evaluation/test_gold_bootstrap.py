# -*- coding: utf-8 -*-
"""gold_bootstrap 与 source_hit_rate 指标测试

覆盖：规则命中/未命中、notes 标记、幂等不覆盖、复合诊断并集、
source_hit_rate 指标计算口径，以及 mock 绿路径能通过阈值门
（CI 阻断门的本地等价验证）。
"""
import asyncio
import unittest

from evaluation.datasets import RagEvalResult, RagGoldCase
from evaluation.gold_bootstrap import (
    AUTO_SUGGESTED_MARK,
    bootstrap_gold_case,
    bootstrap_gold_cases,
    suggest_gold_sources,
)
from evaluation.report import calculate_source_metrics, generate_json_report
from evaluation.runners import create_mock_cases, run_evaluation


def _make_case(**overrides) -> RagGoldCase:
    """构造最小合法 RagGoldCase"""
    data = {
        "case_id": "case-1",
        "split": "regression",
        "department": "未知",
        "difficulty": "medium",
        "chief_complaint": "胃部不适",
        "patient_info": "患者女，30岁",
        "conversation_text": "医生: 哪里不舒服？\n患者: 胃疼。",
        "doctor_diagnosis": "慢性胃炎",
        "expected_stance": "supports",
        "should_refuse": False,
        "notes": "converted from dataset/case-1",
    }
    data.update(overrides)
    return RagGoldCase(**data)


def _make_result(sources, case_id="case-1") -> RagEvalResult:
    return RagEvalResult(
        case_id=case_id,
        mode="legacy",
        evaluation_status="completed",
        human_review_needed=False,
        retrieval_status="sufficient",
        citation_data=[{"id": f"c-{i}", "source": src} for i, src in enumerate(sources)],
    )


class TestSuggestGoldSources(unittest.TestCase):
    def test_digestive_keywords_hit(self):
        self.assertEqual(suggest_gold_sources("慢性胃炎"), ["消化内科", "内科学"])
        self.assertEqual(suggest_gold_sources("腹痛"), ["消化内科", "内科学"])
        self.assertEqual(suggest_gold_sources("幽门螺杆菌感染"), ["消化内科", "内科学"])

    def test_general_exam_hit(self):
        self.assertEqual(suggest_gold_sources("一般性医学检查"), ["诊断学", "全科医学"])

    def test_combined_rules_union_ordered(self):
        # 复合文本命中多条规则时取并集且保持规则顺序
        result = suggest_gold_sources("一般性医学检查 主诉胃部不适")
        self.assertEqual(result, ["消化内科", "内科学", "诊断学", "全科医学"])

    def test_no_match_and_empty(self):
        self.assertEqual(suggest_gold_sources("骨折"), [])
        self.assertEqual(suggest_gold_sources(""), [])


class TestBootstrapGoldCase(unittest.TestCase):
    def test_annotates_and_marks_notes(self):
        case = _make_case()
        updated = bootstrap_gold_case(case)
        self.assertIsNot(updated, case)
        self.assertEqual(updated.gold_relevant_sources, ["消化内科", "内科学"])
        self.assertIn(AUTO_SUGGESTED_MARK, updated.notes)
        # 原有 notes 内容保留
        self.assertIn("converted from dataset/case-1", updated.notes)
        # 原对象不被修改
        self.assertEqual(case.gold_relevant_sources, [])

    def test_existing_gold_not_overwritten(self):
        case = _make_case(gold_relevant_sources=["人工标注来源"])
        updated = bootstrap_gold_case(case)
        self.assertIs(updated, case)
        self.assertEqual(updated.gold_relevant_sources, ["人工标注来源"])

    def test_idempotent_by_notes_mark(self):
        once = bootstrap_gold_case(_make_case())
        twice = bootstrap_gold_case(once)
        self.assertIs(twice, once)

    def test_no_rule_match_returns_original(self):
        case = _make_case(doctor_diagnosis="骨折", chief_complaint="摔伤")
        updated = bootstrap_gold_case(case)
        self.assertIs(updated, case)
        self.assertEqual(updated.gold_relevant_sources, [])

    def test_batch_counts_annotated(self):
        cases = [
            _make_case(case_id="a"),
            _make_case(case_id="b", doctor_diagnosis="骨折", chief_complaint="摔伤"),
        ]
        bootstrapped, annotated = bootstrap_gold_cases(cases)
        self.assertEqual(len(bootstrapped), 2)
        self.assertEqual(annotated, 1)


class TestSourceHitRate(unittest.TestCase):
    def test_hit_by_substring(self):
        case = _make_case(gold_relevant_sources=["消化内科"])
        result = _make_result(["3.内科学 消化内科分册.pdf"])
        metrics = calculate_source_metrics([result], [case])
        self.assertEqual(metrics, {"source_hit_rate": 1.0})

    def test_miss(self):
        case = _make_case(gold_relevant_sources=["消化内科"])
        result = _make_result(["24.眼科学.pdf"])
        metrics = calculate_source_metrics([result], [case])
        self.assertEqual(metrics, {"source_hit_rate": 0.0})

    def test_no_gold_annotation_returns_empty(self):
        # 无标注数据集不产出指标，避免阈值误报
        case = _make_case(gold_relevant_sources=[])
        result = _make_result(["3.内科学 消化内科分册.pdf"])
        self.assertEqual(calculate_source_metrics([result], [case]), {})

    def test_refusal_cases_excluded(self):
        case = _make_case(gold_relevant_sources=["消化内科"], should_refuse=True)
        result = _make_result([])
        self.assertEqual(calculate_source_metrics([result], [case]), {})

    def test_mixed_hit_rate(self):
        cases = [
            _make_case(case_id="a", gold_relevant_sources=["消化内科"]),
            _make_case(case_id="b", gold_relevant_sources=["眼科"]),
        ]
        results = [
            _make_result(["内科学 消化内科分册.pdf"], case_id="a"),
            _make_result(["儿科学.pdf"], case_id="b"),
        ]
        metrics = calculate_source_metrics(results, cases)
        self.assertAlmostEqual(metrics["source_hit_rate"], 0.5)


class TestMockThresholdGate(unittest.TestCase):
    """CI 阻断门等价验证：mock 绿路径必须通过全部阈值"""

    def test_mock_run_passes_thresholds(self):
        cases = create_mock_cases(5)
        results = asyncio.run(run_evaluation(cases, "mock"))
        report = generate_json_report(
            results=results, gold_cases=cases, mode="mock",
            dataset_path="mock", split="dev",
        )
        self.assertTrue(
            report["thresholds"]["passed"],
            f"violations: {report['thresholds'].get('violations')}",
        )
        # mock 自洽性：citation 有效、检索命中 gold、来源命中
        metrics = report["metrics"]
        self.assertEqual(metrics["citation_validity"], 1.0)
        self.assertEqual(metrics["citation_hallucination_rate"], 0.0)
        self.assertEqual(metrics["recall_at_1"], 1.0)
        self.assertEqual(metrics["source_hit_rate"], 1.0)


if __name__ == "__main__":
    unittest.main()
