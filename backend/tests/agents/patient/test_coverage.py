# -*- coding: utf-8 -*-
"""coverage.py 单元测试：披露账本 -> 问诊覆盖报告"""
from app.services.agents.patient.coverage import build_coverage_report, format_coverage_text
from app.services.agents.patient.memory import Fact, MemoryState


def _memory():
    return MemoryState(
        trust=0.65, emotion="缓和",
        stage_history=["greeting", "chief_complaint", "chief_complaint", "hpi"],
        facts=[
            Fact(fact_id="sym_001", content="上腹隐痛", status="disclosed", disclosed_at_turn=2),
            Fact(fact_id="sym_002", content="反酸烧心", status="disclosed", disclosed_at_turn=3),
            Fact(fact_id="his_001", category="history", content="胃溃疡史"),
            Fact(fact_id="his_002", category="history", content="青霉素过敏", status="denied"),
        ],
    )


class TestBuildCoverageReport:
    def test_report_fields(self):
        r = build_coverage_report(_memory())
        assert r["total_facts"] == 4 and r["disclosed_count"] == 2
        assert r["disclosure_rate"] == 0.5
        assert r["undisclosed_facts"] == ["胃溃疡史"]
        assert r["stage_path"] == ["greeting", "chief_complaint", "hpi"]  # 去重保序
        assert r["final_trust"] == 0.65 and r["final_emotion"] == "缓和"

    def test_empty_memory_no_division_error(self):
        r = build_coverage_report(MemoryState())
        assert r["disclosure_rate"] == 0.0 and r["total_facts"] == 0


class TestFormatCoverageText:
    def test_contains_key_numbers(self):
        text = format_coverage_text(build_coverage_report(_memory()))
        assert "50.0%" in text and "胃溃疡史" in text and "0.65" in text
