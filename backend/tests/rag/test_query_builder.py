# -*- coding: utf-8 -*-
"""查询构建测试"""

from app.services.agents.knowledge_agent import build_queries, extract_clinical_facts
from app.services.rag.types import ClinicalFacts


class TestExtractClinicalFacts:
    def test_basic_extraction(self):
        facts = extract_clinical_facts(
            conversation_text="患者：我最近咳嗽两周了，还发烧。\n医生：有没有胸闷？\n患者：有的。",
            patient_info="男，45岁",
            doctor_diagnosis="社区获得性肺炎",
            treatment_plan="阿莫西林 0.5g tid 7天；对症治疗",
        )
        assert isinstance(facts, ClinicalFacts)
        assert len(facts.doctor_diagnoses) >= 1
        assert len(facts.treatment_items) >= 1

    def test_empty_inputs(self):
        facts = extract_clinical_facts(
            conversation_text="",
            patient_info="",
            doctor_diagnosis="",
            treatment_plan="",
        )
        assert isinstance(facts, ClinicalFacts)
        assert facts.doctor_diagnoses == [] or facts.doctor_diagnoses is not None


class TestBuildQueries:
    def test_all_three_queries(self):
        facts = ClinicalFacts(
            age=45,
            gender="男",
            chief_complaint="咳嗽两周",
            symptoms=["咳嗽", "发热", "胸闷"],
            timeline=["2周"],
            red_flags=[],
            comorbidities=["高血压"],
            medications=[],
            allergies=[],
            doctor_diagnoses=["社区获得性肺炎"],
            treatment_items=["阿莫西林 0.5g tid"],
        )
        queries = build_queries(facts)
        assert len(queries) == 3
        assert queries[0].query_type == "case"
        assert queries[1].query_type == "diagnosis"
        assert queries[2].query_type == "treatment"

    def test_case_query_no_diagnosis(self):
        """病例查询不应包含医生诊断"""
        facts = ClinicalFacts(
            age=30,
            gender="女",
            chief_complaint="头痛",
            symptoms=["头痛", "恶心"],
            timeline=["3天"],
            red_flags=[],
            comorbidities=[],
            medications=[],
            allergies=[],
            doctor_diagnoses=["偏头痛"],
            treatment_items=["布洛芬"],
        )
        queries = build_queries(facts)
        case_query = queries[0]
        assert case_query.query_type == "case"
        assert "偏头痛" not in case_query.text  # 确认偏误防护

    def test_no_diagnosis_no_treatment(self):
        """未提交诊断和治疗时只生成病例查询"""
        facts = ClinicalFacts(
            age=25,
            gender="男",
            chief_complaint="腹痛",
            symptoms=["腹痛"],
            timeline=["1天"],
            red_flags=[],
            comorbidities=[],
            medications=[],
            allergies=[],
            doctor_diagnoses=[],
            treatment_items=[],
        )
        queries = build_queries(facts)
        assert len(queries) == 1
        assert queries[0].query_type == "case"
