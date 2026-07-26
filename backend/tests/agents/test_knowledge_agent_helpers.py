# -*- coding: utf-8 -*-
"""knowledge_agent 纯函数辅助单测（拆分前锁定行为）

覆盖此前无测试的三个纯函数：
- _map_consistency_to_score（v1 评分映射）
- _generate_analysis（分析文本生成）
- _extract_consultation_data（consultation 字段提取，dict / 对象两种形态）
"""

from app.services.agents.knowledge_agent import (
    _extract_consultation_data,
    _generate_analysis,
    _map_consistency_to_score,
)
from app.services.rag.types import ClinicalFacts

# ── _map_consistency_to_score（v1）─────────────────────────────────────────────

class TestMapConsistencyToScoreV1:
    def test_supports_full_confidence(self):
        # base=90, conf=1.0 → 90
        assert _map_consistency_to_score("supports", 1.0) == 90

    def test_supports_zero_confidence(self):
        # base=90, conf=0 → 90*0.5 = 45
        assert _map_consistency_to_score("supports", 0.0) == 45

    def test_mixed_mid_confidence(self):
        # base=65, conf=0.5 → 65*0.5 + 65*0.5*0.5 = 48.75 → int 48
        assert _map_consistency_to_score("mixed", 0.5) == 48

    def test_contradicts(self):
        # base=40, conf=1.0 → 40
        assert _map_consistency_to_score("contradicts", 1.0) == 40

    def test_undetermined_uses_base_50(self):
        assert _map_consistency_to_score("undetermined", 1.0) == 50

    def test_unknown_stance_falls_back_to_50(self):
        assert _map_consistency_to_score("nonsense", 1.0) == 50

    def test_returns_int(self):
        assert isinstance(_map_consistency_to_score("supports", 0.7), int)


# ── _generate_analysis ────────────────────────────────────────────────────────

def _make_facts(**overrides) -> ClinicalFacts:
    defaults = dict(
        age=45,
        gender="男性",
        chief_complaint="咳嗽三天",
        symptoms=["咳嗽"],
        timeline=[],
        red_flags=[],
        comorbidities=[],
        medications=[],
        allergies=[],
        doctor_diagnoses=["急性支气管炎"],
        treatment_items=["阿莫西林"],
    )
    defaults.update(overrides)
    return ClinicalFacts(**defaults)


class TestGenerateAnalysis:
    def _call(self, **overrides) -> str:
        kwargs = dict(
            consistency_result={
                "confidence": 0.8,
                "analysis": "证据支持该诊断。",
                "key_findings": ["指南推荐一线用药"],
            },
            facts=_make_facts(),
            doctor_diagnosis="急性支气管炎",
            treatment_plan="阿莫西林口服",
            retrieval_status="sufficient",
            evidence_stance="supports",
            citations=[{"citation_id": "c1"}],
            needs_review=False,
            review_reason=None,
        )
        kwargs.update(overrides)
        return _generate_analysis(**kwargs)

    def test_normal_path_mentions_stance_and_confidence(self):
        text = self._call()
        assert "基本一致" in text
        assert "80%" in text
        assert "引用1条医学证据" in text

    def test_needs_review_path(self):
        text = self._call(needs_review=True, review_reason="insufficient_evidence")
        assert "无法完成自动评估" in text
        assert "insufficient_evidence" in text
        assert "人工复核" in text
        # 附带医生诊断
        assert "急性支气管炎" in text

    def test_max_length_300(self):
        long_analysis = "很长的分析。" * 100
        text = self._call(consistency_result={
            "confidence": 0.9, "analysis": long_analysis, "key_findings": [],
        })
        assert len(text) <= 300

    def test_short_text_padded(self):
        # 极简输入下补充建议语，长度不至于过短
        text = self._call(
            consistency_result={"confidence": 0.5, "analysis": "", "key_findings": []},
            citations=[],
        )
        assert "循证医学证据" in text

    def test_unknown_stance_desc(self):
        text = self._call(evidence_stance="weird")
        assert "一致性未确定" in text


# ── _extract_consultation_data ────────────────────────────────────────────────

class _FakeConsultation:
    """对象形态的 consultation（仅携带被访问的属性）"""

    def __init__(self, **kwargs):
        for k, v in kwargs.items():
            setattr(self, k, v)


class TestExtractConsultationData:
    def test_dict_with_patient_info(self):
        data = {
            "conversation_text": "医生：您好",
            "patient_info": "年龄：50岁",
            "doctor_diagnosis": "高血压",
            "treatment_plan": "氨氯地平",
        }
        conv, info, dx, tx = _extract_consultation_data(data)
        assert conv == "医生：您好"
        assert info == "年龄：50岁"
        assert dx == "高血压"
        assert tx == "氨氯地平"

    def test_dict_builds_patient_info_from_fields(self):
        data = {
            "conversation_text": "",
            "patient_age": 62,
            "patient_gender": "女",
            "chief_complaint": "胸闷一周",
            "symptoms": ["胸闷", "气促"],
            "medical_history": "高血压10年",
        }
        _, info, _, _ = _extract_consultation_data(data)
        assert "年龄：62岁" in info
        assert "性别：女" in info
        assert "主诉：胸闷一周" in info
        assert "症状：胸闷、气促" in info
        assert "病史：高血压10年" in info

    def test_object_with_patient_info(self):
        obj = _FakeConsultation(
            conversation_text="患者：头疼",
            patient_info="年龄：30岁",
            doctor_diagnosis="偏头痛",
            treatment_plan="布洛芬",
        )
        conv, info, dx, tx = _extract_consultation_data(obj)
        assert conv == "患者：头疼"
        assert info == "年龄：30岁"
        assert dx == "偏头痛"
        assert tx == "布洛芬"

    def test_object_builds_patient_info_from_fields(self):
        obj = _FakeConsultation(
            conversation_text="",
            patient_info="",
            patient_age=8,
            patient_gender="男",
            chief_complaint="发热",
            symptoms=["发热", "咳嗽"],
            medical_history="",
            doctor_diagnosis="",
            treatment_plan="",
        )
        _, info, _, _ = _extract_consultation_data(obj)
        assert "年龄：8岁" in info
        assert "性别：男" in info
        assert "症状：发热、咳嗽" in info

    def test_missing_fields_default_empty(self):
        conv, info, dx, tx = _extract_consultation_data({})
        assert conv == ""
        assert info == ""
        assert dx == ""
        assert tx == ""
