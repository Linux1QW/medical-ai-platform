# -*- coding: utf-8 -*-
"""planner.py 单元测试：问诊阶段关键词分类"""
from app.services.agents.patient.planner import STAGES, classify_stage


class TestClassifyStage:
    def test_stage_list_complete(self):
        assert STAGES == [
            "greeting", "chief_complaint", "hpi", "past_history",
            "personal_family_history", "physical_exam",
            "assessment_communication", "closing",
        ]

    def test_chief_complaint(self):
        assert classify_stage("您好，哪里不舒服？", "greeting") == "chief_complaint"

    def test_hpi(self):
        assert classify_stage("疼了多久了？什么时候开始的？", "chief_complaint") == "hpi"

    def test_past_history(self):
        assert classify_stage("以前得过什么病吗？有没有药物过敏？", "hpi") == "past_history"

    def test_physical_exam(self):
        assert classify_stage("来，量一下体温和血压", "hpi") == "physical_exam"

    def test_no_hit_keeps_current(self):
        assert classify_stage("嗯。", "hpi") == "hpi"

    def test_empty_message_keeps_current(self):
        assert classify_stage("", "greeting") == "greeting"
