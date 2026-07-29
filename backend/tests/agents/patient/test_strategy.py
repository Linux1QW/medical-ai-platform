# -*- coding: utf-8 -*-
"""strategy.py 单元测试：人格×阶段策略查询"""
from app.services.agents.patient.planner import STAGES
from app.services.agents.patient.strategy import DisclosureStrategy, get_strategy


class TestGetStrategy:
    def test_anxious_hpi_asks_back(self):
        s = get_strategy("焦虑型", "hpi")
        assert s.ask_back is True and s.tone_hint

    def test_reticent_always_short(self):
        for stage in STAGES:
            assert get_strategy("沉默型", stage).reply_length == "极短"

    def test_unknown_combo_falls_back(self):
        s = get_strategy("未知人格", "未知阶段")
        assert isinstance(s, DisclosureStrategy)

    def test_cooperative_assessment_may_volunteer(self):
        assert get_strategy("配合型", "assessment_communication").volunteer_info is True
