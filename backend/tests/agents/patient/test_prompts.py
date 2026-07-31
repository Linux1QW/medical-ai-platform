# -*- coding: utf-8 -*-
"""prompts.py 单元测试：代问身份判定与角色提示构造（纯函数，零 LLM 成本）"""
from app.services.agents.patient.prompts import (
    _IDENTITY_PROXY,
    _IDENTITY_SELF,
    build_role_prompt,
    is_proxy_consult,
)


class TestIsProxyConsult:
    def test_detects_family_proxy(self):
        assert is_proxy_consult("主诉：代替家人咨询病情") is True

    def test_detects_various_phrasings(self):
        for text in ["替家人问问", "帮家人咨询", "陪同就诊代为咨询", "代其咨询"]:
            assert is_proxy_consult(text) is True

    def test_self_consult_is_false(self):
        assert is_proxy_consult("主诉：上腹隐痛两周") is False

    def test_medical_terms_no_false_positive(self):
        # "代谢""替代治疗"等含"代/替"的医学词不应误判为代问
        assert is_proxy_consult("代谢综合征，既往用替代治疗") is False

    def test_empty_is_false(self):
        assert is_proxy_consult("") is False
        assert is_proxy_consult(None) is False


class TestBuildRolePrompt:
    def test_self_profile_uses_first_person_identity(self):
        prompt = build_role_prompt("45岁男性，上腹痛两周")
        assert _IDENTITY_SELF in prompt
        assert _IDENTITY_PROXY not in prompt
        assert "45岁男性，上腹痛两周" in prompt

    def test_proxy_profile_switches_identity(self):
        prompt = build_role_prompt("54岁男性，代替家人咨询病情")
        assert _IDENTITY_PROXY in prompt
        assert _IDENTITY_SELF not in prompt

    def test_no_unresolved_placeholder(self):
        # 两个占位符都应被替换，不残留 {identity_line} / {system_prompt}
        prompt = build_role_prompt("")
        assert "{identity_line}" not in prompt
        assert "{system_prompt}" not in prompt
