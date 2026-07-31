# -*- coding: utf-8 -*-
"""badcase_attribution 归因规则单元测试（纯逻辑、无网络）。"""
from evaluation.badcase_attribution import (
    classify_badcase,
    label_badcases,
    summarize_badcases,
    LOW_OVERALL,
)


def _rec(rc, med, nat, disc, **kw):
    r = {"scores": {"role_consistency": rc, "medical_plausibility": med,
                    "naturalness": nat, "disclosure_timing": disc}}
    r.update(kw)
    return r


def test_single_dim_breach_maps_to_its_mode():
    # 仅角色维破线 -> 主因 role_consistency
    c = classify_badcase(_rec(2, 5, 5, 4))
    assert c["attribution"] == "role_consistency"
    assert c["modes"] == ["role_consistency"]
    assert c["attribution_label"] == "角色一致性破坏"


def test_lowest_score_wins_as_primary():
    # 多维破线：disc=1 最低 -> 主因 disclosure_timing，modes 含全部破线维
    c = classify_badcase(_rec(3, 2, 2, 1))
    assert c["attribution"] == "disclosure_timing"
    assert set(c["modes"]) == {"medical_plausibility", "naturalness", "disclosure_timing"}


def test_tie_broken_by_priority_order():
    # 分数并列（都=2）：按 MODE_PRIORITY，role_consistency 优先
    c = classify_badcase(_rec(2, 2, 2, 2))
    assert c["attribution"] == "role_consistency"
    # modes 按优先级顺序返回
    assert c["modes"] == ["role_consistency", "medical_plausibility",
                          "disclosure_timing", "naturalness"]


def test_no_dim_breach_falls_back_to_low_overall():
    # 无单维 ≤2（仅综合分触发的场景）-> low_overall
    c = classify_badcase(_rec(3, 4, 3, 3))
    assert c["attribution"] == LOW_OVERALL
    assert c["modes"] == []


def test_threshold_is_configurable():
    c = classify_badcase(_rec(3, 3, 3, 3), threshold=3)
    assert c["attribution"] == "role_consistency"  # 全=3 破线，优先级取 role


def test_label_badcases_does_not_mutate_input():
    rec = _rec(2, 5, 5, 5, case_id="c1")
    out = label_badcases([rec])
    assert "attribution" not in rec  # 原对象不被污染
    assert out[0]["attribution"] == "role_consistency"
    assert out[0]["attribution_modes"] == ["role_consistency"]


def test_summarize_counts_by_mode_and_arm_without_dialogue():
    records = [
        _rec(2, 5, 5, 5, case_id="c1", arm="legacy", turn=1, personality="配合型",
             overall=4.0, doctor="医生问", reply="患者答"),
        _rec(5, 5, 5, 2, case_id="c2", arm="agent_tool", turn=3, personality="对抗型",
             overall=3.75, doctor="d", reply="r"),
        _rec(5, 5, 5, 2, case_id="c3", arm="agent_tool", turn=6, personality="对抗型",
             overall=3.5, doctor="d", reply="r"),
    ]
    s = summarize_badcases(records)
    assert s["total"] == 3
    assert s["by_mode"]["disclosure_timing"] == 2
    assert s["by_mode"]["role_consistency"] == 1
    assert s["by_arm"]["agent_tool"]["disclosure_timing"] == 2
    assert s["by_personality"]["对抗型"] == 2
    # 去标识：清单条目不得泄露对话文本
    for e in s["entries"]:
        assert "doctor" not in e and "reply" not in e
