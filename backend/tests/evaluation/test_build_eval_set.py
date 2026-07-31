# -*- coding: utf-8 -*-
"""分层抽样 scan_dataset / stratified_sample 确定性与覆盖测试"""
import json

from evaluation.patient_eval_set import scan_dataset, stratified_sample


def _make_case(root, name, personality_raw, diagnosis, dialogue_turns):
    d = root / name
    d.mkdir()
    dialogue = [{"医生": f"问题{i}", "患者": f"回答{i}"} for i in range(dialogue_turns)]
    data = {
        "基础信息": {"姓名": name, "性别": "女", "年龄": "30"},
        "人格": {"性格": personality_raw},
        "门诊病历": {"主诉": "腹痛", "现病史": "三天", "既往史": "无"},
        "主诊断": diagnosis,
        "门诊对话": dialogue,
    }
    with open(d / f"{name}.json", "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False)


def _build_fake_dataset(root):
    # 4 personalities x 多诊断，含 1 例空对话应被跳过
    _make_case(root, "patient1_1", "合作", "慢性胃炎", 8)
    _make_case(root, "patient2_1", "合作", "腹痛", 10)
    _make_case(root, "patient3_1", "啰嗦", "腹泻", 6)
    _make_case(root, "patient4_1", "偏执", "便秘", 12)
    _make_case(root, "patient5_1", "怀疑", "反酸", 9)
    _make_case(root, "patient6_1", "合作", "反流性食管炎", 7)
    _make_case(root, "patient7_1", "合作", "慢性胃炎", 0)  # 空对话，跳过


def test_scan_dataset_skips_empty_dialogue(tmp_path):
    _build_fake_dataset(tmp_path)
    records = scan_dataset(tmp_path)
    ids = {r["case_id"] for r in records}
    assert "patient7_1" not in ids  # 空对话被跳过
    assert len(records) == 6
    # 人格归一化：合作 -> 配合型
    rec1 = next(r for r in records if r["case_id"] == "patient1_1")
    assert rec1["personality"] == "配合型"
    assert rec1["diagnosis"] == "慢性胃炎"
    assert rec1["turns_available"] == 8


def test_stratified_sample_deterministic(tmp_path):
    _build_fake_dataset(tmp_path)
    records = scan_dataset(tmp_path)
    a = [r["case_id"] for r in stratified_sample(records, n=4, seed=42)]
    b = [r["case_id"] for r in stratified_sample(records, n=4, seed=42)]
    assert a == b  # 同种子完全一致
    assert len(a) == 4


def test_stratified_sample_covers_multiple_personalities(tmp_path):
    _build_fake_dataset(tmp_path)
    records = scan_dataset(tmp_path)
    selected = stratified_sample(records, n=4, seed=42)
    personalities = {r["personality"] for r in selected}
    # 轮转抽样应覆盖到多种人格，而非全部集中在配合型
    assert len(personalities) >= 3


def test_stratified_sample_respects_n_cap(tmp_path):
    _build_fake_dataset(tmp_path)
    records = scan_dataset(tmp_path)
    # n 超过可用总数时返回全部去重记录
    selected = stratified_sample(records, n=100, seed=42)
    assert len(selected) == len(records)
    assert len({r["case_id"] for r in selected}) == len(records)
