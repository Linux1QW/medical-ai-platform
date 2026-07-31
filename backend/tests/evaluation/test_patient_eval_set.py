# -*- coding: utf-8 -*-
"""患者评测集 load_eval_set schema/去重/跳过 _meta 行测试"""
import json

import pytest
from pydantic import ValidationError

from evaluation.patient_eval_set import EvalCase, load_eval_set


def _write_jsonl(path, rows):
    with open(path, "w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def test_load_eval_set_parses_cases(tmp_path):
    p = tmp_path / "set.jsonl"
    _write_jsonl(p, [
        {"_meta": {"version": "v1", "seed": 42, "n": 2}},
        {"case_id": "patient1_1", "personality": "配合型", "diagnosis": "慢性胃炎", "turns_available": 8},
        {"case_id": "patient2_2", "personality": "焦虑型", "diagnosis": "腹痛", "turns_available": 12},
    ])
    cases = load_eval_set(p)
    assert len(cases) == 2
    assert all(isinstance(c, EvalCase) for c in cases)
    assert cases[0].case_id == "patient1_1"
    assert cases[1].personality == "焦虑型"


def test_load_eval_set_skips_meta_line(tmp_path):
    p = tmp_path / "set.jsonl"
    _write_jsonl(p, [
        {"_meta": {"version": "v1"}},
        {"case_id": "patient1_1", "personality": "配合型", "diagnosis": "慢性胃炎", "turns_available": 8},
    ])
    cases = load_eval_set(p)
    assert [c.case_id for c in cases] == ["patient1_1"]


def test_load_eval_set_dedupes_by_case_id(tmp_path):
    p = tmp_path / "set.jsonl"
    _write_jsonl(p, [
        {"case_id": "patient1_1", "personality": "配合型", "diagnosis": "慢性胃炎", "turns_available": 8},
        {"case_id": "patient1_1", "personality": "配合型", "diagnosis": "慢性胃炎", "turns_available": 9},
        {"case_id": "patient2_2", "personality": "焦虑型", "diagnosis": "腹痛", "turns_available": 12},
    ])
    cases = load_eval_set(p)
    assert [c.case_id for c in cases] == ["patient1_1", "patient2_2"]
    # 去重保留首次出现
    assert cases[0].turns_available == 8


def test_load_eval_set_raises_on_missing_field(tmp_path):
    p = tmp_path / "set.jsonl"
    _write_jsonl(p, [
        {"case_id": "patient1_1", "personality": "配合型"},  # 缺 diagnosis / turns_available
    ])
    with pytest.raises(ValidationError):
        load_eval_set(p)
