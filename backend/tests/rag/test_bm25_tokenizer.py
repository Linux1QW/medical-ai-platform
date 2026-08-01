"""Contract tests for the medical lexical tokenizer V2."""

import pytest

from app.services.rag.lexical.tokenizer import tokenize_medical_text
from app.services.rag.lexical.query_expansion import expand_lexical_query


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("EGFR 19外显子缺失", {"gene:EGFR", "19外显子缺失"}),
        ("eGFR 35 ml/min/1.73m2", {"renal:eGFR", "35", "ml/min/1.73m2"}),
        ("PDL1 TPS≥50%", {"PD-L1", "TPS", ">=50%"}),
        ("阿司匹林100mg qd", {"阿司匹林", "100mg", "qd"}),
        ("2型糖尿病", {"2型糖尿病"}),
        ("EGFR L858R", {"gene:EGFR", "variant:L858R"}),
        ("诊断 I10，合并 E11.9", {"I10", "E11.9"}),
    ],
)
def test_medical_tokens(text, expected):
    assert expected <= set(tokenize_medical_text(text, mode="query"))


def test_bigrams_do_not_cross_punctuation_boundaries():
    tokens = tokenize_medical_text("阿司匹林，氯吡格雷", mode="document")

    assert "林氯" not in tokens
    assert "bg:林氯" not in tokens


def test_medical_negations_are_preserved():
    tokens = tokenize_medical_text("无发热，不咳嗽，未见胸痛，否认腹泻", mode="query")

    assert {"无", "不", "未", "否认"} <= set(tokens)


def test_disease_type_does_not_consume_trailing_cjk_context():
    sentence = "2型糖尿病患者无发热"
    tokens = tokenize_medical_text(sentence, mode="query")

    assert "2型糖尿病" in tokens
    assert "无" in tokens
    assert sentence not in tokens


def test_icd_u_code_is_preserved_as_one_token():
    tokens = tokenize_medical_text("确诊 U07.1", mode="query")

    assert "U07.1" in tokens


def test_renal_unit_without_whitespace_is_preserved():
    tokens = tokenize_medical_text("eGFR 35mL/min/1.73m2", mode="query")

    assert {"renal:eGFR", "35", "mL/min/1.73m2"} <= set(tokens)


def test_protected_tokens_are_stably_deduplicated():
    tokens = tokenize_medical_text("EGFR EGFR 100mg 100mg", mode="query")

    assert tokens.count("gene:EGFR") == 1
    assert tokens.count("100mg") == 1


def test_entity_expansion_preserves_tokens_and_adds_normalized_name_and_codes(monkeypatch):
    monkeypatch.setattr(
        "app.services.rag.lexical.query_expansion.extract_entities",
        lambda text: [
            {
                "normalized": "急性心肌梗死",
                "icd10": "I21.9",
                "icd9cm3": "36.06",
            }
        ],
    )

    tokens = expand_lexical_query("心梗治疗", ["心梗", "治疗", "心梗"])

    assert tokens == ["心梗", "治疗", "急性心肌梗死", "icd10:I21.9", "icd9cm3:36.06"]
