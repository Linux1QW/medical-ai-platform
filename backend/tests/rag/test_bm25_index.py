"""Regression characterization tests for the V1 BM25 tokenizer.

These tests document defects that Task 2's tokenizer V2 is responsible for
fixing.  They deliberately execute against V1, but remain strict xfails so a
surprise pass cannot silently leave an obsolete expected-failure marker behind.
"""

import pytest

from app.services.rag.bm25_search import tokenize_medical_text


@pytest.mark.xfail(
    strict=True,
    reason="Tokenizer V1 lowercases protected Latin clinical tokens, collapsing EGFR and eGFR.",
)
def test_egfr_and_egfr_renal_are_distinct():
    assert tokenize_medical_text("EGFR") != tokenize_medical_text("eGFR")


@pytest.mark.xfail(
    strict=True,
    reason="Tokenizer V1 forms character bigrams after stripping punctuation, so it crosses Chinese punctuation boundaries.",
)
def test_bigram_does_not_cross_punctuation():
    assert "林氯" not in tokenize_medical_text("阿司匹林，氯吡格雷")


@pytest.mark.xfail(
    strict=True,
    reason="Tokenizer V1 splits disease types and compound doses, so it cannot preserve the required lexical units.",
)
def test_preserves_compound_dose_and_disease_type():
    tokens = tokenize_medical_text("2型糖尿病 阿司匹林100mg")
    assert "2型糖尿病" in tokens
    assert "100mg" in tokens
