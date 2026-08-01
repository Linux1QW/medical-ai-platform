"""Medical-aware lexical tokenizer used by BM25 retrieval."""

import logging
import re
from pathlib import Path
from typing import Literal

import jieba

from app.core.config import settings

logger = logging.getLogger(__name__)

TOKENIZER_VERSION = "medical-lexical-v3"
NEGATIONS = frozenset({"无", "不", "未", "否认", "排除"})
CASE_SENSITIVE_TERMS = {"EGFR": "gene:EGFR", "eGFR": "renal:eGFR"}
_NEGATION_TERMS = tuple(sorted(NEGATIONS, key=len, reverse=True))

MEDICAL_STOPWORDS = frozenset(
    {
        "的", "了", "是", "在", "有", "和", "与", "及", "或", "等", "为", "被", "把", "将",
        "从", "到", "对", "以", "可", "也", "就", "都", "而", "且", "但", "则", "要", "能",
        "会", "应", "该", "其", "这", "那", "之", "于", "中", "上", "下", "已", "所", "如",
        "若", "因", "由", "时", "后", "前", "间", "内", "外", "者", "用", "需", "可以", "应该",
    }
)

_CJK_SPAN_PATTERN = re.compile(r"[\u4e00-\u9fff]+")
_PROTECTED_PATTERN = re.compile(
    r"(?P<disease_type>(?<!\d)\d+(?:\.\d+)?[型期级][\u4e00-\u9fff]{2,}?(?:综合征|病|癌|症|炎))"
    r"|(?P<exon>\d+(?:-\d+)?外显子(?:缺失|突变|插入|重复)?)"
    r"|(?P<case_term>(?<![A-Za-z0-9])(?:EGFR|eGFR)(?![A-Za-z0-9]))"
    r"|(?P<pdl1>(?<![A-Za-z0-9])PD-?L1(?![A-Za-z0-9]))"
    r"|(?P<cdna_variant>(?<![A-Za-z0-9])c\.\d+(?:[+-]\d+)?(?:_\d+(?:[+-]\d+)?)?"
    r"(?:[ACGT]+>[ACGT]+|delins[ACGT]+|del[ACGT]*|dup[ACGT]*|ins[ACGT]+)(?![A-Za-z0-9]))"
    r"|(?P<variant>(?<![A-Za-z0-9])(?:p\.)?[A-Z]\d{1,5}[A-Z*](?![A-Za-z0-9]))"
    r"|(?P<icd>(?<![A-Za-z0-9])[A-Z]\d{2}(?:\.\d{1,4})?(?![A-Za-z0-9]))"
    r"|(?P<threshold>(?:>=|≥|>)\s*\d+(?:\.\d+)?%)"
    r"|(?P<percentage>(?<![A-Za-z0-9])\d+(?:\.\d+)?%(?![A-Za-z0-9]))"
    r"|(?P<renal_measurement>\d+(?:\.\d+)?\s*(?:m[lL]|μL|uL)/(?:min|h|d)/(?:\d+(?:\.\d+)?m2)(?![A-Za-z0-9]))"
    r"|(?P<compound_unit>(?<![A-Za-z0-9])(?:m[lL]|μL|uL)/(?:min|h|d)/(?:\d+(?:\.\d+)?m2)(?![A-Za-z0-9]))"
    r"|(?P<clinical_measurement>(?<![A-Za-z0-9])\d+(?:\.\d+)?\s?"
    r"(?:mmHg|kPa|(?:mg|g|mmol|μmol|umol|mEq)/(?:dL|L)|(?:mL|ml)/(?:min|h))"
    r"(?![A-Za-z0-9]))"
    r"|(?P<dose>\d+(?:\.\d+)?\s?(?:mg|g|μg|ug|mcg|mL|ml|IU|U|mmol|mol)"
    r"(?:/(?:kg|d|day|h)){0,2}(?![A-Za-z0-9/]))"
    r"|(?P<number>\d+(?:\.\d+)?)"
    r"|(?P<alphanumeric>(?<![A-Za-z0-9])[A-Za-z]+(?:[-_/][A-Za-z0-9]+)*\d*[A-Za-z0-9]*(?![A-Za-z0-9]))",
)


def _load_medical_dictionary() -> None:
    dictionary_path = Path(__file__).resolve().parents[4] / "data" / "medical_dict.txt"
    if dictionary_path.exists():
        jieba.load_userdict(str(dictionary_path))
        logger.info("Loaded jieba medical dictionary: %s", dictionary_path)
    else:
        logger.warning("Medical dictionary does not exist: %s", dictionary_path)


_load_medical_dictionary()


def stable_unique(tokens: list[str]) -> list[str]:
    """Return tokens once each while preserving their first-seen order."""
    return list(dict.fromkeys(token for token in tokens if token))


def _normalize_protected_term(match: re.Match[str]) -> list[str]:
    kind = match.lastgroup
    value = match.group()

    if kind == "case_term":
        return [CASE_SENSITIVE_TERMS[value]]
    if kind == "pdl1":
        return ["PD-L1"]
    if kind == "variant":
        return [f"variant:{value.removeprefix('p.')}"]
    if kind == "threshold":
        threshold = value.removeprefix(">=").removeprefix("≥").removeprefix(">")
        return [f">={threshold.strip()}"]
    if kind == "renal_measurement":
        number_match = re.match(r"\d+(?:\.\d+)?", value)
        assert number_match is not None
        return [number_match.group(), value[number_match.end():].lstrip()]
    if kind in {"clinical_measurement", "dose"}:
        return [re.sub(r"\s+", "", value)]
    return [value]


def extract_protected_terms(text: str) -> tuple[list[str], list[str]]:
    """Extract clinical terms and leave only safe residual spans for jieba."""
    protected: list[str] = []
    residual_parts: list[str] = []
    cursor = 0

    for match in _PROTECTED_PATTERN.finditer(text):
        residual_parts.append(text[cursor:match.start()])
        protected.extend(_normalize_protected_term(match))
        residual_parts.append(" ")
        cursor = match.end()

    residual_parts.append(text[cursor:])
    return protected, residual_parts


def _tokenize_cjk_span(span: str) -> list[str]:
    words = [word.strip() for word in jieba.cut(span) if word.strip()]
    negations: list[str] = []
    position = 0
    while position < len(span):
        matched = next(
            (term for term in _NEGATION_TERMS if span.startswith(term, position)),
            None,
        )
        if matched:
            negations.append(matched)
            position += len(matched)
        else:
            position += 1

    tokens = negations + [
        word
        for word in words
        if word not in MEDICAL_STOPWORDS and (len(word) > 1 or word in NEGATIONS)
    ]

    if (
        settings.BM25_ENABLE_CJK_BIGRAM
        and 3 <= len(span) <= 8
        and words == [span]
        and jieba.get_FREQ(span) is None
    ):
        tokens.extend(f"bg:{span[index:index + 2]}" for index in range(len(span) - 1))

    return tokens


def tokenize_cjk_spans(residual_spans: list[str]) -> list[str]:
    """Segment CJK-only residual spans without crossing punctuation boundaries."""
    tokens: list[str] = []
    for residual in residual_spans:
        for span in _CJK_SPAN_PATTERN.findall(residual):
            tokens.extend(_tokenize_cjk_span(span))
    return tokens


def tokenize_medical_text(
    text: str,
    *,
    mode: Literal["document", "query"] = "document",
) -> list[str]:
    """Tokenize mixed Chinese/Latin medical text for lexical retrieval."""
    if mode not in {"document", "query"}:
        raise ValueError("mode must be either 'document' or 'query'")
    if not text or not text.strip():
        return []

    protected, residual_spans = extract_protected_terms(text)
    chinese_tokens = tokenize_cjk_spans(residual_spans)
    return stable_unique(protected + chinese_tokens)
