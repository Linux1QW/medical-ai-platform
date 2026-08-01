"""BM25-only medical entity and code query expansion."""

from app.services.rag.entity_resolver import extract_entities
from app.services.rag.lexical.tokenizer import stable_unique


def expand_lexical_query(text: str, tokens: list[str]) -> list[str]:
    """Append entity canonical names and codes to already-tokenized BM25 queries."""
    expanded = list(tokens)
    for entity in extract_entities(text):
        normalized = entity.get("normalized")
        if normalized:
            expanded.append(normalized)
        for key in ("icd10", "atc", "icd9cm3"):
            if code := entity.get(key):
                expanded.append(f"{key}:{code}")
    return stable_unique(expanded)
