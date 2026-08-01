# Combined RAG/BM25 Tasks 1-4 Final Fix Report

## Result

Status: DONE

Base commit: `15bb8a3c8a30c7b10632e04070f15469ab696e44`

The final-review findings were closed in one focused change without changing
Dense retrieval semantics, Redis/Celery/generation scope, dependencies, or
external tool APIs. The pre-existing Ruff import-order edits in
`test_bm25_tokenizer.py` and `test_retriever.py` were preserved.

## Fixes

- Tokenizer protection now preserves HGVS cDNA variants, two-segment dose
  units, standalone percentages, pressure/lab values, and clearance values.
  The exact contracts cover `c.2573T>G`, `0.4U/kg/d`, `50mg/kg/d`,
  `180μmol/L`, `7.5%`, `55mmHg`, and `28ml/min`.
- The tokenizer identity was advanced to `medical-lexical-v3`; existing
  EGFR/eGFR canonicalization, no-space renal units, negations, punctuation
  boundaries, ICD forms, and protein variants remain covered.
- `BM25ArtifactManifest` now records and load-validates
  `enable_cjk_bigram`, `heading_boost`, and `entity_boost`, corresponding to
  `BM25_ENABLE_CJK_BIGRAM`, `BM25_HEADING_BOOST`, and
  `BM25_ENTITY_BOOST`. Focused tests prove each mismatch is rejected.
- BM25 required-token diagnostics now tokenize both the query and each golden
  expectation before comparison. Canonical values for EGFR, eGFR, L858R, and
  ≥50% therefore pass without raw-string false failures, while a real omitted
  eGFR expectation fails.
- Evaluation reports now contain a top-level `token_preservation` result.
  `--fail-on-token-loss` returns exit code 1 after writing the report when
  preservation fails; omitting the flag retains historical baseline generation
  with exit code 0.
- A tiered-retrieval boundary test proves the original `RetrievalQuery.text`,
  including whitespace and protected clinical forms, reaches `hybrid_recall`
  unchanged.

## TDD Evidence

### RED

Command:

```text
cd backend
python -m pytest tests/rag/test_bm25_tokenizer.py::test_extended_clinical_forms_are_complete_canonical_tokens tests/rag/test_bm25_artifacts.py::test_artifact_round_trip_uses_native_mmap_and_preserves_result_shape tests/rag/test_bm25_artifacts.py::test_load_rejects_manifest_identity_and_config_mismatch tests/rag/test_bm25_index.py::test_required_token_diagnostics_compare_canonical_values tests/rag/test_bm25_index.py::test_required_token_diagnostics_report_real_omission tests/rag/test_bm25_index.py::test_token_preservation_gate_is_opt_in_for_historical_baselines tests/rag/test_retriever.py::test_tiered_retrieve_passes_original_query_text_to_hybrid_recall -q
```

Exact result before implementation:

```text
14 failed, 7 passed in 4.07s
```

The seven tokenizer examples split, the manifest omitted/ignored all three new
identity fields, and the canonical diagnostic/preservation helpers did not
exist. The retrieval-boundary test passed immediately, confirming the finding
required a regression contract rather than a production behavior change.

### GREEN

The same focused RED command after implementation:

```text
21 passed in 3.23s
```

Focused source/test files:

```text
cd backend
python -m pytest tests/rag/test_bm25_tokenizer.py tests/rag/test_bm25_artifacts.py tests/rag/test_bm25_index.py tests/rag/test_retriever.py -q
62 passed in 3.57s
```

Full RAG suite:

```text
cd backend
python -m pytest tests/rag -q
139 passed in 3.74s
```

Ruff on every changed Python file:

```text
cd backend
python -m ruff check app/core/config.py app/services/rag/lexical/tokenizer.py app/services/rag/lexical/artifacts.py scripts/eval/evaluate_bm25.py tests/rag/test_bm25_tokenizer.py tests/rag/test_bm25_artifacts.py tests/rag/test_bm25_index.py tests/rag/test_retriever.py
All checks passed!
```

All pytest runs emitted one environment warning from `requests` about the
installed urllib3/chardet/charset_normalizer versions; it did not affect test
results and no dependencies were changed.

## Preservation Diagnostic

A read-only canonical-token scan of all 42 golden cases removed the named
EGFR/eGFR/L858R/≥50% false failures. It reports one genuine current
segmentation loss in `negation_no_allergy`: required `青霉素过敏` is tokenized
in query context as `青霉素` plus `过敏史`. This demonstrates the intended
distinction: normal baseline generation remains available, while
`--fail-on-token-loss` can enforce a failing preservation gate.
