# Task 4 Report: Versioned Persistent BM25 Artifacts

## Result

Implemented versioned native `bm25s==0.3.9` artifacts with staged publication,
manifest/corpus/config validation, mmap loading, generation-aware in-process
registration, and lock-protected candidate swaps. Existing `BM25Index.build()`,
`BM25Index.search()`, `get_bm25_index()`, and `rebuild_bm25_index()` callers remain
compatible.

No real 62,917-document artifact was built or committed. Artifact tests use
`tmp_path` only.

## Files

- `backend/app/services/rag/lexical/artifacts.py` — manifest, staged native save,
  READY-last publication, validation, and mmap load.
- `backend/app/services/rag/bm25_search.py` — local-token build, `token_count`,
  field boosts, legacy fallback, generation registry, and atomic swap.
- `backend/app/core/config.py` — artifact root and 1-3 boost constraints.
- `backend/tests/rag/test_bm25_artifacts.py` — round-trip, mmap, mismatch,
  integrity, READY, cleanup, protected-token, and config tests.
- `backend/tests/rag/test_bm25_index.py` — active-index preservation and swap
  tests.

## RED Evidence

Initial command:

```text
cd backend
python -m pytest tests/rag/test_bm25_artifacts.py tests/rag/test_bm25_index.py -v
```

Exact result:

```text
collected 0 items / 2 errors
ERROR tests/rag/test_bm25_artifacts.py
ImportError: cannot import name 'build_document_tokens' from
'app.services.rag.bm25_search'
ERROR tests/rag/test_bm25_index.py
ModuleNotFoundError: No module named 'app.services.rag.lexical.artifacts'
!!!!!!!!!!!!!!!!!!! Interrupted: 2 errors during collection !!!!!!!!!!!!!!!!!!!
```

Atomic failure-path reproduction:

```text
python -m pytest \
  tests/rag/test_bm25_index.py::test_get_validation_failure_keeps_serving_active_index \
  tests/rag/test_bm25_index.py::test_get_failed_legacy_build_keeps_serving_active_index -v
```

Exact result:

```text
collected 2 items
test_get_validation_failure_keeps_serving_active_index FAILED
test_get_failed_legacy_build_keeps_serving_active_index FAILED
=========================== 2 failed in 1.90s ===========================
```

Root cause: candidate failure returned only the requested generation cache and
discarded the valid active alias; an uninitialized legacy candidate was also
eligible for publication. The corrected path returns the active alias on failure
and never publishes an uninitialized candidate.

Preloaded-generation reproduction:

```text
python -m pytest \
  tests/rag/test_bm25_index.py::test_default_get_promotes_preloaded_generation_to_active_alias -v
```

Exact result:

```text
collected 1 item
test_default_get_promotes_preloaded_generation_to_active_alias FAILED
============================== 1 failed in 1.78s ==============================
```

Root cause: the cached-success branch returned the correct generation but did not
promote it to the compatibility alias. The corrected branch updates the alias and
generation together under `_registry_lock`.

## GREEN Evidence

Focused Task 4 tests:

```text
cd backend
python -m pytest tests/rag/test_bm25_artifacts.py tests/rag/test_bm25_index.py -v
============================= 27 passed in 1.66s ==============================
```

Full RAG tests:

```text
cd backend
python -m pytest tests/rag -v
============================= 121 passed in 3.33s =============================
```

Task 4 Ruff check:

```text
cd backend
python -m ruff check app/core/config.py app/services/rag/bm25_search.py \
  app/services/rag/lexical/artifacts.py tests/rag/test_bm25_artifacts.py \
  tests/rag/test_bm25_index.py
All checks passed!
```

## Validation Coverage

- Manifest carries generation, corpus SHA-256, document count, tokenizer
  version, bm25s version, method, k1, b, created_at, and token count.
- Load rejects generation, tokenizer, bm25s, method, k1, b, corpus hash/count,
  native index count/config, required-file, and READY mismatches.
- Native save is performed in a unique staging directory; a native reload and
  sample query complete before READY is written, and the directory is then
  promoted into the immutable generation path.
- Build failures clean staging and do not mutate the active registry or an
  existing generation artifact.
- `BM25Index` retains no `doc_tokens`; it exposes integer `token_count`.
- mmap-loaded corpus entries preserve the existing result dictionary fields and
  IDs.
- EGFR/eGFR, compound medical tokens, and lexical query expansion behavior remain
  covered by the complete RAG suite.

## Self-review and Concerns

- Manual diff review found no remaining Task 4 correctness or scope issues. The
  optional `ocr` review CLI was not installed, so no external AI review was run.
- A broad `python -m ruff check app tests/rag` is not clean because of two
  pre-existing import-order findings in untouched files:
  `tests/rag/test_bm25_tokenizer.py` and `tests/rag/test_retriever.py`. Task 4
  files pass Ruff, and `git diff --exit-code` confirms those two files were not
  modified.
- Pytest emits an environment-level `RequestsDependencyWarning` about installed
  `urllib3`/charset packages; all focused and full RAG tests pass.
- Generation artifacts are intentionally immutable: publishing over an existing
  `<generation>/bm25` directory is rejected instead of mutating a possibly mmap-
  backed source.
