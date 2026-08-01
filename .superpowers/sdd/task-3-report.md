# Task 3 Report: Isolate lexical query expansion from dense retrieval

Status: DONE

Base commit: `767a646`

## RED evidence

1. `cd backend && python -m pytest tests/rag/test_retriever.py -v`
   - Result: `1 failed, 2 passed in 3.71s`.
   - Failure: the brief's original conflicting assertion expected the BM25 call argument to contain a canonicalized value, but BM25 received the raw string `"心梗治疗"`.
   - After the architecture clarification, this assertion was corrected to require the raw query for both channels; canonicalization is covered by the separate token-expansion test.
2. `cd backend && python -m pytest tests/rag/test_bm25_tokenizer.py -v`
   - Result: collection error because `app.services.rag.lexical.query_expansion` did not exist.

## GREEN evidence

1. `cd backend && python -m pytest tests/rag/test_retriever.py tests/rag/test_bm25_tokenizer.py -v`
   - Result: `17 passed in 2.59s`.
2. `cd backend && python -m pytest tests/rag -v`
   - Result: `100 passed in 2.84s`.

The test environment emitted an existing `RequestsDependencyWarning` about urllib3/chardet compatibility; it did not fail either test command.

## Files

- Added `backend/app/services/rag/lexical/query_expansion.py`.
- Updated `backend/app/services/rag/bm25_search.py` so `BM25Index.search` tokenizes the raw query, expands the resulting tokens, then calls `bm25s.retrieve`.
- Updated `backend/app/services/rag/retriever/tiered.py` to pass `query.text` directly to `hybrid_recall`.
- Updated `backend/tests/rag/test_retriever.py` to assert Dense and BM25 both receive the exact raw query.
- Updated `backend/tests/rag/test_bm25_tokenizer.py` to verify token expansion preserves original tokens and adds canonical names and codes.

## Self-review

- `hybrid_recall` continues to pass the exact raw query to Dense and `BM25Index.search`; the channel-isolation test covers both calls.
- Canonical names and `icd10`/`atc`/`icd9cm3` code tokens are appended only by the BM25 query path, after tokenizer V2 runs.
- `stable_unique` preserves protected tokenizer tokens and first-seen order; existing EGFR/eGFR and protected-token tests pass.
- Removed the whole-string `normalize_query` use from the tiered retrieval path. `normalize_query` remains in `entity_resolver.py` as a compatibility API.
- No public `medical_search` or `medical_rerank` parameter changes, new dependencies, external search services, or BM25 artifacts/generation were introduced.
- `git diff --check` passed.

## Commit

This report is included with commit subject: `fix: isolate lexical query expansion from dense retrieval`.
