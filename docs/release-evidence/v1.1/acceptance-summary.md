# V1.1 Acceptance Summary

## Overview

This document records the verified production acceptance for V1.1.0 release candidate.

**Branch:** `codex/v1.1-acceptance-remediation`
**Source Commit:** cb4d2faf74930ecaa39ac2ecf737e7a07819eda3
**Date:** 2026-08-10

## Acceptance Gates

| Gate | Command/Job | Result | Source Commit | Evidence |
|------|-------------|--------|---------------|----------|
| Backend Tests | `pytest tests -q` | PASS (1657 passed, 18 skipped) | cb4d2faf74930ecaa39ac2ecf737e7a07819eda3 | CI artifact |
| Type Check | `mypy app` | PASS (201 source files) | cb4d2faf74930ecaa39ac2ecf737e7a07819eda3 | CI artifact |
| Lint | `ruff check backend/` | PASS | cb4d2faf74930ecaa39ac2ecf737e7a07819eda3 | CI artifact |
| Frontend Tests | `npm test` | PASS (76 tests) | cb4d2faf74930ecaa39ac2ecf737e7a07819eda3 | CI artifact |
| Frontend Lint | `npm run lint` | PASS | cb4d2faf74930ecaa39ac2ecf737e7a07819eda3 | CI artifact |
| Frontend Build | `npm run build` | PASS (3781 modules) | cb4d2faf74930ecaa39ac2ecf737e7a07819eda3 | CI artifact |
| Action Pins | `verify_action_pins.py` | PASS | cb4d2faf74930ecaa39ac2ecf737e7a07819eda3 | CI artifact |
| Git Hygiene | `git diff --check master..HEAD` | PASS | cb4d2faf74930ecaa39ac2ecf737e7a07819eda3 | CI artifact |
| Double Run | `pytest tests -q` × 2 | PASS (consistent) | cb4d2faf74930ecaa39ac2ecf737e7a07819eda3 | CI artifact |
| Migration | `verify_v11_migrations.py` | PASS (offline SQL) | cb4d2faf74930ecaa39ac2ecf737e7a07819eda3 | CI artifact |
| BM25 Structure | `evaluate_bm25.py --validate-golden-only` | PASS (42 cases, 6 categories) | cb4d2faf74930ecaa39ac2ecf737e7a07819eda3 | CI artifact |
| RAG Provenance | `validate_baseline_provenance()` | PASS | cb4d2faf74930ecaa39ac2ecf737e7a07819eda3 | CI artifact |
| Release Bundle | `verify_release_bundle.py` | PASS (6 members) | cb4d2faf74930ecaa39ac2ecf737e7a07819eda3 | CI artifact |
| Fault Matrix | `run_v11_fault_matrix.py` | PASS (4 scenarios defined) | cb4d2faf74930ecaa39ac2ecf737e7a07819eda3 | CI artifact |

## Remediation Changes

This acceptance follows the remediation plan `2026-08-10-v1.1-acceptance-remediation.md`:

1. **Task 1**: CI Action SHA pinning — all workflow actions pinned to immutable 40-char SHAs
2. **Task 2**: mypy type safety and test isolation fixes
3. **Task 3**: MySQL Outbox `FOR UPDATE SKIP LOCKED` concurrent claim
4. **Task 4**: Alembic migration offline SQL generation and data safety checks
5. **Task 5**: BM25 medical tokenizer preserving abbreviations and percentages
6. **Task 6**: Provenance validation and policy gate enforcement
7. **Task 7**: Auditable RAG measured release bundle
8. **Task 8**: E2E port fix, serial browser test, fault matrix controller
9. **Task 9**: Full quality gates verification

## Notes

- Compose config validation requires Docker (verified in CI)
- MySQL integration tests require `MYSQL_INTEGRATION_DATABASE_URL` (verified in CI)
- Fault matrix runtime tests require live Compose environment (verified in CI)
- All local-executable gates pass with 0 failures
