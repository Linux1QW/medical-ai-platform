# V1.2 Acceptance Summary

> **Status**: NOT PASSED on branch `codex/v1.2-final-blockers-remediation`.
> **Candidate SHA**: `d047966ef428e072ba90929f33fa32fc0ea1c975`
> **Generated**: 2026-08-12T11:07:01.817700+00:00

## Milestones Delivered

### M1: Runtime Foundation (Task 0–4)

| Deliverable | Status |
|---|---|
| ADR-002: Coach Context Partition | Accepted |
| CoachContextView contract (`extra="forbid"`) | Implemented + tested |
| CoachSuggestion contract | Implemented + tested |
| Coach persistence (6 models) | ORM + migration |
| Alembic migration | upgrade/downgrade tested |
| Agent Telemetry (privacy-safe) | 12 event types, HMAC-signed |
| Working Memory + Context Compiler | 16K budget, hidden-fact validation |

### M2: Intelligent Coach (Task 5–10)

| Deliverable | Status |
|---|---|
| Intent Recognition (rules-first + model fallback) | Implemented + tested |
| Follow-up Planner | Implemented + tested |
| Skill Registry + Policy Enforcement | UNTRUSTED_EVIDENCE wrapping |
| MCP Demo Server (read-only, deidentified, stdio) | JSON-RPC 2.0, no network binding |
| 7-node Coach LangGraph | intent → planner → evidence → draft → critic → finalize → persist |
| Coach API + SSE + Feedback | Idempotency key, Last-Event-ID replay, 15s heartbeat |
| Coach Frontend UI (6 states, no auto-submit) | disabled/idle/thinking/suggestion/degraded/error |

### M3: Data Flywheel (Task 11–14)

| Deliverable | Status |
|---|---|
| Trainee Profile Memory (approval lifecycle) | candidate → approved/rejected/expired |
| Coach Attribution Flywheel | admin_reviewed AND deidentified → eligible |
| Prompt Registry + A/B Rollout | PromptBundle + ExperimentAssignment ORM |

### M4: Realtime + Release (Task 15–16)

| Deliverable | Status |
|---|---|
| LiveKit Voice Beta | **NOT ACCEPTED** — default OFF, no real LiveKit evidence |
| Documentation + Evidence | Present |

## Release Metrics

| Metric | Threshold | Actual | Status |
|---|---|---|---|
| Backend tests | ≥ 2000 | None | FAIL |
| Frontend tests | ≥ 100 | None | FAIL |
| mypy errors | 0 | None | FAIL |
| ruff errors | 0 | None | FAIL |
| Intent macro-F1 | ≥ 0.85 | TBD (live-only) | Pending |
| Hidden-fact leakage | 0/72 | TBD (live-only) | Pending |
| Text hint p95 latency | ≤ 2.5 s | TBD (live-only) | Pending |

## Voice Status

**NOT_ACCEPTED** — No real LiveKit credentials configured; Voice feature is opt-in and disabled by default (VOICE_ENABLED=false). Not included in acceptance criteria.

## Safety Verification

| Control | Verified |
|---|---|
| Coach default OFF | Yes — `COACH_ENABLED: bool = False` |
| Context View isolation | Yes — `extra="forbid"` rejects hidden fields |
| Hidden-fact rejection | Yes — `validate_memory_sources()` raises `HiddenContextViolation` |
| Critic agent checks all suggestions | Yes — `critic_node` in graph |
| All tool outputs marked untrusted | Yes — `mark_output_untrusted()` |
| Voice does not save raw audio | Yes — `persisted=False`, no recording |
| Trainee memory consent defaults false | Yes — opt-in required |
| IDOR guard on coach endpoints | Yes — `_verify_consultation_ownership()` |
| PHI rejection in profile memory | Yes — regex patterns |

---
*This document was auto-generated from `final-acceptance.json`. Do not edit manually.*
