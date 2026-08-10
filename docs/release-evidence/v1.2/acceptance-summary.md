# V1.2 Acceptance Summary

> **Status**: Code-complete on branch `codex/v1.2-agent-intelligence-remediation`.
> Metrics marked **TBD** require live-environment benchmark runs; they are NOT claimed as passed.

## Milestones Delivered (code-complete)

### M1: Runtime Foundation (Task 0–4)

| Deliverable | Source location | Status |
|---|---|---|
| ADR-002: Coach Context Partition | `docs/adr/` | Accepted |
| CoachContextView contract (`extra="forbid"`) | `app/agent_runtime/contracts.py` | Implemented + tested |
| CoachSuggestion contract | `app/agent_runtime/contracts.py` | Implemented + tested |
| Coach persistence (6 models) | `app/models/coach_session.py`, `coach_decision.py`, `coach_stream_event.py`, `trainee_memory.py`, `experiment_assignment.py`, `PromptBundle.py` | ORM + migration |
| Alembic migration | `4d5e6f7a8b9c_v12_agent_runtime.py`, `5e6f7a8b9c0d_v12_runtime_remediation.py` | upgrade/downgrade tested |
| Agent Telemetry (privacy-safe) | `app/models/agent_trace_event.py` | 12 event types, HMAC-signed |
| Working Memory + Context Compiler | `app/services/memory/working.py`, `app/agent_runtime/context.py` | 16K budget, hidden-fact validation |

### M2: Intelligent Coach (Task 5–10)

| Deliverable | Source location | Status |
|---|---|---|
| Intent Recognition (rules-first + model fallback) | `app/services/agents/coach/intent.py` | Implemented + tested |
| Follow-up Planner | `app/services/agents/coach/planner.py` | Implemented + tested |
| Skill Registry + Policy Enforcement | `app/agent_runtime/skill_executor.py`, `app/agent_runtime/policy.py` | UNTRUSTED_EVIDENCE wrapping |
| MCP Demo Server (read-only, deidentified, stdio) | `app/mcp_demo/server.py`, `app/mcp_demo/__main__.py` | JSON-RPC 2.0, no network binding |
| 7-node Coach LangGraph | `app/agent_runtime/graph.py` | intent → planner → evidence → draft → critic → finalize → persist |
| Coach API + SSE + Feedback | `app/api/v1/coach.py` | Idempotency key, Last-Event-ID replay, 15s heartbeat |
| Coach Frontend UI (6 states, no auto-submit) | `frontend/src/components/CoachPanel.tsx`, `frontend/src/hooks/useCoachSuggestion.ts` | disabled/idle/thinking/suggestion/degraded/error |

### M3: Data Flywheel (Task 11–14)

| Deliverable | Source location | Status |
|---|---|---|
| Trainee Profile Memory (approval lifecycle) | `app/services/memory/profile.py`, `app/api/v1/trainee_memory.py` | candidate → approved/rejected/expired, consent defaults false |
| Coach Attribution Flywheel | `evaluation/coach_attribution.py` | admin_reviewed AND deidentified → eligible |
| Prompt Registry + A/B Rollout | `app/models/PromptBundle.py`, `app/models/experiment_assignment.py` | PromptBundle + ExperimentAssignment ORM |

### M4: Realtime + Release (Task 15–16)

| Deliverable | Source location | Status |
|---|---|---|
| LiveKit Voice Beta | `app/voice/session.py`, `app/voice/agent.py`, `app/api/v1/voice.py` | Optional, no raw audio saved |
| Documentation + Evidence | This directory | Present |

## Release Metrics

Every metric below requires a live-environment run to fill the **Actual** column.
Thresholds are defined in the V1.2 iteration plan. Source artifacts are identified by commit SHA and dataset/prompt hash.

| Metric | Threshold | Actual | Source SHA | Dataset Hash | Prompt Hash | Workflow Run URL |
|---|---|---|---|---|---|---|
| Intent macro-F1 | ≥ 0.85 | **TBD** | — | — | — | — |
| Hidden-fact leakage | 0/72 | **TBD** | — | — | — | — |
| Unsafe suggestion | 0/72 | **TBD** | — | — | — | — |
| Text hint p95 latency | ≤ 2.5 s | **TBD** | — | — | — | — |
| Forbidden tool calls | 0 | **TBD** | — | — | — | — |
| Trace completeness | 100 % | **TBD** | — | — | — | — |

> **Note**: No metric is claimed as passed. The TBD rows must be populated by running the
> 72-case coach benchmark and safety probe in a deployed environment with real LLM endpoints.

## Safety Verification (code-level)

| Control | Implementation | Verified by |
|---|---|---|
| Coach default OFF | `COACH_ENABLED: bool = False` in `config.py` | `test_v12_production_config.py` |
| Context View isolation | `CoachContextView(extra="forbid")` rejects `expected_diagnosis` etc. | `test_contracts.py` |
| Hidden-fact rejection | `validate_memory_sources()` raises `HiddenContextViolation` | `test_working_memory.py` |
| Critic agent checks all suggestions | `critic_node` in `app/agent_runtime/nodes.py` | `test_critic.py` |
| All tool outputs marked untrusted | `mark_output_untrusted()` in policy layer | `test_skill_executor.py` |
| Voice does not save raw audio | `persisted=False` for partial; no recording in `VoiceAgent` | `test_voice_agent.py` |
| Trainee memory consent defaults false | `TraineeMemoryConsent.consent = False` | `test_v12_runtime_constraints.py` |
| IDOR guard on coach endpoints | `_verify_consultation_ownership()` in `coach.py` | `test_coach_api.py` |
| PHI rejection in profile memory | `_validate_deidentification()` regex patterns | `test_profile_memory.py` |
