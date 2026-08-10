# ADR-002: V1.2 Coach Context Partition and Safety Boundaries

## Status

Accepted

## Date

2026-08-10

## Context

V1.2 introduces a Coach Agent that assists doctors during simulated consultations. The Coach must never access hidden patient facts, expected diagnoses, or evaluation gold data that would compromise training integrity.

## Decision

We define four strict Context Views with explicit allowed/forbidden fields:

### PatientContextView
- **Allowed:** Full case data, personality, undisclosed facts, patient tools
- **Forbidden:** Trainee profile memory, evaluation gold data

### CoachContextView
- **Allowed:** Visible patient data, occurred dialogue, disclosed facts, teaching rubric, approved trainee profile
- **Forbidden:** Expected diagnosis, undisclosed facts, patient system prompt, evaluation answers

### EvaluationContextView
- **Allowed:** Dialogue, submitted diagnosis, teaching rubric, necessary ground truth
- **Forbidden:** User auth keys, unsanitized external trace

### TraceContextView
- **Allowed:** ID, version, latency, tokens, error codes, HMAC digests, controlled metadata
- **Forbidden:** Default no prompt/completion/patient raw text/audio

## Implementation

- Patient Agent, Coach Agent, Evaluation Agent use separate Pydantic models
- Cross-view fields transferred via explicit converter functions
- No direct ORM object or arbitrary dict passing between views
- Coach defaults OFF (`COACH_ENABLED=false`)
- Coach modes: `off`, `on_demand`, `shadow`
- `shadow` mode pre-computes but never auto-inserts or auto-sends
- No raw audio stored; Voice Beta saves only final transcript, latency metrics, error codes
- Trainee Memory defaults OFF; only stores human-approved capability profiles
- No online RL, auto SFT/DPO, auto prompt deployment, or unreviewed self-evolution

## Consequences

- Coach suggestions are bounded to visible information only
- Safety gate rejects any output containing hidden facts or unsafe advice
- All decisions go through schema validation, policy validator, and safety gate before reaching UI
- Trace content capture defaults off; only HMAC and character count stored
