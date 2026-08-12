# V1.2 Production Guide — API & Security Behavior

> This document describes the **exact** runtime behavior of V1.2.
> It is authoritative for deployment, security review, and acceptance.
> Companion: `docs/runbooks/production-v1.2.md` for operational procedures.

## 1. Authentication & Authorization

| Layer | Mechanism | Failure mode |
|---|---|---|
| REST | `Authorization: Bearer <JWT>` (HS256) | 401 if missing/expired |
| WebSocket | First message `{"type":"auth","token":"<JWT>"}` within 5 s | Connection closed |
| JWT blacklist | Redis db=7; `JWT_BLACKLIST_FAIL_CLOSED=true` in staging/production | Fail open in dev; fail closed in staging/prod |
| Token expiry | `ACCESS_TOKEN_EXPIRE_MINUTES` default 1440 (dev), **max 60 in production** | 401 → frontend clears session |
| Refresh | `REFRESH_TOKEN_EXPIRE_DAYS=7` | Expired refresh requires re-login |

### RBAC

Roles: `doctor`, `admin`. Custom `users.permissions` overrides role defaults when non-empty.

| Permission | doctor | admin |
|---|---|---|
| `coach:use` | ✓ | ✓ |
| `coach:trace:view` | — | ✓ |
| `trainee-memory:manage-self` | ✓ | ✓ |
| `trainee-memory:review` | — | ✓ |
| `voice:use` | ✓ | ✓ |
| `prompt:manage` | — | ✓ |
| `experiment:manage` | — | ✓ |

## 2. Object Ownership (IDOR Prevention)

| Resource | Owner check | Admin bypass |
|---|---|---|
| Consultation | `require_consultation_access()` | Yes |
| Coach session | `_verify_consultation_ownership()` → 403 if `doctor_id` mismatch | Yes |
| Voice session | `require_consultation_access()` → 403 if non-owner | Yes |
| Trainee memory | Doctor can only CRUD own memories (`doctor_id` match) | Admin can review all |
| Evaluation | Owner or admin via `require_consultation_access` | Yes |

## 3. Request/Response Schemas

### Coach Endpoints

| Method | Path | Auth | Request | Response |
|---|---|---|---|---|
| POST | `/api/v1/coach/consultations/{id}/suggestions/stream` | `coach:use` | `CoachStreamRequest` (idempotency_key, message, turn_no) | SSE stream |
| GET | `/api/v1/coach/consultations/{id}/state` | `coach:use` | — | `CoachStateResponse` |
| POST | `/api/v1/coach/consultations/{id}/feedback` | `coach:use` | `SuggestionFeedbackRequest` | `SuggestionFeedbackResponse` |
| GET | `/api/v1/coach/admin/traces?session_id=` | `coach:trace:view` | — | Trace list |

### Voice Endpoints

| Method | Path | Auth | Request | Response |
|---|---|---|---|---|
| POST | `/api/v1/voice/sessions` | `voice:use` | `CreateVoiceSessionRequest(consultation_id)` | `VoiceSessionResponse` |
| GET | `/api/v1/voice/sessions/{room_name}/state` | `voice:use` | — | `VoiceStateResponse` |
| POST | `/api/v1/voice/sessions/{room_name}/end` | `voice:use` | — | 200 |

### Trainee Memory Endpoints

| Method | Path | Auth | Request | Response |
|---|---|---|---|---|
| POST | `/api/v1/trainee-memory/memories` | `trainee-memory:manage-self` | `{skill_dimension, summary, evidence_refs?}` | Memory object |
| GET | `/api/v1/trainee-memory/me/memories` | `trainee-memory:manage-self` | — | Memory list |
| PUT | `/api/v1/trainee-memory/me/consent` | `trainee-memory:manage-self` | `{consent: bool}` | Consent status |
| POST | `/api/v1/trainee-memory/memories/{id}/review` | `trainee-memory:review` | `{action: "approve"|"reject"}` | Updated memory |

### Prompt Registry Endpoints

| Method | Path | Auth | Request | Response |
|---|---|---|---|---|
| POST | `/api/v1/prompt-registry/bundles` | `prompt:manage` | Bundle definition | `BundleResponse` |
| GET | `/api/v1/prompt-registry/bundles/{name}/{version}` | `prompt:manage` | — | `BundleResponse` |
| GET | `/api/v1/prompt-registry/bundles/{name}/active` | `prompt:manage` | — | `BundleResponse` |
| POST | `/api/v1/prompt-registry/bundles/{name}/{version}/activate` | `prompt:manage` | — | `BundleResponse` |
| GET | `/api/v1/prompt-registry/bundles` | `prompt:manage` | — | `list[BundleResponse]` |
| POST | `/api/v1/prompt-registry/experiments` | `experiment:manage` | Experiment definition | `ExperimentResponse` |
| POST | `/api/v1/prompt-registry/experiments/{id}/assign` | `experiment:manage` | `{doctor_id}` | `AssignmentResponse` |

## 4. Idempotency

| Endpoint | Mechanism |
|---|---|
| Coach SSE stream | `idempotency_key` in request body; duplicate key returns stored decision |
| Coach turn reservation | `SELECT … FOR UPDATE` + idempotency_key unique check in `CoachRepository.reserve_turn()` |
| Evaluation submission | Outbox `evaluation_dispatch_outbox` with `run_id` unique constraint |

## 5. SSE Replay

Coach SSE endpoint supports replay via `Last-Event-ID` header:
- Events are persisted in `coach_stream_events` table with TTL `COACH_SSE_EVENT_TTL_SECONDS=3600`.
- Client sends `Last-Event-ID` to resume from a specific event; server replays missed events then continues live.
- Heartbeat comments every 15 seconds keep the connection alive.

## 6. Public Error Codes

All errors follow the standard envelope:

```json
{
  "error_code": "STRING_CODE",
  "message": "Human-readable description",
  "detail": {},
  "request_id": "uuid"
}
```

| HTTP | error_code | Trigger |
|---|---|---|
| 401 | `AUTH_TOKEN_EXPIRED` | JWT expired |
| 401 | `AUTH_TOKEN_INVALID` | Malformed/unknown JWT |
| 403 | `AUTH_PERMISSION_DENIED` | Missing required permission |
| 403 | `COACH_IDOR_DENIED` | Coach access to another doctor's consultation |
| 404 | `COACH_CONSULTATION_NOT_FOUND` | Consultation does not exist |
| 409 | `COACH_DISABLED` | Coach feature is disabled (`COACH_ENABLED=false`) |
| 409 | `COACH_SESSION_CONFLICT` | Concurrent session creation race |
| 429 | `RATE_LIMIT_EXCEEDED` | Rate limit hit |
| 503 | `VOICE_NOT_CONFIGURED` | LiveKit credentials missing |

## 7. Trace Redaction

| Component | Redaction rule |
|---|---|
| Agent telemetry events | HMAC-signed with `COACH_HMAC_KEY`; no raw PII |
| Langfuse (if enabled) | `OBSERVABILITY_CAPTURE_CONTENT=false` default; content capture forbidden in staging/prod |
| Audit logs | Detail field sanitized; `X-Request-ID` for correlation without PII |
| Voice transcripts | Only final transcripts persisted; partial transcripts memory-only; raw audio never saved |
| Dispatcher logs | Payload keys logged, values hidden |

## 8. Rate Limits

| Endpoint | Limit |
|---|---|
| `POST /auth/login` | 5/min |
| `POST /auth/register` | 5/min |
| `POST /consultations/{id}/messages` | 10/min |
| `POST /consultations/{id}/messages/stream` | 10/min |
| `POST /evaluations/` | 5/hour |
| Patient CRUD (admin) | 30/min |
| Coach SSE | Bound by `COACH_HARD_TIMEOUT_SECONDS=8` per turn |

## 9. Feature Flags

| Flag | Default | Effect |
|---|---|---|
| `COACH_ENABLED` | `false` | Coach opt-in; 409 on all coach endpoints when false |
| `LANGGRAPH_ENABLED` | `true` | Evaluation LangGraph graph; startup fails if checkpointer init fails |
| `ENABLE_TOOL_USE` | `true` (code) / `false` (base Compose) | Function Call for evaluation agents |
| `LIVEKIT_API_KEY` / `LIVEKIT_API_SECRET` | unset | Voice beta; 503 if unset and endpoint called |

## 10. Coach as Opt-in

The Coach subsystem is **disabled by default** and must be explicitly enabled:

1. Set `COACH_ENABLED=true`
2. Set `COACH_HMAC_KEY` (≥32 bytes) for telemetry signing
3. Run Alembic migration to create coach tables
4. Frontend shows Coach panel only when API returns non-409

When disabled, all coach endpoints return `409 COACH_DISABLED`. No background processing, no LLM calls, no telemetry.

## 11. Voice as Optional Beta

LiveKit voice is **optional** and **beta**:

- Requires `LIVEKIT_URL`, `LIVEKIT_API_KEY`, `LIVEKIT_API_SECRET`
- Room token TTL capped at 600 seconds
- Max 4 participants per room
- Partial transcripts: memory-only, never persisted
- Final transcripts: deduplicated by `(room_sid, participant_sid, turn_id)`
- Raw audio recording: **disabled**
- Does not integrate with evaluation pipeline
