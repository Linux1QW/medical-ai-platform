# V1.2 Production Runbook

> Operational procedures for deploying, monitoring, and rolling back V1.2.
> Companion: `docs/release-evidence/v1.2/production-v1.2.md` for API/security behavior.

## 1. Architecture Overview

```text
Browser → React/Nginx → FastAPI (Uvicorn ×N)
                         ├─ SQLAlchemy async ── MySQL 8
                         ├─ LangGraph Coach ── redis-state checkpoint (db=0)
                         ├─ LangGraph Evaluation ── redis-state checkpoint (db=0)
                         ├─ Outbox enqueue ── MySQL evaluation_dispatch_outbox
                         ├─ Celery submit ── redis-state broker(db=4)/result(db=5)
                         ├─ JWT/cache/control ── redis-state (db=2/3/6/7/8)
                         ├─ LLM cache ── redis-cache (db=0)
                         ├─ Retrieval cache ── redis-cache (db=1)
                         └─ RAG ── Chroma + BM25 + optional Sparse

Evaluation Dispatcher (independent process)
  └─ Outbox poll → claim lease → Celery publish → acknowledge

Celery Worker (fork/prefork)
  ├─ run_evaluation
  ├─ RAG index tasks
  ├─ cleanup_expired_records
  └─ Pub/Sub listener for generation switch

Celery Beat ── daily cleanup (single instance)
Prometheus/Grafana ── optional monitoring profile
```

### Visible-Context Boundary

The Coach subsystem enforces a strict visible-context boundary:

- **CoachContextView** (`extra="forbid"`): Rejects hidden fields (`expected_diagnosis`, `system_prompt`, gold labels) at construction time.
- **CoachContextBuilder**: Only loads public VirtualPatient fields (age, gender, chief_complaint), never standard answers.
- **WorkingMemoryState**: `validate_memory_sources()` raises `HiddenContextViolation` if any slot references non-visible sequences.
- **Trainee Profile Memory**: At most 5 approved, consented, non-expired entries loaded per context compilation.

### LangGraph Nodes

**Evaluation graph** (14 nodes):
```
START → load_context → classify_consultation → safety_check
  → plan_evaluation → validate_plan
  → Wave 1: knowledge / inquiry / humanistic (parallel via Send)
  → extract_knowledge_citations
  → Wave 2: diagnosis / treatment (parallel via Send, citations injected)
  → aggregate_results → deterministic_scoring
  → reflection_check → review_gate_node
  → generate_suggestion → finalize_completed → END
  (or finalize_needs_review → END)
```

**Coach graph** (7 nodes):
```
intent → planner → evidence → draft → critic → finalize → persist
```

### MySQL Responsibilities

- All persistent state: users, consultations, evaluations, runs, outbox, coach sessions/decisions/events, trainee memories, prompt bundles, experiment assignments.
- Transactional Outbox guarantees at-least-once delivery to Celery.

### Redis Responsibilities

| Instance | Strategy |承载 |
|---|---|---|
| `redis-state` | AOF + noeviction | Checkpoint (db=0，RedisVL/RediSearch 要求), JWT blacklist (db=2/3/7), broker (db=4), result (db=5), progress bus (db=6), evaluation control (db=8) |
| `redis-cache` | allkeys-LRU | LLM response cache (db=0), retrieval cache (db=1) |

## 2. Failure Modes

| Failure | Symptom | Auto-recovery | Manual action |
|---|---|---|---|
| MySQL unreachable | `/health` → 503; all writes fail | SQLAlchemy pool recycle | Check MySQL service, network |
| redis-state down | Checkpoint/broker/JWT fail; Coach 500 | JWT blacklist fail-open (dev) / fail-closed (prod) | Restore Redis; if lost, active sessions invalidated |
| redis-cache down | LLM/RAG cache miss; increased latency | Degrades to direct LLM/RAG calls | Restore Redis; no data loss |
| LLM provider timeout | Agent timeout → error envelope → needs_review | Celery retry (2×, 30s/60s backoff) | Check provider status; review run trace |
| RAG generation mismatch | Stale results or 503 | Worker reconciles every 5s against active pointer | Verify manifest, BM25 READY, Chroma count |
| Coach LLM failure | SSE returns `degraded` event | Critic node catches errors; suggestion marked degraded | Check `COACH_HARD_TIMEOUT_SECONDS`; review trace |
| Dispatcher down | Evaluations stay `queued` | Outbox lease expiry → reconciliation re-queues | Start `evaluation-dispatcher` service |
| Celery Worker crash | Running evaluations → `retrying` | Reconciliation detects stale lease → re-queue or fail | Check Worker logs; checkpoint resume on retry |
| Voice LiveKit unreachable | `/voice/sessions` → 503 | No auto-recovery; voice is optional | Check LiveKit credentials and network |

## 3. Multi-Worker Consistency

- **RAG generation switch**: Redis CAS (`rag:active_generation`) + Pub/Sub (`rag:index-switched`). Each fork Worker loads and validates new generation atomically; failure retains old reference. Listener reconciles every 5s.
- **Coach sessions**: `SELECT … FOR UPDATE` for turn reservation; idempotency_key prevents duplicate suggestions across Workers.
- **Outbox dispatch**: Single Dispatcher process (must be single instance). Multiple Workers consume Celery tasks but only one dispatches.
- **Beat**: Must be single instance. Uses `--concurrency=2` in base Compose.

## 4. Model/RAG Degradation

| Scenario | Behavior |
|---|---|
| LLM provider unreachable | Agent error envelope → needs_review; Celery retry with backoff |
| Reranker failure | Falls back to pre-rerank ordering |
| BM25 load failure | Worker retains previous generation; logs warning |
| Chroma count mismatch | Generation validation fails; Worker keeps old reference |
| Coach evidence retrieval failure | Falls back to MCP demo fixtures (static, deidentified) |
| Embedding model change | New generation required; old generation remains active until CAS switch |

## 5. Memory Consent & Prompt Approval

### Trainee Memory Consent

- Consent defaults to **false** (opt-in).
- Doctor must explicitly call `PUT /trainee-memory/me/consent` with `{consent: true}`.
- Only approved + consented + non-expired memories enter Coach context.
- Max 5 memories loaded per context compilation.
- PHI validation: `_validate_deidentification()` rejects SSN, credit card, email, IP patterns.

### Prompt Approval

- Prompt bundles (`PromptBundle`) have lifecycle: `draft → active → deprecated`.
- Experiment assignments link doctors to specific prompt bundles for A/B testing.
- Admin must approve prompt bundles via `prompt:manage` permission.

## 6. Deployment Procedure

### Pre-deployment

1. **Backup**: `mysqldump` full database; `redis-state` and `redis-cache` RDB/AOF snapshots.
2. **Migration dry run**: On a replica or staging copy:
   ```bash
   alembic current
   alembic upgrade head
   alembic check
   ```
3. Verify V1.2 migration creates expected tables: `coach_sessions`, `coach_decisions`, `coach_stream_events`, `trainee_memories`, `trainee_memory_consents`, `prompt_bundles`, `experiment_assignments`, `agent_trace_events`.

### Deployment Order

```
1. Backup (MySQL + Redis)
2. Migration dry run (staging/replica)
3. alembic upgrade head (production)
4. App rollout with COACH_ENABLED=false
   - Deploy new Docker image / restart services
   - Verify /health returns 200
   - Verify existing evaluation flow works
5. Smoke test
   - Login/logout flow
   - Create consultation, send messages, trigger evaluation
   - Verify WebSocket progress
   - Check /metrics endpoint
6. RC probe (release candidate validation)
   - Run evaluation end-to-end with real LLM
   - Verify 5-dimension scoring output
   - Check RAG citations and evidence traces
7. Enable Coach for 5% traffic
   - Set COACH_ENABLED=true
   - Enable for test group (admin users first)
   - Monitor error rates, latency, SSE reconnects
8. Monitor 24h
   - Watch /metrics, Grafana dashboards
   - Check coach attribution flywheel
   - Verify no hidden-fact leakage in traces
9. Advance to 25%
10. Monitor 24h
11. Advance to 100%
```

### Service Restart Order

1. `migrate` (one-shot, completes before backend starts)
2. `redis-state`, `redis-cache` (if restarted)
3. `mysql` (if restarted)
4. `evaluation-dispatcher`
5. `celery-worker` (fork Workers start Pub/Sub listeners)
6. `celery-beat` (single instance)
7. `backend` (FastAPI)
8. `frontend` / Nginx

## 7. Rollback

> **See also**: `docs/release-evidence/v1.2/rollback-drill.md` for the complete step-by-step rollback procedure with RTO target.

### App Rollback (schema-compatible)

V1.2 tables are **additive** — they do not modify existing V1.1 tables. App rollback:

1. Set `COACH_ENABLED=false`
2. Revert to previous Docker image / git tag
3. Restart services
4. V1.2 tables remain in DB but are unused; no data migration needed

### Forward-Only Schema Compatibility

If future V1.3 adds columns to V1.2 tables, rollback to V1.2 requires:
- V1.3 migration must be backward-compatible (additive columns only, no drops/renames)
- If V1.3 breaks compatibility, rollback requires restoring from pre-V1.3 backup

### RAG Generation Rollback

No REST API for RAG rollback. Manual procedure:
1. Verify old generation's full artifact set (Chroma + BM25 + optional Sparse + manifest)
2. CAS `rag:active_generation` from current to target old generation
3. Publish `rag:index-switched` event with correct manifest SHA-256
4. Verify each Worker reconciles within 5s

## 8. Voice (Optional, NOT ACCEPTED for V1.2)

LiveKit voice integration is **optional beta** and **NOT ACCEPTED** for V1.2 release:

- **Status**: NOT_ACCEPTED — no real LiveKit credentials configured
- **Default**: `VOICE_ENABLED=false`
- **Not required** for core evaluation functionality
- Room tokens generated via LiveKit SDK JWT builder; API secret never logged
- Token TTL capped at 600 seconds; max 4 participants
- Partial transcripts: memory-only, never persisted to database
- Final transcripts: deduplicated by `(room_sid, participant_sid, turn_id)`
- Raw audio recording: **disabled by design**
- Does not feed into evaluation pipeline
- Deidentified: no patient identity in transcript processing

## 9. MCP Demo Server (Optional, Local stdio, Deidentified, Read-only)

MCP demo server is **optional** and **demonstration-only**:

- Communication: **stdio-only** (JSON-RPC 2.0), no network binding
- Two tools: `search_teaching_rubric`, `search_medical_kb`
- All data: static fixtures, deidentified, read-only
- No connection to production databases
- Used as fallback when real RAG retrieval is unavailable
- All outputs wrapped as `UNTRUSTED_EVIDENCE` by policy layer

## 10. Monitoring Checklist

| Check | Command / Endpoint | Expected |
|---|---|---|
| Health | `GET /health` | 200; MySQL + Redis connected |
| Metrics | `GET /metrics` (with `METRICS_TOKEN`) | 200; Prometheus format |
| Coach status | `POST /coach/.../stream` | 200 SSE or 409 if disabled |
| Evaluation flow | Create + evaluate consultation | Run completes within timeout |
| Outbox lag | `SELECT COUNT(*) FROM evaluation_dispatch_outbox WHERE status='pending'` | Near 0 |
| Worker heartbeat | Celery inspect ping | All Workers respond |
| RAG generation | `GET /knowledge-base/rebuild/status` | Active generation matches Redis pointer |
| Redis memory | `INFO memory` | Below `maxmemory` threshold |

## 11. Environment Variables (V1.2 additions)

| Variable | Default | Required | Description |
|---|---|---|---|
| `COACH_ENABLED` | `false` | No | Enable coach feature |
| `COACH_HMAC_KEY` | unset | Yes if Coach enabled | HMAC key for telemetry (≥32 bytes) |
| `COACH_CONTEXT_TOKEN_LIMIT` | `16000` | No | Context window budget |
| `COACH_OUTPUT_TOKEN_LIMIT` | `250` | No | Max output tokens per suggestion |
| `COACH_HARD_TIMEOUT_SECONDS` | `8` | No | Per-turn hard timeout |
| `COACH_SSE_EVENT_TTL_SECONDS` | `3600` | No | SSE event persistence TTL |
| `LIVEKIT_URL` | unset | No | LiveKit server URL |
| `LIVEKIT_API_KEY` | unset | No | LiveKit API key |
| `LIVEKIT_API_SECRET` | unset | No | LiveKit API secret |
| `FEEDBACK_EXPORT_HMAC_KEY` | unset | No | HMAC for feedback export deidentification |
