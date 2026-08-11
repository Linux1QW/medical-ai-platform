# V1.2 Rollback Drill

> **Purpose**: Document the step-by-step rollback procedure for V1.2 deployment.
> **RTO Target**: 15 minutes
> **Last Drill**: 2026-08-11 (simulated — no live environment)

## Overview

This drill records the commands, expected outcomes, and actual results for rolling back
a V1.2 deployment to V1.1. Each step includes the command to execute, the expected result,
and the actual observed result.

## Pre-Requisites

- Access to production Kubernetes cluster or Docker Compose environment
- `kubectl` or `docker-compose` configured with production credentials
- Database backup from pre-V1.2 state available
- V1.1 Docker images tagged and available in registry
- V1.1 Prompt Bundle backup available (if Coach was enabled)

## Drill Steps

### Step 1: Coach 关闭状态部署 (Disable Coach)

| Item | Value |
|------|-------|
| **Command** | `kubectl set env deployment/medical-ai-backend COACH_ENABLED=false` |
| **Expected** | All Coach API endpoints return 409 Conflict |
| **Actual** | Pending live drill |
| **Time** | — |

**Verification**:
```bash
curl -s -o /dev/null -w "%{http_code}" https://prod.example.com/api/v1/coach/consultations/1/suggestions/stream
# Expected: 409
```

---

### Step 2: 开启 5% 流量观察 (Canary 5%)

| Item | Value |
|------|-------|
| **Command** | `kubectl set env deployment/medical-ai-backend COACH_ENABLED=true COACH_CANARY_PERCENT=5` |
| **Expected** | Only 5% of eligible requests receive Coach suggestions |
| **Actual** | Pending live drill |
| **Time** | — |

**Verification**:
```bash
# Monitor Coach suggestion rate
kubectl logs deployment/medical-ai-backend | grep "coach_suggestion_served" | wc -l
# Expected: ~5% of total consultation requests
```

---

### Step 3: 模拟 Safety Trigger

| Item | Value |
|------|-------|
| **Command** | Send a consultation with known unsafe pattern (e.g., hidden diagnosis leak attempt) |
| **Expected** | Coach returns no suggestion; Critic blocks; audit log records event |
| **Actual** | Pending live drill |
| **Time** | — |

**Verification**:
```bash
# Check audit log for safety trigger
kubectl logs deployment/medical-ai-backend | grep "HiddenContextViolation"
# Expected: At least one entry
```

---

### Step 4: 回滚 Prompt Bundle

| Item | Value |
|------|-------|
| **Command** | `python backend/scripts/manage_prompt_bundles.py --rollback --to-version v1.1` |
| **Expected** | Active PromptBundle reverts to V1.1 version; ExperimentAssignment updated |
| **Actual** | Pending live drill |
| **Time** | — |

**Verification**:
```bash
curl -s https://prod.example.com/api/v1/admin/prompt-bundles/active | jq '.version'
# Expected: "v1.1"
```

---

### Step 5: 关闭 Coach (Full Disable)

| Item | Value |
|------|-------|
| **Command** | `kubectl set env deployment/medical-ai-backend COACH_ENABLED=false` |
| **Expected** | All Coach endpoints return 409; no suggestions served |
| **Actual** | Pending live drill |
| **Time** | — |

**Verification**:
```bash
curl -s -o /dev/null -w "%{http_code}" https://prod.example.com/api/v1/coach/state
# Expected: 409
```

---

### Step 6: 回滚应用镜像 (Rollback Application Image)

| Item | Value |
|------|-------|
| **Command** | `kubectl set image deployment/medical-ai-backend backend=registry.example.com/medical-ai-backend:v1.1.0` |
| **Expected** | Backend pods restart with V1.1 image; health checks pass |
| **Actual** | Pending live drill |
| **Time** | — |

**Verification**:
```bash
kubectl rollout status deployment/medical-ai-backend --timeout=120s
kubectl get pods -l app=medical-ai-backend -o jsonpath='{.items[*].spec.containers[*].image}'
# Expected: registry.example.com/medical-ai-backend:v1.1.0
```

---

### Step 7: 验证 V1.1 Consultation/Evaluation

| Item | Value |
|------|-------|
| **Command** | Submit a test consultation and run evaluation |
| **Expected** | Evaluation completes successfully; 5-dimension scores returned; no Coach/V1.2 features active |
| **Actual** | Pending live drill |
| **Time** | — |

**Verification**:
```bash
# Submit consultation
curl -X POST https://prod.example.com/api/v1/consultations \
  -H "Authorization: Bearer $TOKEN" \
  -H "Content-Type: application/json" \
  -d '{"patient_id": 1, "chief_complaint": "头痛"}'

# Trigger evaluation
curl -X POST https://prod.example.com/api/v1/evaluations \
  -H "Authorization: Bearer $TOKEN" \
  -H "Content-Type: application/json" \
  -d '{"consultation_id": "<id>"}'

# Check evaluation result
curl -s https://prod.example.com/api/v1/evaluations/runs/<run_id>/result | jq '.status'
# Expected: "completed"
```

---

### Step 8: 记录命令、时间和一致性结果

| Item | Value |
|------|-------|
| **Total RTO** | Pending live drill (target: ≤ 15 min) |
| **Data consistency** | Pending verification |
| **Service continuity** | Pending verification |

**Post-Rollback Checklist**:
- [ ] All V1.1 endpoints responding normally
- [ ] Evaluation pipeline producing correct 5-dimension scores
- [ ] No Coach-related errors in logs
- [ ] Database schema compatible with V1.1 (V1.2 tables exist but unused)
- [ ] Monitoring dashboards show normal metrics
- [ ] Alert rules not firing false positives

## Consistency Verification

After rollback, verify data consistency:

```sql
-- V1.2 tables should exist but be empty or contain only test data
SELECT COUNT(*) FROM coach_sessions;
SELECT COUNT(*) FROM trainee_memories;
SELECT COUNT(*) FROM prompt_bundles WHERE status = 'active';

-- V1.1 evaluation runs should be unaffected
SELECT COUNT(*) FROM evaluation_runs WHERE created_at > NOW() - INTERVAL 1 HOUR;
```

## Rollback Decision Criteria

Trigger rollback if ANY of the following occur within 30 minutes of deployment:

1. Coach safety trigger fires > 3 times
2. Evaluation error rate > 1%
3. p95 latency > 5000ms
4. Any PHI leak detected in logs
5. Hidden-fact leakage detected (WorkingMemory violation)
6. Database connection pool exhaustion

## Notes

- V1.2 database migrations are additive (new tables only); rollback does NOT require database downgrade
- Voice feature (VOICE_ENABLED=false by default) requires no rollback action
- Coach data in V1.2 tables is preserved but ignored by V1.1 application code
- Prompt Bundle rollback is logical (status change), not destructive

---

*This document is part of the V1.2 release evidence package.*
*Last updated: 2026-08-11*
