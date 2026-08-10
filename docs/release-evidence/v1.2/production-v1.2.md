# V1.2 Release Runbook

## Pre-release Checklist

- [ ] All 72 benchmark cases pass (intent macro-F1 ≥ 0.85)
- [ ] Safety probe: 0 hidden-fact leaks
- [ ] Safety probe: 0 unsafe suggestions
- [ ] Load test: p95 ≤ 2.5s
- [ ] Load test: error rate < 1%
- [ ] mypy: 0 errors
- [ ] ruff: exit 0
- [ ] pytest: 0 failed
- [ ] Frontend: tsc + vitest pass

## Deployment Steps

1. Merge `codex/v1.2-agent-intelligence` to `master`
2. Create tag `v1.2.0`
3. Deploy with `COACH_ENABLED=false` (default off)
4. Run safety probe in production
5. Run load test in production
6. Enable coach for test group (5%)
7. Monitor for 24h
8. Advance to 25%, then 100%

## Rollback Plan

1. Set `COACH_ENABLED=false`
2. Revert to previous Docker image
3. No data migration needed (all V1.2 tables are additive)

## Key Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| COACH_ENABLED | false | Enable coach feature |
| COACH_HMAC_KEY | (required) | HMAC key for telemetry |
| LIVEKIT_API_KEY | (optional) | LiveKit voice beta |
| LIVEKIT_API_SECRET | (optional) | LiveKit voice beta |
