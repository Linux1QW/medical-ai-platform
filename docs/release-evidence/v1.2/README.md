# V1.2 Release Evidence

This directory documents the V1.2 release gating workflows, artifact naming conventions, retention policies, and final acceptance evidence.

## Final Acceptance Bundle

The **single source of truth** for V1.2 acceptance is:

| File | Description |
|------|-------------|
| `final-acceptance.json` | Machine-readable acceptance bundle (JSON) — **the only source of truth** |
| `final-acceptance.md` | Human-readable report generated from JSON |
| `acceptance-summary.md` | Executive summary generated from JSON |
| `rollback-drill.md` | Step-by-step rollback procedure with RTO target |

> **Important**: All Markdown files in this directory are auto-generated from `final-acceptance.json`. Do not edit them manually. Re-run `build_v12_acceptance_bundle.py` to regenerate.

### Build Command

```bash
python backend/scripts/ci/build_v12_acceptance_bundle.py
```

`--skip-tests` 仅生成不可发布的草稿，输出必须为 `passed=false`。迁移、E2E、
负载和审批字段只能由受保护的发布工作流通过环境证据写入；本地默认均为未验证。

### Key Findings

- **Passed**: 以 `final-acceptance.json` 的实测结果为准；缺少任一证据即为 false
- **Voice**: NOT_ACCEPTED — no real LiveKit evidence, default OFF
- **Coach**: CODE_COMPLETE — default OFF, 8 safety controls verified
- **Backend / Frontend tests**: 由生成命令实时采集，不使用固定数字
- **mypy / ruff**: 由生成命令实时执行，不使用历史结果

## Workflows

### CI (`.github/workflows/ci.yml`)

The CI workflow runs on every push to `main`/`master` and all pull requests. V1.2 adds the following blocking jobs:

| Job Name | Description |
|----------|-------------|
| `v12-route-auth-inventory` | Verifies V1.2 route and auth coverage |
| `v12-72case-benchmark` | 72-case structural benchmark (mock mode) |
| `v12-safety-corpus` | Deterministic safety probe (zero-leak verification) |
| `v12-mcp-protocol` | MCP protocol conformance tests |
| `v12-voice-integration` | Mocked Voice integration tests |

All V1.2 jobs must pass before `docker-build-candidate` proceeds.

### V1.2 Release Candidate Gate (`.github/workflows/v12-rc.yml`)

Triggered manually via `workflow_dispatch` with a required `candidate_sha` parameter (full 40-char commit SHA).

| Job Name | Description |
|----------|-------------|
| `coach-72-case-live` | Real-model 72-case benchmark (live LLM) |
| `v12-auth-e2e` | Authenticated end-to-end API tests |
| `v12-load-test` | HTTP load test (p95 <= 2500ms, error rate < 1%) |
| `v12-fault-recovery` | Fault injection and recovery verification |
| `v12-rc-manifest` | Generates signed JSON manifest of all results |

**Never substitutes mock mode** — all jobs use real models and real infrastructure.

### Deploy (`.github/workflows/deploy.yml`)

The deploy workflow includes a `v12-evidence-gate` job that **blocks deployment** of any V1.2 tag unless:
1. The CI workflow has succeeded for the tag's commit SHA
2. The V1.2 Release Candidate Gate workflow has succeeded for the tag's commit SHA

## Artifact Naming Convention

| Workflow | Artifact Name | Content |
|----------|---------------|---------|
| CI | `v12-72case-benchmark-report` | `v12_coach_gate_report.json` |
| CI | `v12-safety-probe-report` | `v12_safety_probe_report.json` |
| RC | `v12-rc-72case-live-report` | `v12_coach_gate_report.json` (live mode) |
| RC | `v12-rc-auth-e2e-report` | API test results |
| RC | `v12-rc-load-test-report` | Load test metrics |
| RC | `v12-rc-fault-recovery-report` | Fault matrix results |
| RC | `v12-rc-manifest` | Signed manifest JSON + SHA256 |

## Retention Rules

| Artifact Type | Retention |
|---------------|-----------|
| CI benchmark/safety reports | 14 days |
| RC workflow reports | 90 days |
| RC signed manifest | 365 days |
| Docker candidate images | 1 day |
| Compose smoke logs | 7 days |

## Download Commands

```bash
# Download CI benchmark report for a specific run
gh run download <run-id> -n v12-72case-benchmark-report --dir ./evidence

# Download RC signed manifest
gh run download <run-id> -n v12-rc-manifest --dir ./evidence

# Download all RC artifacts
gh run download <run-id> --dir ./evidence
```

## Action Pin Policy

All GitHub Actions references must be pinned to immutable 40-character SHAs. The `verify_action_pins.py` script enforces this — any mutable tag like `@v4` fails CI.
