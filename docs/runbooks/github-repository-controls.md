# GitHub Repository Controls Runbook

本文档记录仓库的 ruleset、Environment 和 Dependabot 配置，作为 RC 证据。

## Owner

- 主要维护者: 仓库 owner
- 季度复核: 每季度首月检查 ruleset 和 Environment 配置

---

## Master Ruleset

### 配置位置

`Settings → Rules → Rulesets → New branch ruleset`

### 规则配置

| 配置项 | 值 |
|--------|-----|
| Name | `master-protection` |
| Target | `master` |
| Enforcement | Active |

### Required Pull Request

- [x] Require a pull request before merging
- [x] Require approvals: 0 (单人仓库) / 1 (多人后启用)
- [x] Dismiss stale pull request approvals upon new commits
- [x] Require conversation resolved

### Required Status Checks

以下 checks 必须通过:

1. `backend-test`
2. `frontend-test`
3. `migration-gate`
4. `code-quality`
5. `security-scan`
6. `rag-pr-gate`
7. `compose-smoke`
8. `image-security-scan`

### Branch Protection

- [x] Restrict deletions
- [x] Require branches to be up to date
- [x] Block force pushes

### Bypass

- Break-glass 团队可 bypass，需审计日志

---

## Tag Protection

### 配置位置

`Settings → Tags → New rule`

### 规则

| Pattern | 限制 |
|---------|------|
| `v*` | 禁止更新和删除 |

---

## Environments

### staging

- 单人批准即可部署
- 无额外限制

### production

- 至少一名非提交者批准（多人后启用）
- 限制部署 workflow: `ci.yml`
- 限制部署 branch: `master`, `v*` tags

> 注: 当前单人仓库无法实现非提交者审批。正式 production 发布须由第二名维护者加入后配置。

---

## Dependabot

配置文件: `.github/dependabot.yml`

| Ecosystem | 频率 | PR 限制 |
|-----------|------|---------|
| pip | 每周一 | 5 |
| npm | 每周一 | 5 |
| github-actions | 每周一 | 3 |
| docker | 每周一 | 3 |

---

## Break-Glass 流程

1. 记录 bypass 原因和审批人
2. 执行紧急变更
3. 24 小时内补审并记录到 issue
4. 季度复核时检查 bypass 日志

---

## 验证证据

- [ ] 尝试直接 push master → 被拒绝
- [ ] 尝试 force-push → 被拒绝
- [ ] 尝试删除受保护 tag → 被拒绝
- [ ] Required check 失败时 PR 不可合并

验证命令 (GitHub CLI):

```bash
# 检查 ruleset
gh api repos/{owner}/{repo}/rulesets

# 检查 tag protection
gh api repos/{owner}/{repo}/tags/protection

# 检查 environments
gh api repos/{owner}/{repo}/environments
```

---

## 季度复核清单

- [ ] Ruleset 配置未变或被正确更新
- [ ] Tag protection 规则有效
- [ ] Environment 审批人列表正确
- [ ] Dependabot 配置覆盖所有生态
- [ ] Bypass 日志已审查
