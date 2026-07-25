# 贡献指南（团队内部协作规范）

> 本项目为私有软件（All Rights Reserved），本文档面向获得授权的团队成员。

## 开发环境准备

```bash
# 后端
cd backend
python -m venv venv
venv\Scripts\activate          # Windows
pip install -r requirements.txt
copy .env.example .env         # 按需修改数据库/API Key 配置

# 前端
cd frontend
npm ci
```

## 分支与提交规范

- 主分支为 `master`，功能开发使用 `feat/<描述>`、修复使用 `fix/<描述>` 分支
- 提交信息遵循 [Conventional Commits](https://www.conventionalcommits.org/zh-hans/)：
  `feat:` / `fix:` / `docs:` / `refactor:` / `test:` / `chore:` 等
- **严禁提交任何密钥、密码、真实患者数据**（`.env`、`dataset/` 已被 .gitignore 排除，请勿绕过）

## 代码质量要求

| 检查项 | 要求 |
|---|---|
| ruff | `ruff check backend/` 零违规（含 B、C90 规则；新函数圈复杂度 ≤ 15，禁止新增 noqa） |
| 测试 | `pytest tests/` 全过；新功能必须附带测试；覆盖率不得低于 CI 门槛（40%） |
| 前端 | `npm run lint` 零违规；`npm test`（vitest）全过；`tsc -b` 无类型错误 |
| 类型 | 公共函数补类型标注，mypy 警告尽量清零 |

提交前本地自查：

```bash
cd backend
venv\Scripts\python.exe -m ruff check .
venv\Scripts\python.exe -m pytest tests/ -q
```

已配置 pre-commit 钩子，首次使用执行 `pre-commit install`。

## 架构约定

- 分层：`api（路由）→ services（业务）→ rag / llm / agents（能力层）`，路由层禁止直写 SQL
- 配置一律进 `app/core/config.py` 的 `Settings`，禁止散落硬编码；新增配置项须同步 `.env.example`
- 错误响应统一 `{"error_code": ..., "message": ...}` 结构，经全局异常处理器输出
- 数据库结构变更需提供迁移 SQL（`database/` 目录）并在 PR 描述中说明

## PR 流程

1. 从 `master` 拉出功能分支，完成开发与自测
2. 推送后创建 PR，CI 必须全绿（lint / 测试 / 安全扫描 / 构建）
3. 至少一名成员 review 通过后合并；合并采用 squash，保持主干历史整洁
