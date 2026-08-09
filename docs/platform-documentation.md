# 平台使用与管理摘要

本文面向医生用户、教师/管理员和一线运维人员。完整接口、配置、架构、部署和限制以 [PROJECT_GUIDE](PROJECT_GUIDE.md) 为准。事实基线为 `codex/v1.1-a-runtime` 分支的当前代码（v1.1.0）。

> **临床使用边界声明**：平台只用于医学教学、训练、研究和质量改进。所有评估报告均为**辅助质量评估，非自主诊断/治疗决策**。平台不替代真实诊断、处方、急救或专业审核。证据不足与安全门触发时必须进入人工复核。真实急危重症应立即转交医疗机构。禁止录入未经授权的真实患者身份、生产病历、密钥；对话、导出、日志和评测报告都必须去标识化并受控保存。

## 医生用户

### 问诊流程

1. 访问 `/register` 注册或 `/login` 登录。
2. 在"虚拟患者"选择病例。医生看到脱敏姓名、主诉和训练信息，不会看到标准诊断和系统 Prompt。
3. 创建问诊并发送问题。普通回复和 SSE 流式回复均受每分钟 10 次限制；默认最多 20 个医生轮次，可每次延长 10 轮。
4. 提交诊断和治疗方案会结束问诊；也可直接结束。

### 评估流程

5. 打开评估页触发评估。非测试环境由 Celery 异步执行；页面通过 WebSocket 和锁状态轮询显示进度。
6. 查看五维结果、总分、分析、建议、引用和复核状态。

> **重要**：维度缺失、安全风险或一致性问题都可能使总分为 `null` 或进入人工复核，这不等于系统故障，而是**证据不足时必须进入人工复核**的安全机制。所有报告均为辅助质量评估，不能作为自主诊断/治疗决策依据。

7. "数据统计"对医生只显示本人数据；可通过 `GET /api/v1/users/me/data-export` 导出本人数据。

### 进度、取消与恢复

- **进度显示**：评估页通过 WebSocket 实时推送进度；Redis 不可用时回退到轮询锁状态。
- **取消评估**：不要刷新正在进行的长任务页面；如需取消，使用评估页取消操作。取消是 Redis 标志加 Celery revoke 的协作式流程，运行中的任务可能需要等待看守轮询（默认 2 秒间隔）。
- **恢复**：Celery 重试时自动从 LangGraph checkpoint 断点续跑（`EVAL_RESUME_FROM_CHECKPOINT=true`），避免从头重跑。

## 管理员

管理员在前端可使用"全部问诊"和"患者管理"，并可通过 API：

- 查看全平台统计、问诊和完整患者资料；
- 创建、更新、删除和导出虚拟患者；
- 查看待复核项并提交复核意见；
- 管理知识库 generation；
- 查看缓存、Tool runtime、run trace、失败原因和 Token 用量；
- 登记、废弃或回滚模型版本状态。

### 管理员复核

管理员通过 API 查看待复核评估并提交复核意见：

```bash
# 查看待复核列表
curl -H "Authorization: Bearer $TOKEN" \
  http://localhost:8000/api/v1/reviews/pending

# 提交复核意见
curl -H "Authorization: Bearer $TOKEN" \
  -H "Content-Type: application/json" \
  -d '{"decision":"approved","feedback":"..."}' \
  http://localhost:8000/api/v1/reviews/{evaluation_id}/submit
```

复核状态机：`pending → in_review → approved/rejected/returned`。

### 报告与引用解释

评估报告包含五维评分、分析、建议和知识库引用。引用包含：

- **稳定 Citation ID**：基于 `(kb_version + doc_id + chunk_id + content_hash)` 的 SHA-256 确定性 ID
- **证据来源**：指向知识库中的具体指南段落
- **Claim-Evidence 关联**：治疗/诊断 claim 必须附带证据链接

> **所有报告页明确**：报告为**辅助质量评估，非自主诊断/治疗决策**。证据不足与安全门触发时必须进入人工复核。

当前 `AdminReviews` 页面没有接入路由，知识库、模型版本和监控也没有完整 UI，请使用 <http://localhost:8000/docs>。模型版本"rollback"只改登记状态，不回滚 RAG generation。

### 创建管理员

在已完成 Alembic 初始化的环境执行：

```powershell
cd backend
.\.venv\Scripts\python.exe scripts/init_admin.py
```

`database/seed.sql` 含 `admin/admin123`，只允许隔离演示并必须立即改密。

### 知识库操作

将 PDF/DOCX 放入仓库根目录 `data/`。文件名必须是相对路径，不能含绝对路径、盘符或 `..`。

```bash
# 全量重建
curl -H "Authorization: Bearer $TOKEN" -X POST \
  http://localhost:8000/api/v1/knowledge-base/rebuild

# 查询任务
curl -H "Authorization: Bearer $TOKEN" \
  "http://localhost:8000/api/v1/knowledge-base/rebuild/status?task_id=<task-id>"

# 添加或替换
curl -H "Authorization: Bearer $TOKEN" -H "Content-Type: application/json" \
  -d '{"filename":"guide.pdf","force_replace":true}' \
  http://localhost:8000/api/v1/knowledge-base/add-pdf
```

状态 phase 依次为 snapshot、parse、chunk、embed、chroma、bm25、sparse、validate、switch、publish。只有完整校验后才切换 Redis active generation。若最终结果为 `completed_with_warning`，说明指针已切换但 Pub/Sub 通知 3 次重试仍失败；Worker 会每 5 秒按 active pointer 自动重同步，不能把它当作"未切换"直接重跑。持续不收敛时检查 Redis、manifest、artifact 和 Worker 日志。

`python -m app.services.rag.build_medical_index` 是旧兼容入口，不会执行 manifest/CAS/Pub/Sub 发布，不要用于生产发布。当前也没有受支持的 RAG 回滚 API；回滚需按 [总手册回滚 runbook](PROJECT_GUIDE.md#142-回滚原则) 由运维实施。

## 启动与值守

推荐先使用本地进程路径：空 MySQL 数据库经 Alembic 初始化，分别启动 FastAPI、Celery Worker、Celery Beat 和 Vite。命令见 [README](../README.md#10-分钟-quickstart)。

基础地址：

- 前端：`http://localhost:5173`（Vite）或 Compose 的 80/443；
- API/OpenAPI：`http://localhost:8000/docs`、`/api/v1/openapi.json`；
- 健康：`/health`；
- Prometheus：`/metrics`，生产必须配置 `METRICS_TOKEN`。

本地 Windows Celery：

```powershell
cd backend
.\.venv\Scripts\python.exe -m celery -A app.celery_app worker --loglevel=info -P solo
.\.venv\Scripts\python.exe -m celery -A app.celery_app beat --loglevel=info
```

Compose 一键启动（V1.1 已修正历史差异）：

```powershell
docker compose up -d
```

V1.1 已修正：Celery broker/result URL 注入、RAG 源文件挂载、Alembic 自动迁移。完整说明见 [PROJECT_GUIDE](PROJECT_GUIDE.md#7-启动与部署)。

## 常见问题

### 登录后仍返回 401

检查 access token 是否过期、是否被加入 Redis 黑名单、系统时间和 `SECRET_KEY` 是否一致。前端收到 401 会清空 sessionStorage 并跳回登录。

### 评估一直 pending 或提交 500

确认 FastAPI 和 Worker 使用相同 Redis db=4/5，Worker 已注册 `run_evaluation`。V1.1 使用 Outbox + Dispatcher，还需确认 `evaluation-dispatcher` 服务正在运行。当前 `POST /evaluations/` 还存在响应模型与异步返回体不一致，可能导致响应校验失败；这是已知代码限制，不应通过文档掩盖。

### WebSocket 立即断开

连接后必须在 5 秒内发送 `{"type":"auth","token":"<JWT>"}`，且用户必须有该问诊访问权。反向代理需要 WebSocket Upgrade。

### 知识库无文件或无结果

确认源文件实际位于代码 `PDF_DIR`；容器中是 `/app/data`。检查 status API 的 active generation、manifest、BM25/Sparse `READY`、Chroma count 和 Worker 切换日志。

### 数据库 unknown table/column

不要依赖应用 `create_all` 修补已有表。检查是否由不完整的 `database/init.sql` 创建，备份后按 Alembic schema 做结构对比和迁移演练，禁止盲目 `stamp head`。

### Task 8 是否已经通过

没有可据此宣称通过。CI mock gate 只验证评测管道；当前本地 baseline schema 和一致性遥测不足以证明真实候选通过。真实要求和命令见 [总手册质量门禁](PROJECT_GUIDE.md#153-task-8-真实-generation-门禁)。

## 上线前最低检查

- 医疗免责声明、人工复核和急救转交流程已落实；数据已授权和去标识化。
- 默认管理员、示例密码和所有密钥已移除或轮换。
- 数据库通过 Alembic/结构审计；Celery、RAG 挂载和共享存储已修正。
- HTTPS、最小 CORS、metrics token、最小权限和备份恢复已验证。
- active/previous generation 均可加载；真实评测和一致性遥测有审计报告。
- `/health`、评估、WebSocket、知识库任务、监控和告警已做端到端演练。
- 所有报告页已明确"辅助质量评估、非自主诊断/治疗决策"。

更完整的生产检查清单和全部已知限制见 [PROJECT_GUIDE](PROJECT_GUIDE.md#19-生产发布检查清单)。
