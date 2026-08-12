# 技术架构摘要

本文面向后端、RAG 和运维开发者，是当前实现的技术导读（v1.1.0）。完整且权威的配置、API、数据模型、部署、评测门禁和限制见 [PROJECT_GUIDE](PROJECT_GUIDE.md)。事实基线为 `codex/v1.1-a-runtime` 分支的当前代码。

> 本项目仅用于医学教学、训练、研究和质量改进，不替代真实诊断、处方、急救决策或人工审核。代码、日志、知识库、缓存、trace、导出和评测报告不得包含未经授权的可识别患者信息。

## 1. 运行架构

```text
React/Nginx
    │ REST + SSE + WebSocket
FastAPI ── SQLAlchemy async ── MySQL
    ├─ LangGraph ── redis-state Checkpoint (db=1)
    ├─ Outbox enqueue ── MySQL evaluation_dispatch_outbox
    ├─ Celery client ── redis-state broker(db=4)/result(db=5)
    └─ RAG ── Chroma Dense + BM25 + optional BGE-M3 Sparse
                         └─ weighted RRF → tiered → rerank

Evaluation Dispatcher（独立进程）
    └─ Outbox 轮询 → claim lease → Celery publish → acknowledge

Celery Worker/Beat ── redis-state broker/result
redis-cache ── LLM cache (db=0) + retrieval cache (db=1)
```

关键入口：

- `backend/app/main.py`：生命周期、中间件、异常、健康检查和 metrics。
- `backend/app/api/v1/`：`/api/v1` 下的 REST/WS 路由。
- `backend/app/services/evaluation_service.py`：LangGraph/legacy 评估入口与结果持久化。
- `backend/app/orchestration/graph.py`：StateGraph、Safety、Plan-Execute、两波 fan-out/fan-in、评分和复核。
- `backend/app/services/evaluation_run_service.py`：EvaluationRun 状态机（创建/claim/续期/重试/终态/取消）。
- `backend/app/services/evaluation_dispatch_service.py`：Outbox enqueue/claim/acknowledge/reject/cancel/purge。
- `backend/app/tasks/evaluation_dispatcher.py`：Dispatcher 派发循环（独立进程）。
- `backend/app/tasks/evaluation_reconciliation.py`：定时对账（过期 lease、stale dispatch、retrying 超时）。
- `backend/app/celery_app.py`：Celery 序列化、超时、Beat 和 Worker generation listener。
- `backend/app/tasks/evaluation_task.py`：异步评估、网络类重试、心跳和 checkpoint resume。
- `backend/app/tasks/rag_index_task.py`：immutable generation 构建、锁、CAS 和发布。

## 2. EvaluationRun 状态机

`evaluation_run_service.py` 集中管理 run 的完整生命周期：

```text
         create_queued_run
              │
              ▼
          ┌────────┐    claim_run     ┌─────────┐
          │ queued │ ───────────────→ │ running │ ← heartbeat + lease 续期
          └────────┘                  └─────────┘
              ▲                           │    │
              │   mark_run_retrying       │    │  mark_run_terminal
              │  ◄────────────────────────┘    │
          ┌──────────┐                         │
          │ retrying │                         │
          └──────────┘                         ▼
                                    ┌──────────────────────┐
                                    │ completed            │
                                    │ needs_review         │
                                    │ failed               │
                                    │ cancelled            │
                                    │ reviewed             │
                                    └──────────────────────┘
```

### 状态转换规则

| 转换 | 函数 | 条件 |
|---|---|---|
| → queued | `create_queued_run` | 创建时，attempt=0 |
| queued → running | `claim_run` | Dispatcher claim，attempt+1，设置 execution_owner/lease |
| retrying → running | `claim_run` | 重试 claim，attempt+1 |
| running → retrying | `mark_run_retrying` | 网络类异常，清除执行租约 |
| running → completed/needs_review | `mark_run_terminal` | 正常完成 |
| running → failed | `mark_run_terminal` | 不可恢复异常 |
| running → cancelled | `mark_run_terminal` | cancel_requested_at 已设置 |

- **终态集合**：`{completed, needs_review, reviewed, failed, cancelled}`
- **非法转换**：抛 `InvalidRunTransition`
- **Owner 不匹配**：抛 `RunLeaseLost`
- **Heartbeat**：运行中每 N 秒续期 `lease_expires_at`，防止长评估期间锁过期

### 幂等与重试

- Celery 评估任务最多重试 2 次，只重试网络/连接/超时类异常，退避 30/60 秒
- 重试时从 checkpoint 恢复（`EVAL_RESUME_FROM_CHECKPOINT=true`），避免从头重跑
- Outbox 保证至少一次投递，Celery 任务应实现幂等（通过 run_id 去重）

### 协作式取消

- `request_run_cancel` 设置 `cancel_requested_at` + Redis 取消标志
- 运行中任务由 `EVAL_CANCEL_POLL_SECONDS` 轮询看守协作中断
- 排队任务 best-effort Celery revoke

## 3. Transactional Outbox 与 Dispatcher

### 3.1 Outbox 事务边界

评估任务投递使用 Outbox 模式，保证**业务写入与任务投递原子一致**：

```text
BEGIN TRANSACTION
  INSERT INTO evaluation_runs (status='queued', ...)
  INSERT INTO evaluation_dispatch_outbox (status='pending', run_id=..., ...)
COMMIT
```

Outbox 与 run 在同一 MySQL 事务中写入，要么都成功，要么都失败。Dispatcher 独立进程异步轮询派发。

### 3.2 Outbox 状态机

```text
pending ──claim──→ leased ──publish──→ published
   │                  │
   │  reject          │  reject (max attempts/24h)
   ▼                  ▼
pending (retry)    dead_letter
                    │
                    │  purge (>7d)
                    ▼
                 (deleted)

任何状态 ──cancel──→ cancelled
```

### 3.3 Dispatcher Loop

`evaluation_dispatcher.py` 实现独立派发循环：

```text
while running:
    leases = claim_dispatch_batch(batch_size=20, lease_seconds=30)
    for lease in leases:
        try:
            task_id = celery.send_task(lease.task_name, lease.payload)
            acknowledge(lease.event_id, task_id)  # → published
        except Exception:
            reject(lease.event_id, error_code)     # → pending or dead_letter
    sleep(DISPATCH_POLL_INTERVAL_SECONDS)
```

- **Worker ID**：`hostname:pid:uuid`，唯一标识派发者
- **Lease 租约**：30 秒默认，过期可被其他 Worker 回收
- **Circuit Breaker**：连续 5 次失败后熔断 30 秒
- **日志脱敏**：payload 只输出 key 名，隐藏值

### 3.4 Reconciliation

`evaluation_reconciliation.py` 定时对账：

| 场景 | 处理 |
|---|---|
| queued + leased + lease 过期 | 释放 outbox → pending |
| queued + published > 5min + 从未 claim + < 24h | requeue_stale_dispatch |
| queued + published > 24h + 从未 claim | outbox dead_letter + run failed(dispatch_unclaimed) |
| running + lease 过期 + 有 cancel_requested_at | → cancelled |
| running + lease 过期 + 有剩余 attempt | → retrying |
| running + lease 过期 + 耗尽 attempt | → failed |
| retrying + cancel | → cancelled |
| retrying + 2min 未 claim | published outbox → pending |
| retrying + 20min 未 claim | → failed(redelivery_lost) |

## 4. 双 Redis 物理拓扑

| 实例 | 策略 | 承载 | 本地端口 |
|---|---|---|---|
| `redis-state` | Redis Stack 7；AOF + noeviction | checkpoint(db=1)、broker(db=4)、result(db=5)、progress(db=6)、JWT blacklist(db=7)、evaluation control(db=8) | 6379 |
| `redis-cache` | allkeys-LRU | LLM cache(db=0)、retrieval cache(db=1) | 6380 |

**设计权衡**：

- **分离原因**：状态数据（checkpoint、broker）不能因缓存淘汰而丢失；缓存数据不应受 AOF 持久化拖累
- **模块要求**：LangGraph Redis checkpointer 在 Redis 8 以下依赖 RedisJSON 和 RediSearch，因此 `redis-state` 使用 Redis Stack；普通 `redis:7` 仅用于 `redis-cache`
- **noeviction**：redis-state 满时拒绝写入而非丢数据，保证 checkpoint/broker 完整性
- **allkeys-LRU**：redis-cache 满时自动淘汰最少使用 key，对缓存场景可接受
- **本地开发**：两个实例都可用 `localhost:6379` 不同 DB，但生产必须物理分离

## 5. LangGraph 评估链

主图按以下顺序运行：

```text
load_context → classify_consultation → safety_check
  → plan_evaluation → validate_plan
  → Wave 1: knowledge/inquiry/humanistic
  → extract_knowledge_citations
  → Wave 2: diagnosis/treatment
  → aggregate_results → deterministic_scoring
  → reflection_check → review_gate
  → suggestion/completed 或 needs_review
```

Safety 对硬性红旗 fail closed；无结论时进入复核。Wave 2 可以消费 Knowledge Agent 引用。默认评分权重为 inquiry 0.25、knowledge 0.25、humanistic 0.20、diagnosis 0.15、treatment 0.15；缺失维度不会临时重分配权重，五项未全部 scored 时总分为 `null`。

Redis checkpoint thread 为 `evaluation:<run_id>`。

## 6. RAG 检索

### 6.1 在线检索

- `lexical/tokenizer.py`：`medical-lexical-v3`，保护医学缩写、变异、剂量/单位和代码，再分词；可选 CJK bigram 默认关闭。
- `bm25_search.py` + `lexical/artifacts.py`：bm25s 0.3.9、`k1=1.2`、`b=0.8`、heading/entity 有界 boost、generation-scoped mmap artifact。
- `medical_store.py`：Chroma collection `medical_guidelines_<generation>`，Dense embedding 固定为 `qwen3.7-text-embedding` 1024 维。
- `sparse_search.py`：可选 BGE-M3 learned sparse，默认关闭且依赖未默认安装。
- `retriever/fusion.py`：BM25/Dense/可选 Sparse 并行召回，稳定 `doc_id` 去重，generation 一致性校验，加权 RRF 默认权重 `0.30/0.45/0.25`、`RRF_K=35`。
- `retriever/tiered.py`：Base → MQE → HyDE；最多 2 个 MQE 扩展、1 次 HyDE、20 个候选。
- `reranker.py`：专用 reranker 后接 LLM 精排；失败退回前序排序。Metadata filter、多样性、上下文扩展/压缩和 OCR 默认关闭。

检索缓存 key 含 generation 和检索设置；缓存不保存正文，命中后从指定 generation 的 Chroma 回填并丢弃 stale 文档。

### 6.2 Immutable generation 发布

生产链只认 `rag_index_task.py`：

```text
snapshot → parse → chunk → embed → chroma → bm25 → sparse
         → validate → switch → publish
```

generation 为 `rag-YYYYMMDDHHMMSS-<sha8>`。候选包含：

- Chroma collection `medical_guidelines_<generation>`；
- `backend/data/rag_indexes/<generation>/manifest.json`；
- `<generation>/bm25/` 的原生索引、manifest、hash inventory 和 `READY`；
- BGE-M3 启用时 `<generation>/sparse/` 的 documents、sparse payload、manifest、hash 和 `READY`。

构建使用 Redis `rag:index-build-lock`（30 分钟、heartbeat 续租）。校验、manifest 摘要和最终锁确认完成后，以旧 generation 为 expected 对 `rag:active_generation` 执行 CAS，再向 `rag:index-switched` 发布 new/previous/manifest SHA-256。CAS 后通知最多重试 3 次；仍失败则任务以 `completed_with_warning` 返回，而不是把已切换指针误报为 FAILURE。每个 Celery fork Worker 先完整加载并验证新组件，再原子替换本地引用；失败时保留旧引用。listener 每 5 秒读取 Redis active pointer 做 reconciliation，因此漏掉瞬时 Pub/Sub 事件也能自动收敛。

以下入口**不是**上述生产发布：

- `python -m app.services.rag.build_medical_index` 调用旧兼容 builder，默认临时构建 `rag-v2` collection，不写 Task 7 顶层 manifest、不 CAS、不发布事件。
- `switch_index_version` 是旧进程内兼容 helper，不是集群 generation 回滚。
- `rebuild_kb_from_cache.py` 使用旧 `backend/data/embed_cache/*.npz`；当前 generation builder 只使用进程内 Embedding LRU。

当前没有 RAG generation 回滚 REST/CLI。生产回滚必须验证旧 generation 全套组件，以当前 pointer 为 expected 做 CAS，并发布同格式事件；完整 runbook 见 [总手册"知识库操作与回滚"](PROJECT_GUIDE.md#14-知识库操作与回滚)。

## 7. Celery 拓扑

已注册任务：

- `run_evaluation`；
- `rebuild_rag_index`、`add_rag_index`、`replace_rag_index`、`delete_rag_index`；
- `cleanup_expired_records`。

Beat 每 86400 秒投递一次清理任务；Beat 必须单实例。基础 Compose Worker 使用 `--concurrency=2`。

### Worker Loop Ownership

每个 Celery fork Worker 在 `worker_process_init` 启动 `rag:index-switched` Pub/Sub listener，负责接收 RAG generation 切换事件并原子替换本地引用。Listener 每 5 秒读取 Redis active pointer 做 reconciliation，保证最终一致。

## 8. 数据、认证和可观测性

数据库权威路径是空库执行 Alembic `upgrade head`。`database/init.sql` 缺少当前模型的一部分表/列，且不能与 Alembic baseline 直接串联。数据表和迁移差异见 [总手册"数据库与迁移"](PROJECT_GUIDE.md#8-数据库与迁移)。

REST 使用 Bearer JWT；access 默认 60 分钟（生产不得超过 60）、refresh 默认 7 天。医生资源通过 `require_consultation_access` 限制为本人，管理员可跨用户。JWT 黑名单使用 Redis db=7；Redis 不可用时当前实现 fail open（开发允许，staging/production 必须设置 `JWT_BLACKLIST_FAIL_CLOSED=true`）。

`/health` 检查 MySQL/Redis、缓存、Token 和 checkpointer；`/metrics` 在生产需要 `METRICS_TOKEN`。RAG 重点 trace：`index_generation`、BM25 load/query、channel candidates、cache hit、retrieval level、generation mismatch 和 stale cache。

## 9. 隐私威胁模型

| 威胁 | 缓解 |
|---|---|
| 患者身份泄露 | 不录入真实患者信息；虚拟患者数据去标识化；日志/trace 脱敏 |
| LLM API 泄露对话 | `OBSERVABILITY_CAPTURE_CONTENT=false`（默认）；staging/production 禁止开启 |
| 外部观测平台泄露 | Langfuse 在 staging/production 必须配置 `OBSERVABILITY_HMAC_KEY`（≥32 字节） |
| 日志文件泄露 | JSON 日志只输出脱敏 detail；`X-Request-ID` 追踪但不含 PII |
| 导出泄露 | 默认导出只含去标识元数据；含完整问诊需显式确认 |
| 备份泄露 | 加密 off-host 备份；密钥独立管理 |
| 未授权访问 | JWT + RBAC + 资源归属校验；管理员账户最小化分配 |

## 10. 数据保留策略

| 数据类型 | 默认保留 | 配置 | 说明 |
|---|---|---|---|
| Dispatch 终态 outbox | 7 天 | `DISPATCH_RETENTION_DAYS=7` | published/cancelled/dead_letter |
| 无报告 failed/cancelled run | 180 天 | `UNREPORTED_RUN_RETENTION_DAYS=180` | 无 Evaluation 关联 |
| LLM cache | 24h | `LLM_CACHE_TTL=86400` | TTL 自动过期 |
| Retrieval cache | 24h | `RETRIEVAL_CACHE_TTL=86400` | TTL 自动过期 |
| Progress bus 事件 | 1h | `PROGRESS_EVENT_TTL_SECONDS=3600` | TTL 自动过期 |
| Checkpoint | 24h | `REDIS_CHECKPOINT_TTL=86400` | TTL 自动过期 |
| 审计日志 | **auto-delete 默认关闭** | `AUDIT_LOG_AUTO_DELETE_ENABLED=false` | 需显式开启 + `DATA_RETENTION_POLICY_ID` |

**重要**：报告关联的 run、Evaluation、ReviewRecord 和 AuditLog **不**由默认通用 cleanup 删除。其保留/删除周期由部署组织的数据治理策略、`DATA_RETENTION_POLICY_ID` 与审批流程决定。

## 11. 评测与当前真实性状态

Task 8 合约要求 overall Recall@10/nDCG@10 不低于真实 baseline、exact-term Recall@10 至少提高 0.05、cold load ≤10 秒、search p95 ≤5 ms、generation mismatch/stale cache 均为 0；缺测即失败。

当前不得宣称候选已通过：本地未跟踪的 `backend/evaluation_reports/bm25-v1.json` 缺少新 gate schema 的 overall/exact-term/consistency，且 `evaluate_bm25.py` CLI 不能注入真实一致性计数。CI mock gate 只验证评测管道。完整命令、网格和解释见 [总手册"评测、调参与质量门禁"](PROJECT_GUIDE.md#15-评测调参与质量门禁)。

## 12. 已知接口/部署风险

- `POST /api/v1/evaluations/` 声明 `EvaluationOut`，生产分支却返回 task submission dict，可能触发响应校验失败。
- 模型版本 list/active 当前未鉴权；review status 和 task status 未做对象归属校验。
- `VITE_API_BASE_URL` 没有被 Axios 使用。
- ChromaDB 1.5.7 使用极大 HNSW sync threshold 规避跨进程段加载缺陷，冷读可能从 WAL 重建。
- 单主机限制：Dispatcher、Beat 均必须单实例，不支持多节点水平扩展。

实际 API 目录、配置默认值、测试与生产清单统一查阅 [PROJECT_GUIDE](PROJECT_GUIDE.md)。
