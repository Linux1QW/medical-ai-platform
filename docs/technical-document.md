# 技术架构摘要

本文面向后端、RAG 和运维开发者，是当前实现的技术导读。完整且权威的配置、API、数据模型、部署、评测门禁和限制见 [PROJECT_GUIDE](PROJECT_GUIDE.md)。事实基线为 `codex/rag-bm25-optimization` 分支的当前代码。

> 本项目仅用于医学教学、训练、研究和质量改进，不替代真实诊断、处方、急救决策或人工审核。代码、日志、知识库、缓存、trace、导出和评测报告不得包含未经授权的可识别患者信息。

## 1. 运行架构

```text
React/Nginx
    │ REST + SSE + WebSocket
FastAPI ── SQLAlchemy async ── MySQL
    ├─ LangGraph ── Redis Checkpoint
    ├─ Celery client ── Redis broker/result ── Celery Worker/Beat
    └─ RAG ── Chroma Dense + BM25 + optional BGE-M3 Sparse
                         └─ weighted RRF → tiered → rerank
```

关键入口：

- `backend/app/main.py`：生命周期、中间件、异常、健康检查和 metrics。
- `backend/app/api/v1/`：`/api/v1` 下的 REST/WS 路由。
- `backend/app/services/evaluation_service.py`：LangGraph/legacy 评估入口与结果持久化。
- `backend/app/orchestration/graph.py`：StateGraph、Safety、Plan-Execute、两波 fan-out/fan-in、评分和复核。
- `backend/app/celery_app.py`：Celery 序列化、超时、Beat 和 Worker generation listener。
- `backend/app/tasks/evaluation_task.py`：异步评估、网络类重试、心跳和 checkpoint resume。
- `backend/app/tasks/rag_index_task.py`：immutable generation 构建、锁、CAS 和发布。

## 2. LangGraph 评估链

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

Redis checkpoint thread 为 `evaluation:<run_id>`。Celery 评估任务最多重试 2 次，只重试网络/连接/超时类异常，退避 30/60 秒；重试时可复用失败 run 的 checkpoint。全局 Celery hard/soft limit 是 600/300 秒，评估运行预算默认 240 秒。

## 3. RAG 检索

### 3.1 在线检索

- `lexical/tokenizer.py`：`medical-lexical-v3`，保护医学缩写、变异、剂量/单位和代码，再分词；可选 CJK bigram 默认关闭。
- `bm25_search.py` + `lexical/artifacts.py`：bm25s 0.3.9、`k1=1.2`、`b=0.8`、heading/entity 有界 boost、generation-scoped mmap artifact。
- `medical_store.py`：Chroma collection `medical_guidelines_<generation>`，Dense embedding 固定为 `qwen3.7-text-embedding` 1024 维。
- `sparse_search.py`：可选 BGE-M3 learned sparse，默认关闭且依赖未默认安装。
- `retriever/fusion.py`：BM25/Dense/可选 Sparse 并行召回，稳定 `doc_id` 去重，generation 一致性校验，加权 RRF 默认权重 `0.30/0.45/0.25`、`RRF_K=35`。
- `retriever/tiered.py`：Base → MQE → HyDE；最多 2 个 MQE 扩展、1 次 HyDE、20 个候选。
- `reranker.py`：专用 reranker 后接 LLM 精排；失败退回前序排序。Metadata filter、多样性、上下文扩展/压缩和 OCR 默认关闭。

检索缓存 key 含 generation 和检索设置；缓存不保存正文，命中后从指定 generation 的 Chroma 回填并丢弃 stale 文档。

### 3.2 Immutable generation 发布

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

当前没有 RAG generation 回滚 REST/CLI。生产回滚必须验证旧 generation 全套组件，以当前 pointer 为 expected 做 CAS，并发布同格式事件；完整 runbook 见 [总手册“知识库操作与回滚”](PROJECT_GUIDE.md#14-知识库操作与回滚)。

## 4. Celery 拓扑

已注册任务：

- `run_evaluation`；
- `rebuild_rag_index`、`add_rag_index`、`replace_rag_index`、`delete_rag_index`；
- `cleanup_expired_records`。

Beat 每 86400 秒投递一次清理任务；Beat 必须单实例。基础 Compose Worker 使用 `--concurrency=2`，但固定 `container_name` 不适合直接水平扩容。

当前 Compose 有两项关键差异，部署前必须覆盖：FastAPI `backend` 未注入容器内 Celery broker/result URL；代码 RAG `PDF_DIR` 在容器为 `/app/data`，Compose 却挂到 `/app/backend/data/medical_pdfs`。详见 [总手册“Compose 服务与端口”](PROJECT_GUIDE.md#72-compose-服务与端口)。

## 5. 数据、认证和可观测性

数据库权威路径是空库执行 Alembic `upgrade head`。`database/init.sql` 缺少当前模型的一部分表/列，且不能与 Alembic baseline 直接串联。数据表和迁移差异见 [总手册“数据库与迁移”](PROJECT_GUIDE.md#8-数据库与迁移)。

REST 使用 Bearer JWT；access 默认 1440 分钟、refresh 默认 7 天。医生资源通过 `require_consultation_access` 限制为本人，管理员可跨用户。JWT 黑名单使用 Redis；Redis 不可用时当前实现 fail open。

`/health` 检查 MySQL/Redis、缓存、Token 和 checkpointer；`/metrics` 在生产需要 `METRICS_TOKEN`。RAG 重点 trace：`index_generation`、BM25 load/query、channel candidates、cache hit、retrieval level、generation mismatch 和 stale cache。

## 6. 评测与当前真实性状态

Task 8 合约要求 overall Recall@10/nDCG@10 不低于真实 baseline、exact-term Recall@10 至少提高 0.05、cold load ≤10 秒、search p95 ≤5 ms、generation mismatch/stale cache 均为 0；缺测即失败。

当前不得宣称候选已通过：本地未跟踪的 `backend/evaluation_reports/bm25-v1.json` 缺少新 gate schema 的 overall/exact-term/consistency，且 `evaluate_bm25.py` CLI 不能注入真实一致性计数。CI mock gate 只验证评测管道。完整命令、网格和解释见 [总手册“评测、调参与质量门禁”](PROJECT_GUIDE.md#15-评测调参与质量门禁)。

## 7. 已知接口/部署风险

- `POST /api/v1/evaluations/` 声明 `EvaluationOut`，生产分支却返回 task submission dict，可能触发响应校验失败。
- 模型版本 list/active 当前未鉴权；review status 和 task status 未做对象归属校验。
- `VITE_API_BASE_URL` 没有被 Axios 使用。
- ChromaDB 1.5.7 使用极大 HNSW sync threshold 规避跨进程段加载缺陷，冷读可能从 WAL 重建。

实际 API 目录、配置默认值、测试与生产清单统一查阅 [PROJECT_GUIDE](PROJECT_GUIDE.md)。
