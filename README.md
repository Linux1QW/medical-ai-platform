# 基于多智能体的医生临床问诊评估平台

一个面向医生临床问诊训练与考核场景的 AI 评估平台：医生与「虚拟患者」进行模拟问诊，平台通过 **LangGraph 编排的多智能体系统** 对问诊全过程进行 **五维度自动评估**（问诊分析 / 医学知识核对 / 人文关怀 / 诊断评估 / 治疗方案），并结合 **混合 RAG 检索 200+ 本医学指南与教材** 为每条评分提供可追溯的循证依据。

---

## 目录

- [项目简介](#项目简介)
- [系统架构](#系统架构)
- [核心功能模块](#核心功能模块)
- [技术栈](#技术栈)
- [多智能体系统](#多智能体系统)
- [RAG 检索系统](#rag-检索系统)
- [后端架构](#后端架构)
- [前端架构](#前端架构)
- [数据库设计](#数据库设计)
- [快速开始](#快速开始)
- [Docker 部署](#docker-部署)
- [测试与代码质量](#测试与代码质量)
- [可观测性与监控](#可观测性与监控)
- [安全设计](#安全设计)
- [迭代进度](#迭代进度)
- [项目目录结构](#项目目录结构)
- [版本与贡献](#版本与贡献)

---

## 项目简介

### 项目定位

本平台是一个面向医学教育与临床能力评估的 AI 驱动系统，核心目标是：

- **客观评估**：通过多智能体协作对医生问诊全过程进行五维度自动评分，消除主观偏差
- **循证可追溯**：每条评分结论附带指南引用（来源文献 / 页码 / 原文片段），杜绝幻觉引用
- **安全兜底**：高危红旗 fail-closed 机制，证据不足时自动转人工复核
- **可复现**：统一报告协议 + 版本化基准集 + 确定性种子，保证评估结果可复现

### 核心价值

| 维度 | 价值 |
|------|------|
| 对医生 | 获得结构化、循证的问诊能力反馈，明确改进方向 |
| 对教学机构 | 标准化评估工具，支持大规模临床能力考核 |
| 对平台运营 | 全链路可观测、成本可控、数据安全合规 |

---

## 系统架构

### 整体架构

```
┌─────────────────────────────────────────────────────────────────────┐
│                        React 前端 (Vite + Ant Design)               │
│  ┌──────────┐  ┌──────────┐  ┌──────────┐  ┌──────────┐           │
│  │ 问诊页面  │  │ 评估页面  │  │ 复核工作台│  │ 统计看板  │           │
│  └────┬─────┘  └────┬─────┘  └────┬─────┘  └────┬─────┘           │
│       └──────────────┴──────────────┴──────────────┘                │
│                        WebSocket / REST API                          │
└────────────────────────────────┬────────────────────────────────────┘
                                 │
┌────────────────────────────────▼────────────────────────────────────┐
│                     FastAPI 后端 (Python 3.10+)                     │
│  ┌──────────┐  ┌──────────┐  ┌──────────┐  ┌──────────┐           │
│  │ API 路由  │  │ 评估引擎  │  │ RAG 检索  │  │ 安全治理  │           │
│  │ (11模块)  │  │ (多智能体)│  │ (混合检索)│  │ (脱敏/审计)│          │
│  └────┬─────┘  └────┬─────┘  └────┬─────┘  └────┬─────┘           │
│       │              │              │              │                 │
│  ┌────▼──────────────▼──────────────▼──────────────▼────┐          │
│  │              LangGraph 编排引擎 (Wave DAG)            │          │
│  │  Wave 1: inquiry / knowledge / humanistic (并行)      │          │
│  │  Wave 2: diagnosis / treatment (并行, 消费上游证据)    │          │
│  │  Wave 3: scoring / suggestion (汇总)                  │          │
│  └──────────────────────────────────────────────────────┘          │
└────────────────────────────────┬────────────────────────────────────┘
                                 │
┌────────────────────────────────▼────────────────────────────────────┐
│                         数据存储层                                   │
│  ┌──────────┐  ┌──────────┐  ┌──────────┐  ┌──────────┐           │
│  │ MySQL 8  │  │ Redis 7  │  │ChromaDB  │  │ 文件存储  │           │
│  │(业务数据) │  │(缓存/队列)│  │(向量索引) │  │(PDF/日志) │           │
│  └──────────┘  └──────────┘  └──────────┘  └──────────┘           │
└─────────────────────────────────────────────────────────────────────┘
```

### 架构分层

| 层次 | 职责 | 关键组件 |
|------|------|---------|
| **表现层** | 用户交互、实时进度展示 | React 19 + TypeScript + Ant Design 6 + WebSocket |
| **API 层** | RESTful 接口、认证鉴权、限流 | FastAPI + JWT 双令牌 + RBAC |
| **业务层** | 评估编排、智能体调度、复核流程 | LangGraph + Celery + 多 Agent 协作 |
| **检索层** | 知识获取、证据追溯 | BM25 + Dense + Learned Sparse + Reranker |
| **数据层** | 持久化、缓存、消息队列 | MySQL + Redis + ChromaDB |
| **治理层** | 安全、审计、成本控制 | 数据分级 + PII 脱敏 + Token 预算 + Trace 追踪 |

---

## 核心功能模块

### 评估流水线

| 模块 | 说明 |
|------|------|
| **多智能体评估流水线** | 6 个 Agent（inquiry / knowledge / humanistic / diagnosis / treatment / summary）按 Wave DAG 三波次并行执行 |
| **统一报告协议 (ReportManifest)** | Pydantic 模型统一报告元数据（报告类型 / 版本 / 种子 / 时间戳），保证新旧报告可区分 |
| **回归门禁与退出码协议** | 4 种退出码（PASS=0 / FAIL=1 / SKIP=2 / INVALID=3），smoke/regression/benchmark 分级门禁 |
| **五维原子 Rubric 评估体系** | 每维度独立评分（pass / partial / fail / unassessed / not_applicable），unassessed ≠ 0 分 |
| **Judge 稳定性校准** | AB 对比验证评分一致性，Bootstrap 置信区间，防止评分漂移 |
| **安全红旗回归集** | 高危症状 fail-closed 机制：LLM 失败 + 无规则匹配 → 自动转人工复核 |
| **人工复核状态机** | pending → in_review → approved/rejected/returned 合法迁移，非法迁移拒绝 |
| **稳定 Citation ID** | 基于 (kb_version + doc_id + chunk_id + content_hash) 的 SHA-256 确定性 ID，版本隔离 |
| **Claim-Evidence Graph** | 治疗/诊断 claim 必须附带证据链接，unsupported claim 自动标记需复核 |
| **PlanStep DAG 通用校验** | 依赖图环检测 + ready 步骤计算，适用于任意 Agent 编排 |
| **并发/Token/成本预算控制** | RunBudget 限制并发 Agent 数 / Token 总量 / 成本上限，安全路径豁免 |
| **全链路 Trace 与可观测性** | TraceContext 贯穿 Celery 重试，trace_id 不变 + attempt 递增，PII 自动脱敏 |
| **可版本化临床能力基准集** | BenchmarkManifest 管理 dev/test/regression/safety/benchmark 分组，固定 seed 可重放 |
| **数据分级与生命周期管控** | P0-P3 四级分类，按级别设定保留期限，过期 trace 自动清理，导出权限控制 |

### 前端功能

| 模块 | 说明 |
|------|------|
| **证据化评估报告** | RubricItemList（评分卡片）+ EvidenceTrace（证据追踪）+ RiskBanner（风险横幅） |
| **人工复核工作台** | 复核队列排序/筛选/详情弹窗，决策表单含 reason_code + feedback 必填 |
| **WebSocket 实时进度** | 评估过程节点级进度推送（0-100%），断线自动恢复 |
| **统计看板** | 个人/团队多维度评估数据趋势分析（Recharts） |

---

## 技术栈

| 层次 | 技术选型 | 说明 |
|------|---------|------|
| **后端** | Python 3.10 + FastAPI 0.115 | 异步非阻塞，Pydantic V2 数据校验 |
| **前端** | React 19 + TypeScript + Vite 7 + Ant Design 6 | 严格类型检查，manualChunks 代码分割 |
| **智能体编排** | LangGraph | Wave DAG 三波次并行，Checkpoint 持久化到 Redis |
| **LLM** | 阿里云百炼 Qwen API | 默认 `qwen3.7-max`，支持模型路由与跨 Provider 熔断 |
| **向量存储** | ChromaDB | BGE 系列嵌入，索引蓝绿发布 |
| **关键词检索** | bm25s | 自实现医学分词 + jieba 760+ 术语词典 |
| **重排** | DashScope gte-rerank + Cross-Encoder | 两阶段重排（粗排 20→10，精排 10→5） |
| **数据库** | MySQL 8.0 | Alembic 管理迁移，四层防重复提交 |
| **缓存/队列** | Redis 7 | Checkpoint / 检索缓存 / Celery broker |
| **异步任务** | Celery + Celery beat | 评估任务异步执行 + 定时数据清理 |
| **可观测性** | Prometheus + Grafana + Langfuse | 结构化日志 + 指标暴露 + 链路追踪 |

---

## 多智能体系统

### 五维评估智能体

1. **问诊分析智能体（inquiry）** — 评估病史采集的全面性（主诉 / 现病史 / 既往史 / 过敏史等覆盖度）与问诊技巧
2. **医学知识核对智能体（knowledge）** — 检索医学指南，核对诊断与治疗方案和循证医学证据的一致性
3. **人文关怀智能体（humanistic）** — 评估医患沟通质量、共情表达与人文关怀体现
4. **诊断评估智能体（diagnosis）** — 结合指南证据评估诊断准确性与鉴别诊断逻辑
5. **治疗方案智能体（treatment）** — 评估治疗方案的合理性、安全性（用药禁忌 / 剂量 / 随访计划）

### Wave DAG 三波次编排

| 波次 | 节点 | 说明 |
|------|------|------|
| Wave 1 | inquiry、humanistic、knowledge | 三个基础评估 Agent 并行执行 |
| Wave 2 | diagnosis、treatment | 消费 Wave 1 的知识核对证据后并行执行 |
| Wave 3 | scoring、suggestion | 汇总五维分数，生成总分与改进建议 |

### 知识核对的三种推理模式

| 模式 | 入口 | 机制 | 配置开关 |
|------|------|------|---------|
| RAG 管线 | `pipeline.run_knowledge_check` | 确定性流程：事实提取 → 查询构建 → 分级检索 → 两阶段重排 → LLM 一致性判断 | 默认兜底 |
| Tool Use | `tool_use.run_knowledge_check_with_tools` | LLM 通过 Function Calling 自主调用检索工具，带调用预算与引用校验 | `ENABLE_TOOL_USE` |
| ReAct | `react.run_knowledge_check_react` | 显式 Thought → Action → Observation 推理链，全程可见 | `ENABLE_REACT_KNOWLEDGE` |

### 工具系统

- 统一 ToolRegistry 注册 + ToolExecutor 执行 + ToolBudget 预算控制
- `verify_citation` 引用校验工具：核对引用 ID 是否真实存在，非法引用触发修正或强制失败

---

## RAG 检索系统

### 三路融合检索

| 检索路 | 权重 | 实现 |
|--------|------|------|
| BM25 关键词 | 0.30 | `bm25s` 引擎，jieba 医学分词 + 760+ 术语词典 |
| Dense 语义向量 | 0.45 | BGE 系列嵌入模型 + ChromaDB |
| Learned Sparse（可选） | 0.25 | BGE-M3 稀疏表示，`BGE_M3_ENABLED` 控制 |

融合采用 **加权 RRF（Weighted Reciprocal Rank Fusion，k=60）**。

### 分级检索（L1 → L2 → L3 级联）

- **L1 Base**：三路融合基础召回
- **L2 MQE**：多查询扩展，带语义漂移防护
- **L3 HyDE**：假设性文档嵌入，处理复杂语义查询

### CRAG 置信度闸门

- **HIGH** → 直接使用
- **MEDIUM** → 触发 MQE / HyDE 增强检索
- **LOW** → 拒答并转人工复核

### 两阶段重排

1. DashScope `gte-rerank` 粗排（20 → 10）
2. LLM Cross-Encoder 精排（10 → 5）

### 其他检索能力

- 医学实体归一化（313 个 ICD/ATC 实体别名映射）
- 标题层级感知分块
- 检索结果 Redis 缓存（TTL 24h）
- 索引蓝绿发布 + A/B 对比
- OCR 兜底（Qwen-VL）
- Small-to-Big / 上下文压缩

---

## 后端架构

### API 路由（`/api/v1`）

| 路由 | 职责 |
|------|------|
| `/auth` | 登录 / 注册 / JWT 双令牌刷新 / 登出（黑名单） |
| `/patients` | 虚拟患者管理 |
| `/consultations` | 问诊会话与消息交互（含 WebSocket） |
| `/evaluations` | 评估发起 / 查询 / 进度推送 |
| `/stats` | 个人与团队统计 |
| `/knowledge-base` | 知识库文档与索引管理 |
| `/cases` | 病例难度自适应推荐 |
| `/review` | 人工复核工作流 |
| `/admin` | 用户 / 权限 / 系统管理 |
| `/model-versions` | 模型版本注册表 |
| `/data-export` | 评估数据批量导出 |

### 评估可靠性设计

- **四层防重复提交**：前端按钮防抖 → API 幂等校验 → 数据库行级锁 → UNIQUE 约束兜底
- **评估状态机**：`pending → running → completed / needs_review / failed`，失败支持重试链
- **LLM 高可用**：模型路由 + 跨 Provider 熔断 + 客户端懒初始化 + 并发信号量限流 + 语义缓存
- **统一 JSON 解析**：三层策略（标准 JSON → 代码块提取 → 截断修复）
- **WebSocket 鉴权**：首条消息传 token，避免 token 暴露在 URL / 日志中

### 异步任务

- Celery worker 承接耗时评估任务
- Celery beat 执行定时任务（数据留存清理、Token 统计汇总等）
- Broker / Backend 使用独立 Redis DB

---

## 前端架构

### 页面结构

- **问诊页**：医生与虚拟患者实时对话界面
- **评估页**：五维雷达图、分维度评分卡、循证引用展示、改进建议
- **复核工作台**：复核队列排序/筛选/详情弹窗，决策表单
- **管理页**：病例 / 用户 / 知识库管理（RBAC 控制可见性）
- **统计页**：个人与团队评估数据趋势分析（Recharts）

### 关键机制

- **WebSocket 进度推送**：实时显示评估进度、当前节点与预计剩余时间
- **评估锁轮询恢复**：页面刷新 / 网络中断后轮询评估锁状态，自动恢复进度展示
- **构建优化**：Vite manualChunks 代码分割 + 路由懒加载
- **测试**：Vitest + Testing Library 覆盖关键页面组件

---

## 数据库设计

### 核心表

| 表 | 说明 |
|----|------|
| `users` | 用户与 JSON 细粒度权限 |
| `virtual_patients` | 虚拟患者配置（人格 / 病情 / 难度） |
| `consultations` / `consultation_messages` | 问诊会话与消息历史 |
| `evaluations` | 五维评分结果与循证引用 |
| `evaluation_runs` | 评估运行记录（状态机 / 耗时 / Token 用量） |
| `evaluation_locks` | 评估防重锁 |
| `evaluation_checkpoints` | LangGraph 检查点 |
| `review_records` | 人工复核记录 |
| `model_versions` | 模型版本注册表 |
| `audit_logs` | 审计日志 |

### 迁移管理（Alembic）

```bash
cd backend
# 应用全部迁移到最新
venv\Scripts\python.exe -m alembic upgrade head
# 修改 ORM 模型后生成新迁移
venv\Scripts\python.exe -m alembic revision --autogenerate -m "描述"
```

---

## 快速开始

### 环境要求

- Python 3.10+、Node.js 18+、MySQL 8.0、Redis 7.0+
- 阿里云百炼平台 API Key（[DashScope](https://dashscope.aliyuncs.com/)）

### 1. 后端

```bash
cd backend
python -m venv venv
venv\Scripts\activate              # Windows（Linux/macOS: source venv/bin/activate）
pip install -r requirements.txt

copy .env.example .env             # 按注释修改配置
```

`.env` 必配项（完整清单见 [.env.example](backend/.env.example)）：

```env
# 数据库
MYSQL_USER=medical_app
MYSQL_PASSWORD=<你的密码>
MYSQL_DATABASE=medical_ai

# JWT
SECRET_KEY=<足够长的随机字符串>

# Qwen API
QWEN_API_BASE_URL=https://dashscope.aliyuncs.com/compatible-mode/v1
QWEN_MODEL=qwen3.7-max

# Redis
REDIS_CHECKPOINT_URL=redis://localhost:6379/1
```

### 2. 初始化数据库

```bash
# 建库（MySQL 中执行 database/init.sql + seed.sql），然后应用迁移：
venv\Scripts\python.exe -m alembic upgrade head
# 创建管理员账户
venv\Scripts\python.exe init_admin.py
```

### 3. 构建知识库索引（可选）

将医学指南 PDF 放入项目根 `data/` 目录后执行索引构建脚本（详见 `backend/scripts/`）。

### 4. 启动服务

```bash
# 后端 API（端口 8000）
venv\Scripts\python.exe -m uvicorn app.main:app --host 0.0.0.0 --port 8000

# Celery worker（另开终端）
venv\Scripts\python.exe -m celery -A app.celery_app worker -l info -P solo   # Windows 需 -P solo

# Celery beat 定时任务（另开终端，可选）
venv\Scripts\python.exe -m celery -A app.celery_app beat -l info
```

```bash
# 前端（端口 5173）
cd frontend
npm install
npm run dev
```

访问 `http://localhost:5173`，API 文档见 `http://localhost:8000/docs`。

---

## Docker 部署

`docker-compose.yml` 定义完整服务栈：**mysql、redis、backend、celery-worker、celery-beat、frontend**，以及 monitoring profile 下的 **prometheus、grafana**。

```bash
# 基础服务栈
docker compose up -d

# 含监控（需先在 .env 设置 GRAFANA_ADMIN_PASSWORD）
docker compose --profile monitoring up -d
```

- `docker-compose.override.yml`：本地开发覆盖（热重载 / 端口映射）
- `docker-compose.staging.yml` / `docker-compose.prod.yml`：多环境部署差异配置
- 网络隔离：`backend-net` 与 `frontend-net` 分离；数据卷持久化 MySQL / Redis / ChromaDB / Prometheus / Grafana

---

## 测试与代码质量

### 后端

```bash
cd backend
venv\Scripts\python.exe -m ruff check .            # lint：零违规
venv\Scripts\python.exe -m pytest tests/ -q        # 1093 passed / 18 skipped
```

- 覆盖单元 / 集成 / 端到端测试，CI 强制覆盖率门槛 40%
- 重构遵循「先补测试锁定行为，再动代码」
- E2E 验收测试覆盖 12 个场景（`tests/e2e/`）

### 前端

```bash
cd frontend
npm run lint     # ESLint 零违规
npm test         # Vitest + Testing Library（64 passed）
npm run build    # tsc -b 严格类型检查 + Vite 构建
```

### 工程规范

- `pre-commit` 钩子提交前自动检查（首次 `pre-commit install`）
- Conventional Commits 提交信息规范
- 详见 [CONTRIBUTING.md](CONTRIBUTING.md)

---

## 可观测性与监控

- **结构化 JSON 日志**：统一格式，便于集中收集（`LOG_FORMAT=json`）
- **Prometheus 指标**：`/metrics` 端点暴露 HTTP 请求量 / 延迟分布 / 检索命中率等
- **Grafana 面板**：`monitoring/` 预置数据源与总览 dashboard
- **Langfuse 链路追踪**（可选）：端到端可视化 Agent 推理链路与 Token 消耗
- **全链路 Trace 传播**：TraceContext 贯穿 Celery 重试，trace_id 不变 + attempt 递增
- **告警**：钉钉 / 企微 Webhook 告警，LLM 错误率超阈值自动通知
- **成本管控**：TokenTracker 按模型 / 用户 / 维度统计用量，每日预算超限告警

---

## 安全设计

- **认证**：JWT Access + Refresh 双令牌；登出令牌进 Redis 黑名单
- **鉴权**：细粒度 RBAC（用户级 JSON 权限配置）
- **WebSocket**：首条消息传 token 鉴权，token 不出现在 URL 与访问日志
- **安全响应头**：全局中间件注入 `X-Content-Type-Options` / `X-Frame-Options` / `Referrer-Policy` 等
- **数据脱敏**：姓名 / 手机号 / 身份证号自动掩码（Trace 日志自动脱敏）
- **数据分级**：P0（原始）/ P1（脱敏）/ P2（Rubric）/ P3（聚合），按级别设定保留期限
- **审计日志**：关键操作全量记录，支持留存策略自动清理
- **导出控制**：普通用户不可导出 P0 原始数据，管理员需审批
- **密钥管理**：`.env` 不入库；生产环境 `SECRET_KEY` 强制显式配置
- **传输加密**：HTTPS/TLS

---

## 迭代进度

### 9 周迭代计划（Task 0 ~ Task 16）

| Task | 名称 | 状态 | 说明 |
|------|------|------|------|
| Task 0 | 基线冻结 | ✅ 完成 | 冻结评估基线，制定迭代计划 |
| Task 1 | 统一报告协议 | ✅ 完成 | `ReportManifest` Pydantic 模型统一报告元数据 |
| Task 2 | 回归门禁 | ✅ 完成 | 退出码协议（PASS/FAIL/SKIP/INVALID） |
| Task 3 | 五维原子 Rubric | ✅ 完成 | 每维度独立评分，unassessed ≠ 0 分 |
| Task 4 | Judge 稳定性校准 | ✅ 完成 | AB 对比验证评分一致性 |
| Task 5 | 安全红旗回归集 | ✅ 完成 | 高危症状 fail-closed 机制 |
| Task 6 | 人工复核状态机 | ✅ 完成 | 合法迁移验证 + 非法迁移拒绝 |
| Task 7 | 稳定 Citation ID | ✅ 完成 | SHA-256 确定性 ID，版本隔离 |
| Task 8 | Claim-Evidence Graph | ✅ 完成 | 治疗/诊断 claim 证据验证 |
| Task 9 | PlanStep DAG 校验 | ✅ 完成 | 依赖图环检测 + ready 步骤计算 |
| Task 10 | 并发/Token/成本预算 | ✅ 完成 | RunBudget + 安全路径豁免 |
| Task 11 | 全链路 Trace | ✅ 完成 | TraceContext 传播 + PII 脱敏 |
| Task 12 | 评估报告前端升级 | ✅ 完成 | RubricItemList + EvidenceTrace + RiskBanner |
| Task 13 | 人工复核工作台 | ✅ 完成 | 复核队列排序/筛选/决策表单 |
| Task 14 | 可版本化基准集 | ✅ 完成 | BenchmarkManifest + 分组 + 确定性重放 |
| Task 15 | 数据治理与部署安全 | ✅ 完成 | P0-P3 分级 + 保留策略 + 导出控制 |
| Task 16 | 端到端发布验收 | ✅ 完成 | 25 个 E2E 场景验收测试 |

### 测试统计

| 范围 | 数量 |
|------|------|
| 后端全量测试 | 1093 passed, 18 skipped |
| 前端测试 | 64 passed |
| E2E 验收测试 | 25 passed |

---

## 项目目录结构

```
medical-ai-platform/
├── backend/
│   ├── app/
│   │   ├── api/v1/               # 11 个 API 路由模块
│   │   ├── core/                 # Settings 配置中心 / 安全 / 中间件
│   │   ├── models/               # SQLAlchemy ORM 模型
│   │   ├── orchestration/        # LangGraph Wave DAG 编排与节点适配器
│   │   ├── evaluation/           # 评估治理模块
│   │   │   ├── benchmark.py      # 可版本化临床能力基准集
│   │   │   └── data_governance.py # 数据分级与生命周期管控
│   │   ├── services/
│   │   │   ├── agents/           # 评估智能体
│   │   │   │   └── knowledge/    # 知识核对包（pipeline/tool_use/react）
│   │   │   ├── rag/              # 混合检索 / 重排 / 分块 / 索引
│   │   │   ├── tools/            # 工具注册 / 执行 / 预算
│   │   │   ├── prompts/          # Prompt 版本管理
│   │   │   ├── run_budget.py     # 并发/Token/成本预算控制
│   │   │   └── observability/    # 全链路 Trace 与可观测性
│   │   └── main.py
│   ├── evaluation/               # 评估核心模块
│   │   ├── report_schema.py      # 统一报告协议 (ReportManifest)
│   │   ├── gate.py               # 回归门禁与退出码协议
│   │   ├── rubric.py             # 五维原子 Rubric 评估
│   │   ├── safety_cases.py       # 安全红旗回归集
│   │   ├── review_audit.py       # 人工复核状态机
│   │   ├── citation_registry.py  # 稳定 Citation ID
│   │   ├── rag_claims.py         # Claim-Evidence 验证
│   │   ├── plan_dag.py           # PlanStep DAG 通用校验
│   │   └── ...
│   ├── alembic/                  # 数据库迁移
│   ├── scripts/                  # 索引构建等运维脚本
│   ├── tests/                    # pytest 测试（1093 用例）
│   │   ├── evaluation/           # 评估模块测试
│   │   ├── services/             # 服务层测试
│   │   └── e2e/                  # 端到端验收测试
│   ├── .env.example              # 环境变量全量模板
│   └── requirements.txt
├── frontend/
│   ├── src/
│   │   ├── pages/                # 页面（问诊/评估/复核/统计/管理）
│   │   ├── components/           # 通用组件（RubricItemList/EvidenceTrace/RiskBanner）
│   │   ├── types/                # TypeScript 类型定义
│   │   ├── api/ services/ store/ # 请求封装与状态管理
│   │   └── test/                 # Vitest 测试
│   └── package.json
├── database/                     # init.sql / seed.sql / archive
├── monitoring/                   # prometheus.yml + Grafana provisioning
├── data/                         # 医学指南 / 教材 PDF（200+，不入库）
├── dataset/                      # 虚拟患者病例数据集（150+，不入库）
├── docker-compose*.yml           # 容器编排（base / override / staging / prod）
├── CHANGELOG.md
├── CONTRIBUTING.md
└── README.md
```

---

## 版本与贡献

- 版本遵循语义化版本（当前 `v1.0.0`），变更记录见 [CHANGELOG.md](CHANGELOG.md)
- 团队协作规范（分支 / 提交 / 质量门槛 / PR 流程）见 [CONTRIBUTING.md](CONTRIBUTING.md)

## 版权声明

Copyright © 2026。本项目为私有软件，保留所有权利（All Rights Reserved）。未经作者书面授权，任何个人或组织不得复制、修改、分发或商用本项目的任何部分。
