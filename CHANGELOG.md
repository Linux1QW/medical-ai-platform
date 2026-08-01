# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/zh-CN/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/lang/zh-CN/spec/v2.0.0.html).

## [Unreleased]

### Added
- Prompt 外置化 + 版本管理（`PromptManager`）：13 个 agent system prompt 抽离为文件，支持 `PROMPT_ACTIVE_VERSIONS` 灰度覆盖、变量渲染、缓存与热重载
- LLM Provider 适配器抽象层（`ProviderAdapter` + `OpenAICompatibleAdapter` + 注册表），通用 `LLM_*` 配置（空回退 `QWEN_*`）
- 单测：`test_prompt_manager.py`（18）、`test_llm_adapter.py`（17）
- 文档：`docs/prompt-and-provider-adapter.md`

### Added - 临床评估可靠性迭代（Task 0 ~ Task 16）

- **统一报告协议 (ReportManifest)**：Pydantic 模型统一报告元数据，区分 smoke/regression/benchmark 报告类型
- **回归门禁与退出码协议**：PASS=0 / FAIL=1 / SKIP=2 / INVALID=3 四级退出码
- **五维原子 Rubric 评估体系**：每维度独立评分（pass/partial/fail/unassessed/not_applicable），unassessed ≠ 0 分
- **Judge 稳定性校准**：AB 对比验证评分一致性，Bootstrap 置信区间
- **安全红旗回归集**：高危症状 fail-closed 机制，LLM 失败 + 无规则匹配 → 自动转人工复核
- **人工复核状态机**：pending → in_review → approved/rejected/returned 合法迁移验证
- **稳定 Citation ID**：基于 (kb_version + doc_id + chunk_id + content_hash) 的 SHA-256 确定性 ID
- **Claim-Evidence Graph**：治疗/诊断 claim 必须附带证据链接，unsupported claim 自动标记需复核
- **PlanStep DAG 通用校验**：依赖图环检测 + ready 步骤计算
- **并发/Token/成本预算控制**：RunBudget 限制并发 Agent 数 / Token 总量 / 成本上限，安全路径豁免
- **全链路 Trace 与可观测性**：TraceContext 贯穿 Celery 重试，PII 自动脱敏
- **前端证据化报告组件**：RubricItemList + EvidenceTrace + RiskBanner
- **人工复核工作台**：复核队列排序/筛选/详情弹窗/决策表单
- **可版本化临床能力基准集**：BenchmarkManifest 管理 dev/test/regression/safety/benchmark 分组
- **数据分级与生命周期管控**：P0-P3 四级分类，按级别设定保留期限，过期 trace 自动清理
- **端到端发布验收测试**：25 个 E2E 场景覆盖全部迭代任务

### Changed
- 后端测试从 579 用例增至 1093 用例（+100%）
- 前端测试覆盖 64 用例（9 个测试文件）
- 新增 `tests/e2e/` 端到端验收测试目录

### Fixed
- 修复 failover「半接线」缺陷：熔断切换 Provider 时真实重建底层 LLM 客户端（此前仅更新索引/计数，请求仍打向原端点）

## [1.0.0] - 2026-07-21

### Added
- 评估防重复提交机制 + ORM 模型补全 + Pydantic V2 迁移
- 前端 chunk 优化与评估结果自动展示
- 安全 Agent 确定性红旗规则门控
- RAG 模块 V2 核心架构（混合检索 + Reranker + HyDE）
- 多智能体临床问诊评估平台完整架构
- ReAct 推理链稳定化 + Suggestion Agent 集成
- LLM 缓存层 + 安全加固（限流/审计/密码策略）
- LangGraph 编排重构 + Redis Checkpoint + Function Call
- Tool Use 加固 + 评估指标体系完善
- Docker Compose 部署编排配置
- 前端组件库统一（Ant Design 5.x）
- 代码质量工具链：ruff + pre-commit + pyproject.toml 统一配置

### Changed
- 评估页面交互优化：进入后自动生成评估报告
- 五维评估架构升级：Plan-Execute / Send fan-out / ReAct / Reflection / SSE
- 数据集管理：移除真实医疗数据，保护隐私
- CI/CD 流程完善：测试覆盖率上报 + 健康检查

### Fixed
- 前端 TypeScript 构建错误修复
- test_llm_cache CI mock 三层失效修复
- flake8 F821 前向引用、auth db=None 防御、限流器测试隔离
- RAG 评估测试 CLI 子进程 cwd 路径错误
- 多项 CI 测试稳定性修复
