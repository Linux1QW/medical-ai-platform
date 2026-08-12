# 项目文档中心

本页是项目文档入口。根目录 [README](../README.md) 用于首次了解和快速启动；[PROJECT_GUIDE](PROJECT_GUIDE.md) 是源码级总手册；专题文档负责解释某个子系统或操作流程。若说明与代码冲突，应以配置、路由、迁移、Compose、测试和当次验收证据为事实来源，并在同一次变更中修正文档。

## 按阅读目标选择

| 目标 | 建议阅读顺序 |
|---|---|
| 第一次了解项目 | [README](../README.md) → [平台操作说明](platform-documentation.md) → [技术架构说明](technical-document.md) |
| 后端/API 开发 | [PROJECT_GUIDE](PROJECT_GUIDE.md) → `backend/app/api/v1` → `backend/app/services` → `backend/tests` |
| Agent/Coach 开发 | [V1.2 Production Guide](release-evidence/v1.2/production-v1.2.md) → `backend/app/agent_runtime` → `backend/app/orchestration` |
| RAG 开发 | [PROJECT_GUIDE RAG 章节](PROJECT_GUIDE.md) → `backend/app/services/rag` → `backend/tests/rag` |
| 前端开发 | [README 页面与接口](../README.md#7-主要页面与接口) → `frontend/src/App.tsx` → `frontend/src/api` → 前端测试 |
| 部署与故障处理 | [V1.2 Runbook](runbooks/production-v1.2.md) → Compose → [验收证据](release-evidence/v1.2/) |
| 发布验收 | [验收摘要](release-evidence/v1.2/acceptance-summary.md) → `final-acceptance.json` → `.github/workflows/v12-rc.yml` |
| 参与贡献 | [CONTRIBUTING](../CONTRIBUTING.md) → [CHANGELOG](../CHANGELOG.md) |

## 文档分层与优先级

1. **机器可验证事实**：源码、配置、迁移、Compose、锁文件、自动化测试和受保护工作流产物。
2. **发布验收事实**：`release-evidence/v1.2/final-acceptance.json`。只有 `passed=true` 且候选 SHA、实测指标和审批有效时，才能宣称生产验收通过。
3. **总览说明**：根 README 与 PROJECT_GUIDE。
4. **专题说明**：technical、platform、runbook、ADR、评测和 Prompt 文档。
5. **历史计划**：`superpowers/plans` 描述计划或实施过程，不自动代表代码已完成，也不能替代验收证据。

## 主要文档

| 文档 | 面向对象 | 内容 |
|---|---|---|
| [README](../README.md) | 所有人 | 项目边界、能力、架构、目录、接口、启动、配置、测试和限制 |
| [PROJECT_GUIDE](PROJECT_GUIDE.md) | 开发/测试/运维 | 详细配置、数据、API、Agent、RAG、Celery、测试与发布清单 |
| [技术架构说明](technical-document.md) | 架构/开发 | 状态机、可靠投递、Redis 拓扑、RAG generation 和保留策略 |
| [平台操作说明](platform-documentation.md) | 医生/管理员/运维 | 端到端业务操作、安全边界和常见故障 |
| [评测基线](evaluation-baseline.md) | 算法/QA | 评分语义、评测维度、安全红旗与反馈闭环 |
| [Prompt 与 Provider](prompt-and-provider-adapter.md) | Agent/平台开发 | Prompt 管理、Provider 抽象、降级和缓存 |
| [V1.2 Production Guide](release-evidence/v1.2/production-v1.2.md) | 发布/运维 | V1.2 生产行为、接口与安全控制 |
| [V1.2 Runbook](runbooks/production-v1.2.md) | 运维/值班 | 部署、探针、故障处置和回滚演练 |
| [ADR](adr/) | 架构/评审 | 关键设计决策及其上下文和取舍 |

## 当前发布状态

- V1.2 功能代码位于当前修复分支，但尚未合并到 `master`。
- 本地自动化质量检查通过不等于生产验收通过。
- 当前 `release-evidence/v1.2/final-acceptance.json` 明确记录 `passed=false`；真实 RC 运行、迁移演练、部署 E2E、负载指标与人工审批未齐全前，所有文档都必须保持这一口径。
- Voice 保持 `NOT_ACCEPTED` 且默认关闭；Coach 为 `CODE_COMPLETE` 但默认关闭。
- 后端版本元数据已统一为 `1.2.0`；正式发布仍需在最终候选 SHA 上重新生成验收证据。

## 文档维护检查清单

代码修改涉及以下任一项时，应同步更新对应文档：

- API 路径、请求/响应 schema、错误码或认证授权；
- 环境变量、默认值、Feature Flag 或安全启动约束；
- 数据模型、Alembic 迁移、数据保留或导出行为；
- Compose 服务、Redis 拓扑、Celery 队列、Dispatcher/Beat 单例要求；
- LangGraph 节点、Agent、Prompt、Tool/Skill/MCP 策略；
- RAG tokenizer、召回、融合、rerank、generation 或发布门禁；
- 前端路由、页面入口、Coach/Voice 状态和管理员操作；
- CI、RC、E2E、负载、回滚和验收证据格式。

文档不得包含真实患者数据、密钥、生产 URL 中的凭据或无法追溯来源的性能数字。
