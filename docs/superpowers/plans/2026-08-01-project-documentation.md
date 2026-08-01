# 项目完整说明文档实施计划

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 建立一份以当前代码和部署配置为准的中文项目手册，使新成员能够理解架构、启动系统、调用 API、执行评估、维护数据和排查故障。

**Architecture:** README 负责项目入口、关键结论和导航；`docs/PROJECT_GUIDE.md` 作为唯一的完整技术手册，覆盖运行流程、配置、接口、数据、测试和运维。文档中的版本、服务名、路径、命令和默认值全部从当前仓库源码与配置文件核对。

**Tech Stack:** Markdown、FastAPI、React/Vite、LangGraph、Celery、MySQL、Redis、ChromaDB、Docker Compose、Alembic。

## Global Constraints

- 文档必须明确区分本地开发、Docker 生产和可选监控 profile。
- 不记录真实密钥、密码或用户隐私数据；示例只使用占位值。
- 不把未在当前代码中实现的功能写成已交付能力。
- 文档命令必须使用仓库真实路径、服务名和脚本名。

---

### Task 1: 建立完整项目手册

**Files:**
- Create: `docs/PROJECT_GUIDE.md`

- [x] 编写项目定位、能力边界、用户角色和端到端业务流程。
- [x] 编写系统架构、目录结构、后端模块、前端页面、数据库表和异步任务说明。
- [x] 编写环境变量表、依赖安装、数据库初始化、Alembic、Docker、Celery 和 RAG 索引操作说明。
- [x] 编写 API 目录、认证授权、监控、测试、故障排查、安全注意事项和当前限制。

### Task 2: 重构 README 为可靠入口

**Files:**
- Modify: `README.md`

- [x] 保留项目简介和最短启动路径。
- [x] 链接完整手册、相关专题文档和关键源码入口。
- [x] 明确安装前置条件、Docker 启动命令、默认访问地址和验证命令。

### Task 3: 文档一致性校对

- [x] 用 `rg` 对服务名、端口、命令、环境变量和 API 前缀做交叉检查。
- [x] 用 Markdown 标题、代码块和链接检查文档可读性。
- [x] 运行后端与前端已有验证命令，记录真实结果，不虚构运行结果。

### Task 4: 提交并推送

- [ ] 查看 diff，确认只包含本次文档与前一轮已确认的 bug 修复。
- [ ] 创建中文 Conventional Commit。
- [ ] 推送当前分支到 `origin`，报告提交号和远程分支。
