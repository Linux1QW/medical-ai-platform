# 基于多智能体的医生临床问诊评估平台

## 项目概述

本项目是一个基于多智能体系统的医生临床问诊评估平台，旨在通过人工智能技术对医生的临床问诊过程进行全面、客观的评估。系统集成了问诊分析、医学知识核对、人文关怀评估、诊断评估和治疗方案评估五大维度，为医生提供全方位的问诊质量反馈。

**核心技术栈：**
- **后端**: Python 3.10 + FastAPI 0.115
- **前端**: React 19 + TypeScript + Vite 7 + Ant Design 6
- **编排引擎**: LangGraph
- **向量存储**: ChromaDB
- **数据库**: MySQL 8.0
- **缓存/队列**: Redis
- **AI引擎**: 阿里云百炼平台 Qwen API (qwen3-max)

## 系统架构

### 整体架构
```
[前端界面] <---> [FastAPI后端] <---> [LangGraph编排引擎] <---> [多智能体系统] <---> [RAG检索系统] <---> [知识库]
     |              |                   |                     |                    |                 |
  WebSocket    RESTful API         图状态机            5个评估Agent       向量检索      200+医学指南PDF
```

### 五维评估智能体
1. **问诊分析智能体** - 评估病史采集的全面性和问诊技巧的有效性
2. **医学知识核对智能体** - 检查诊断和治疗方案与临床指南的一致性
3. **人文关怀智能体** - 评估医患沟通质量和人文关怀体现
4. **诊断评估智能体** - 评估诊断准确性和逻辑性
5. **治疗方案智能体** - 评估治疗方案的合理性和安全性

### LangGraph Wave DAG 编排
系统采用创新的 Wave DAG 编排模式，将评估流程分为三个波次：
- **Wave 1**: 并行执行 inquiry、humanistic、knowledge 智能体
- **Wave 2**: 基于 Wave 1 结果，执行 diagnosis、treatment 智能体
- **Wave 3**: 执行 scoring、suggestion 智能体，生成最终评估报告

### 证据链传递
- Knowledge Agent 检索到的指南证据通过 citations 传递给 Diagnosis/Treatment Agent
- 实现端到端的证据可追溯，确保每条评分都有具体的知识库来源支持

## RAG 检索系统

### 三路融合检索管道
- **BM25 关键词检索**（权重 0.30）：基于 `bm25s` 引擎，相较 `rank_bm25` 实现 10x 索引速度和 5x 内存效率提升
- **Dense Vector 语义检索**（权重 0.45）：基于 BGE 系列模型的稠密向量检索
- **Learned Sparse 检索**（权重 0.25，可选）：基于 BGE-M3 的稀疏表示检索
- **加权 RRF 融合**：采用 Weighted Reciprocal Rank Fusion 算法，医学场景调优参数 k=35，实现三路检索结果的最优融合

### BGE-M3 双表示模型（可选）
- BGE-M3 模型同时提供 dense + learned sparse 双表示，一次推理即可生成两种检索信号
- 通过 `BGE_M3_ENABLED` 环境变量控制开关
- **降级策略**：`BGE_M3_ENABLED=False` 时自动降级为 BM25 + Dense 两路融合，确保系统在任何环境下稳定运行

### 分级检索架构
- **L1 Base**: 三路融合检索（BM25 + Dense + Sparse + Weighted RRF），基础召回
- **L2 MQE**: 查询扩展（Multi-Query Expansion），提升召回覆盖率
- **L3 HyDE**: 假设性文档嵌入，处理复杂语义查询

### 两阶段重排
- **第一阶段**: DashScope gte-rerank 专用模型粗排（20→10）
- **第二阶段**: LLM Cross-Encoder 精排（10→5），判断相关性和完整性

### CRAG 置信度闸门
- **HIGH**: 多来源、高分、充分覆盖 → 直接使用
- **MEDIUM**: 部分满足 → 触发 MQE/HyDE 增强
- **LOW**: 严重不足 → 拒答/人工复核

### 医学分词与实体归一化
- 基于 jieba 的医学分词，集成 760+ 医学术语自定义词典
- 313 个 ICD/ATC 实体归一化，实现别名到规范名的映射（如"心梗"→"急性心肌梗死"）

### Retrieval Bundle Cache
- 基于 Redis 的检索结果缓存（TTL=24h）
- 使用索引版本控制，索引重建后自动失效
- 避免相同查询的重复检索和 LLM 调用

### 索引蓝绿发布
- 支持索引的蓝绿部署和平滑切换
- A/B 对比评估，确保索引更新不影响服务质量

## 后端架构

### API 路由清单
- `/auth`: 用户认证
- `/patients`: 虚拟患者管理
- `/consultations`: 问诊交互
- `/evaluations`: 评估管理
- `/stats`: 数据统计
- `/knowledge-base`: 知识库管理
- `/admin`: 管理员功能
- `/cases`: 病例推荐
- `/reviews`: 人工复核

### 评估防重复提交
四层防重机制确保同一问诊不会被重复评估：
1. **前端UI**: 防止用户多次点击
2. **API检查**: 服务端初步校验
3. **行级锁**: 数据库层面锁定
4. **UNIQUE约束**: 最终数据一致性保障

### 评估状态机
```
pending → running → completed/needs_review/failed
                    ↘ retry_pending → retry_running → ...
```

### Checkpoint 人工复核恢复
- 基于 Redis 的 LangGraph Checkpoint 机制
- 支持暂停的评估流程在人工复核后继续执行
- 确保需要人工介入的评估能够正确恢复

### 病例难度自适应推荐
- 基于"最近发展区"理论，推荐略高于当前能力的病例
- 根据医生历史评估表现动态调整推荐难度
- 提供个性化的学习路径

### 统一 JSON 解析
- 三层解析策略：标准JSON → 提取代码块 → 修复不完整JSON
- 确保 LLM 输出的结构化数据能够正确解析

## 前端架构

### 页面结构
- **问诊页**: 医生与虚拟患者交互界面
- **评估页**: 显示多维度评估结果
- **管理页**: 病例管理、用户管理等功能
- **统计页**: 个人和团队评估数据分析

### WebSocket 进度推送
- 实时推送评估进度（0%-100%）
- 显示当前执行的评估节点和预计剩余时间
- 提升用户体验和等待感知

### 评估锁状态轮询恢复
- 前端定期轮询评估锁状态
- 在页面刷新或网络中断后能够恢复评估进度
- 保证用户操作的连续性

### Vite 构建优化
- 代码分割（manualChunks）优化首屏加载
- 按需加载减少初始包大小
- 生产环境压缩和缓存优化

## 数据库设计

### 核心表清单
- `users`: 用户信息
- `virtual_patients`: 虚拟患者配置
- `consultations`: 问诊记录
- `consultation_messages`: 问诊消息历史
- `evaluations`: 评估结果（五维度评分）
- `evaluation_runs`: 评估运行记录
- `evaluation_locks`: 评估锁状态
- `review_records`: 人工复核记录
- `evaluation_checkpoints`: LangGraph 检查点
- `audit_logs`: 审计日志

### 迁移脚本
- `init.sql`: 初始数据库结构
- `migrate_v2-v10.sql`: 增量迁移脚本（含模型版本注册表 + 细粒度权限）
- 支持平滑升级和版本回滚

## 环境配置与部署

### 环境要求
- **Python**: 3.10+
- **Node.js**: 18+
- **MySQL**: 8.0
- **Redis**: 7.0+

### 环境变量说明
```env
# 数据库配置
DATABASE_URL=mysql+pymysql://username:password@localhost:3306/medical_ai

# Redis 配置
REDIS_CHECKPOINT_URL=redis://localhost:6379/1
REDIS_CACHE_URL=redis://localhost:6379/2

# Qwen API 配置
DASHSCOPE_API_KEY=your_api_key
QWEN_MODEL=qwen3-max

# LangGraph 配置
LANGGRAPH_ENABLED=true
```

### 启动命令
```bash
# 后端启动
cd backend
pip install -r requirements.txt
python -m uvicorn app.main:app --host 0.0.0.0 --port 8000

# 前端启动
cd frontend
npm install
npm run dev
```

### 数据库迁移
```bash
# 初始化数据库
python migrate_db.py

# 创建管理员账户
python init_admin.py
```

## 测试

### 后端测试
- **框架**: pytest
- **测试数量**: 403 个测试用例全部通过（25 skipped）
- **覆盖范围**: 单元测试、集成测试、端到端测试

### 前端构建
- **类型检查**: TypeScript 严格模式
- **构建工具**: Vite 7
- **UI组件**: Ant Design 6

## 项目目录结构

```
medical-ai-platform/
├── backend/                  # 后端代码
│   ├── app/
│   │   ├── api/v1/          # API路由
│   │   ├── models/          # ORM模型
│   │   ├── services/
│   │   │   ├── agents/      # 评估智能体
│   │   │   └── rag/         # RAG系统
│   │   ├── orchestration/   # LangGraph编排
│   │   └── core/            # 核心配置
│   ├── data/                # 数据文件
│   ├── tests/               # 测试代码
│   └── requirements.txt
├── frontend/                # 前端代码
│   ├── src/
│   │   ├── pages/           # 页面组件
│   │   ├── components/      # 通用组件
│   │   └── services/        # 前端服务
│   └── package.json
├── data/                    # 医学知识库PDF
├── dataset/                 # 评估数据集
├── database/                # 数据库脚本
└── docs/                    # 文档
```

## 性能优化

### 检索性能
- 三路融合检索（BM25 + Dense + Sparse）+ 加权 RRF 融合，显著提升召回质量
- `bm25s` 引擎替换 `rank_bm25`，索引速度提升 10x，内存效率提升 5x
- 两阶段重排优化排序精度（DashScope gte-rerank 粗排 + LLM Cross-Encoder 精排）
- Redis 缓存减少重复查询开销（TTL=24h，索引版本控制自动失效）
- BGE-M3 可选双表示，支持自动降级保障可用性

### 系统性能
- LangGraph 并行执行优化评估耗时
- 数据库连接池和查询优化
- 前端懒加载和代码分割提升加载速度

## 安全与审计

### 访问控制
- JWT Refresh Token 双令牌认证授权
- 细粒度 RBAC 权限控制（用户级 JSON 权限配置）
- 数据脱敏（姓名/手机号/身份证号自动掩码）
- HTTPS/TLS 传输加密

### 审计日志
- 所有关键操作记录审计日志
- 用户行为追踪和分析
- 数据变更历史追溯

## 企业工程化能力

### 可观测性
- 结构化 JSON 日志，统一日志格式便于集中收集与分析
- Langfuse 链路追踪，端到端可视化 Agent 推理过程
- Prometheus 指标暴露（HTTP 请求量、延迟分布、检索命中率等）
- 告警管理器（AlertManager），支持阈值告警与异常通知

### 安全性
- HTTPS/TLS 传输加密
- JWT Refresh Token 双令牌机制，Access Token 短过期 + Refresh Token 长过期
- 数据脱敏（姓名、手机号、身份证号自动掩码）
- 细粒度 RBAC 权限控制，支持用户级 JSON 权限配置

### 高可用
- Celery 异步任务队列，耗时评估任务异步执行
- 数据库连接池管理，避免连接泄漏
- 数据定期备份策略
- 模型降级策略，BGE-M3 不可用时自动回退两路融合
- 跨 Provider 熔断（LLMFailoverManager），主模型故障自动切换备用 Provider

### CI/CD
- Docker 镜像构建，确保环境一致性
- 多环境部署支持（开发/测试/生产）
- CD 流水线自动化发布

### 代码质量
- `pyproject.toml` 统一项目配置
- `pre-commit` 钩子，提交前自动检查
- `ruff` 高速 Python linter + formatter
- `codecov` 代码覆盖率门禁

### 数据治理
- 数据留存策略，自动清理过期数据
- 模型版本注册表（`model_versions`），追踪模型配置与状态
- 数据导出 API，支持评估数据批量导出

### 成本管控
- Token 用量统计（TokenTracker），按模型/用户/维度精细统计
- 每日预算限制，超出阈值自动告警

## 扩展性设计

### 插件化架构
- Agent 系统支持插件化扩展
- 工具系统支持动态注册
- 评估维度可灵活配置

### 微服务准备
- 模块化设计便于拆分
- API 设计遵循 RESTful 规范
- 配置中心化管理

---

该项目通过先进的 AI 技术和工程实践，为医生临床问诊能力的提升提供了科学、客观、全面的评估工具，有助于提高医疗服务质量和患者满意度。