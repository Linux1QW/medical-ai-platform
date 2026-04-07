# 多智能体协同的临床问诊评估平台

基于多 AI 智能体协同的医学问诊实训与自动评估系统。医生可与虚拟患者进行模拟问诊，系统自动调用五个智能体进行多维度评估并生成改进建议。

---

## 一、技术栈

| 层级 | 技术 | 版本 |
|------|------|------|
| 前端 | React + TypeScript + Vite + Ant Design | React 19 / Vite 7 / Antd 6 |
| 后端 | FastAPI + Python | Python 3.10 / FastAPI 0.115 |
| 数据库 | MySQL | 8.0 |
| AI 引擎 | 阿里云百炼平台 Qwen API | qwen-max |

---

## 二、环境要求

在启动项目之前，请确保本机已安装以下软件：

| 软件 | 最低版本 | 用途 |
|------|----------|------|
| Python | 3.10+ | 后端运行环境 |
| Node.js | 18+ | 前端运行环境 |
| MySQL | 8.0 | 数据存储 |
| npm | 9+ | 前端包管理 |

此外，需要在系统环境变量中配置阿里云百炼平台的 API Key：

- 环境变量名：`DASHSCOPE_API_KEY`
- 值：你的阿里云百炼平台 API Key（sk-xxx 格式）

---

## 三、项目结构

```
medical-ai-platform/
├── backend/                          # 后端服务（FastAPI）
│   ├── app/
│   │   ├── api/v1/                   # REST API 路由
│   │   │   ├── auth.py               #   注册 / 登录 / 获取当前用户
│   │   │   ├── patients.py           #   虚拟患者管理
│   │   │   ├── consultations.py      #   问诊交互（对话 / 结束）
│   │   │   ├── evaluations.py        #   触发评估 / 查看评估报告
│   │   │   └── stats.py              #   管理员数据统计
│   │   ├── core/                     # 核心基础设施
│   │   │   ├── config.py             #   配置（数据库 / JWT / Qwen）
│   │   │   ├── security.py           #   密码加密 + JWT 令牌
│   │   │   └── deps.py               #   认证守卫（依赖注入）
│   │   ├── models/                   # 数据库模型（SQLAlchemy ORM）
│   │   ├── schemas/                  # 请求 / 响应模型（Pydantic）
│   │   ├── services/                 # 业务逻辑
│   │   │   ├── agents/               #   五个 AI 智能体
│   │   │   │   ├── inquiry_agent.py      # 问诊分析智能体
│   │   │   │   ├── knowledge_agent.py    # 医学知识核对智能体
│   │   │   │   ├── humanistic_agent.py   # 人文关怀评估智能体
│   │   │   │   ├── scoring_agent.py      # 综合评分智能体
│   │   │   │   └── suggestion_agent.py   # 建议指导智能体
│   │   │   ├── evaluation_service.py #   评估编排器
│   │   │   ├── consultation_service.py   # 问诊逻辑
│   │   │   ├── qwen_client.py        #   Qwen API 客户端
│   │   │   ├── user_service.py       #   用户管理
│   │   │   └── patient_service.py    #   患者管理
│   │   └── db/session.py             # 数据库连接
│   ├── venv/                         # Python 虚拟环境（已创建）
│   ├── requirements.txt              # Python 依赖
│   ├── .env                          # 环境变量（实际配置）
│   └── .env.example                  # 环境变量示例
├── frontend/                         # 前端应用（React + Vite）
│   ├── src/
│   │   ├── api/                      # 后端接口封装
│   │   ├── layouts/MainLayout.tsx    # 主布局（侧边栏 + 顶栏）
│   │   ├── pages/                    # 页面组件
│   │   │   ├── Login/                #   登录
│   │   │   ├── Register/             #   注册
│   │   │   ├── Dashboard/            #   工作台
│   │   │   ├── PatientList/          #   虚拟患者列表
│   │   │   ├── ConsultationList/     #   问诊记录列表
│   │   │   ├── Consultation/         #   问诊对话
│   │   │   ├── Evaluation/           #   评估报告
│   │   │   └── AdminStats/           #   管理员统计
│   │   ├── store/useAuth.ts          # 登录状态管理
│   │   ├── types/index.ts            # TypeScript 类型
│   │   └── utils/request.ts          # Axios 请求封装
│   ├── node_modules/                 # Node 依赖（已安装）
│   ├── vite.config.ts                # Vite 配置（含 API 代理）
│   └── package.json
├── database/
│   ├── init.sql                      # 建表 SQL
│   └── seed.sql                      # 种子数据（1 管理员 + 4 虚拟患者）
├── .gitignore
└── README.md
```

---

#
---

## 四、部署初始化

首次部署项目时，请务必按照以下步骤完成管理员账号初始化：

1. **后端初始化**：
   ```powershell
   cd backend
   .\venv\Scripts\Activate.ps1
   pip install -r requirements.txt
   ```
   
   **生产部署务必运行python init_admin.py完成初始化**，确保平台可零手动SQL启动。
   ```powershell
   python init_admin.py
   ```
   *注意：脚本会交互式提示输入管理员用户名、邮箱和密码。*

2. **数据库初始化**：
   - 确保 MySQL 服务已启动并创建了名为 `medical_ai` 的数据库。
   - 导入 `database/init.sql` 进行建表。

---

## 五、日常启动步骤

每次开发或使用时，只需执行以下两步：

### 终端 1 — 启动后端

```powershell
cd backend
.\venv\Scripts\Activate.ps1
uvicorn app.main:app --reload --port 8001
```

看到 `Application startup complete.` 表示后端启动成功。

### 终端 2 — 启动前端

```powershell
cd frontend
npm run dev
```

看到 `Local: http://localhost:5173/` 表示前端启动成功。

### 访问

打开浏览器访问 **http://localhost:5173**

---

## 六、预置账号

| 角色 | 用户名 | 密码 | 权限 |
|------|--------|------|------|
| 管理员 | admin | admin123 | 全部功能 + 数据统计 + 患者管理 |

普通医生账号可通过注册页面自行创建。

---

## 七、功能使用流程

### 7.1 医生完整使用流程

```
登录 → 选择虚拟患者 → 开始问诊 → 与患者对话 → 结束问诊 → 生成评估 → 查看报告与建议
```

**第一步：登录系统**

- 访问 http://localhost:5173
- 使用已有账号登录，或点击"立即注册"创建新医生账号

**第二步：选择虚拟患者**

- 点击左侧菜单「虚拟患者」
- 可通过人格类型下拉框筛选（配合型 / 焦虑型 / 沉默型 / 对抗型）
- 查看难度星级（1-5 星），选择合适的患者
- 点击「开始问诊」按钮

**第三步：进行问诊对话**

- 进入对话界面后，左侧显示患者基本信息（姓名、年龄、主诉等）
- 在底部输入框输入问诊内容，按 Enter 发送
- 虚拟患者会根据其人格类型和病例设定自动回复
- 建议按照临床问诊规范依次询问：主诉 → 现病史 → 既往史 → 个人史 → 家族史

**第四步：结束问诊**

- 问诊完毕后，点击右上角「结束问诊」按钮
- 确认后问诊状态变为"已完成"

**第五步：生成评估报告**

- 问诊结束后，点击「查看评估」进入评估页面
- 点击「生成评估报告」按钮
- 系统后台五个 AI 智能体将协同工作（约需 15-30 秒）：
  - **问诊分析智能体**：评估问诊流程的系统性与完整性
  - **医学知识核对智能体**：核对鉴别诊断思路与临床指南
  - **人文关怀评估智能体**：评估沟通态度与共情能力
  - **综合评分智能体**：汇总三项评估生成综合分数
  - **建议指导智能体**：生成具体的改进建议

**第六步：查看评估结果**

- 顶部显示四个维度的仪表盘评分（0-100 分）
- 中部显示各维度的详细分析
- 底部显示综合评价和改进建议

### 7.2 管理员功能

管理员除了拥有医生的全部功能外，还可以：

- **数据统计**：点击左侧菜单「数据统计」，查看全平台的实训次数、评估报告数、各维度平均评分

---

## 八、预置虚拟患者说明

系统内置 4 个虚拟患者，覆盖四种人格类型：

| 姓名 | 年龄/性别 | 人格类型 | 主诉 | 难度 |
|------|-----------|----------|------|------|
| 张明 | 45/男 | 配合型 | 反复胸闷气短 1 个月 | ★★ |
| 李芳 | 32/女 | 焦虑型 | 头痛头晕伴失眠 2 周 | ★★★ |
| 王建国 | 68/男 | 沉默型 | 咳嗽咳痰带血丝 1 周 | ★★★★ |
| 赵红梅 | 55/女 | 对抗型 | 腹痛腹胀反复发作 3 个月 | ★★★★★ |

**人格类型说明：**

- **配合型**：积极回答问题，有问必答
- **焦虑型**：话多、反复追问、担心病情严重
- **沉默型**：回答简短，需要医生耐心引导
- **对抗型**：质疑医生、态度强硬，需要展现专业性才会配合

---

## 九、API 接口文档

后端启动后，访问以下地址查看自动生成的交互式 API 文档：

| 文档 | 地址 |
|------|------|
| Swagger UI | http://localhost:8000/docs |
| ReDoc | http://localhost:8000/redoc |

### 主要接口一览

| 方法 | 路径 | 说明 | 认证 |
|------|------|------|------|
| POST | /api/v1/auth/register | 注册 | 否 |
| POST | /api/v1/auth/login | 登录 | 否 |
| GET | /api/v1/auth/me | 获取当前用户 | Bearer |
| GET | /api/v1/patients/ | 患者列表（支持筛选） | Bearer |
| GET | /api/v1/patients/{id} | 患者详情 | Bearer |
| POST | /api/v1/patients/ | 创建患者（管理员） | Bearer |
| POST | /api/v1/consultations/ | 创建问诊 | Bearer |
| GET | /api/v1/consultations/ | 我的问诊列表 | Bearer |
| GET | /api/v1/consultations/{id} | 问诊详情（含消息） | Bearer |
| POST | /api/v1/consultations/{id}/messages | 发送消息 | Bearer |
| POST | /api/v1/consultations/{id}/end | 结束问诊 | Bearer |
| POST | /api/v1/evaluations/ | 触发评估 | Bearer |
| GET | /api/v1/evaluations/{id} | 查看评估报告 | Bearer |
| GET | /api/v1/stats/ | 数据统计（管理员） | Bearer |

### 认证密码说明

- 注册与登录接口不再限制密码长度
- 后端密码哈希使用可处理任意长度输入的算法链路
- 历史 bcrypt 哈希密码可继续登录，系统会自动兼容验证

### 错误响应规范

- 登录接口错误响应统一为：`error_code`、`message`、`request_id`
- 参数错误返回 `422 VALIDATION_ERROR`
- 认证失败返回 `401 AUTH_INVALID_CREDENTIALS`
- 数据库故障返回 `503 DB_UNAVAILABLE`
- 未知异常返回 `500 INTERNAL_SERVER_ERROR`，不返回敏感堆栈信息

### 灰度验证建议

- 灰度期按 `request_id` 聚合 `POST /api/v1/auth/login` 的 `5xx` 占比
- 指标口径：`同类错误率 = error_code=INTERNAL_SERVER_ERROR 的登录请求数 / 登录总请求数`
- 发布前门禁：最近 24 小时同类错误率 `< 0.1%`
- 出现告警时按 `request_id` 回查日志中的完整堆栈并执行回滚

---

## 十、多智能体协同架构

```
                         ┌────────────────────┐
                         │   问诊对话记录      │
                         └────────┬───────────┘
                                  │
                    ┌─────────────┼─────────────┐
                    ▼             ▼             ▼
            ┌──────────┐  ┌──────────┐  ┌──────────┐
            │ 问诊分析  │  │ 知识核对  │  │ 人文关怀  │
            │ 智能体    │  │ 智能体    │  │ 智能体    │
            └─────┬────┘  └─────┬────┘  └─────┬────┘
                  │             │             │
                  │    （三者并行执行）         │
                  └─────────────┼─────────────┘
                                ▼
                       ┌──────────────┐
                       │ 综合评分智能体 │
                       └───────┬──────┘
                               ▼
                       ┌──────────────┐
                       │ 建议指导智能体 │
                       └───────┬──────┘
                               ▼
                        ┌────────────┐
                        │  评估报告   │
                        └────────────┘
```

- 前三个智能体通过 `asyncio.gather` **并行调用**，提高响应速度
- 综合评分智能体汇总三项结果后**串行调用**建议智能体
- 所有智能体均通过阿里云百炼平台 Qwen API 驱动

---

## 十一、常见问题

### Q: 后端启动报 `ModuleNotFoundError`
确保已激活虚拟环境：`.\venv\Scripts\Activate.ps1`，终端前方应显示 `(venv)`。

### Q: 数据库连接失败
检查 `backend/.env` 中的 MySQL 密码是否正确，确认 MySQL 服务正在运行：`Get-Service MySQL80`。

### Q: 问诊时虚拟患者不回复
检查系统环境变量 `DASHSCOPE_API_KEY` 是否已配置。可在终端验证：`echo $env:DASHSCOPE_API_KEY`。

### Q: 评估报告生成很慢
正常现象，需要串行调用 5 个 AI 智能体（其中 3 个并行），通常需要 15-30 秒。

### Q: 前端页面空白或报 CORS 错误
确保后端已在 8000 端口运行，前端 Vite 的 API 代理配置会自动将 `/api` 请求转发到后端。
