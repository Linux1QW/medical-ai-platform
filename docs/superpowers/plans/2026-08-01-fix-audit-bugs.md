# 修复项目审计问题实施计划

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 修复前一轮审计发现的部署、依赖、路由、动态难度和消息并发问题，并用回归测试验证。

**Architecture:** 保留现有 `app` 顶层导入约定，在 Docker 运行时显式加入 `/app/backend`。病例动态难度通过虚拟患者的稳定 `case_id` 关联评估历史；消息写入通过锁定问诊行保证同一问诊串行分配序号。

**Tech Stack:** FastAPI、SQLAlchemy async、Alembic、Celery、Pytest、Docker Compose。

## Global Constraints

- 不改变现有 API 响应字段和 Celery 任务名称。
- 修复必须兼容现有数据库初始化脚本和 Alembic 迁移流程。
- 每项行为修复都要有针对性回归测试；不因格式化顺手改写无关文件。

---

### Task 1: 修复 Docker Python 导入路径和 Celery 依赖

**Files:**
- Modify: `Dockerfile.backend`
- Modify: `backend/setup.py`
- Modify: `docker-compose.yml`
- Test: `backend/tests/test_deployment_configuration.py`

- [x] 增加 Docker 的 `PYTHONPATH=/app/backend`，使 `backend.app.main` 与代码中的 `from app...` 同时可导入。
- [x] 在 `setup.py` 加入 `celery[redis]==5.6.3`，与 `requirements.txt` 保持一致。
- [x] 将 worker/beat 的 Celery app 路径改为与容器导入路径一致且可验证的 `app.celery_app`。
- [x] 测试配置文本包含运行时路径、Celery 依赖和两个 Celery 命令。

### Task 2: 修复患者导出路由被动态路由拦截

**Files:**
- Modify: `backend/app/api/v1/patients.py`
- Test: `backend/tests/test_deployment_configuration.py`

- [x] 把静态 `GET /export` 路由注册到 `GET /{patient_id}` 之前。
- [x] 用路由表断言 `/export` 的匹配项先于 `/{patient_id}`。

### Task 3: 修复病例与虚拟患者的动态难度关联

**Files:**
- Modify: `backend/app/models/patient.py`
- Modify: `backend/app/schemas/patient.py`
- Modify: `backend/app/services/case_recommender.py`
- Modify: `backend/app/api/v1/cases.py`
- Modify: `database/init.sql`
- Create: `backend/alembic/versions/<new>_add_case_id_to_virtual_patients.py`
- Test: `backend/tests/services/test_case_recommender.py`

- [x] 为虚拟患者增加可空、唯一索引的 `case_id`，避免用姓名承担关联键职责。
- [x] 病例加载时把 `case_id` 传给动态难度查询；没有数据库关联时继续使用静态难度。
- [x] API 按 `VirtualPatient.case_id` 查询，不再用 `name == case_id`。
- [x] 为初始化 SQL 和 Alembic 增加同样的字段定义，并覆盖“有关联/无关联”两种结果。

### Task 4: 修复同一问诊消息序号并发竞争

**Files:**
- Modify: `backend/app/services/consultation_service.py`
- Modify: `backend/app/models/consultation.py`
- Modify: `database/init.sql`
- Create: `backend/alembic/versions/<new>_unique_message_sequence.py`
- Test: `backend/tests/services/test_consultation_service.py`

- [x] 在普通和 SSE 消息入口开始时锁定对应 `Consultation` 行，再读取消息和分配序号。
- [x] 保持一轮医生/患者消息使用连续的两个序号。
- [x] 为 `(consultation_id, sequence)` 增加唯一约束，防止并发异常静默落库。
- [x] 增加锁定查询和唯一约束的回归断言。

### Task 5: 全量验证

- [x] 运行新增回归测试。
- [x] 运行 `python -m compileall -q backend`。
- [x] 运行 `python -m ruff check backend`。
- [ ] 运行前端已有测试和构建，确认无回归。
