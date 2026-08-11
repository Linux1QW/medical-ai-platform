# V1.2 Remediation Preflight Baseline (v3)

## 基线信息

| 字段 | 值 |
|------|-----|
| 基线提交 SHA | `3c75ac4c268ef2f9ef571e5895cb794d91899ee5` |
| 分支名称 | `codex/v1.2-acceptance-remediation-v3` |
| 记录时间 | 2026-08-11 |

## 工具版本

| 工具 | 版本 |
|------|------|
| Python | 3.10.4 |
| Node.js | v24.14.0 |
| npm | 11.9.0 |
| Docker Compose | **不可用**（本机未安装 Docker） |
| Git | 2.45.1.windows.1 |

## 失败基线结果

### 后端 pytest

```
1 failed, 2019 passed, 18 skipped, 13 warnings in 99.55s
```

**失败用例：**
- `backend/tests/scripts/test_migrate_v11.py::TestMigrateExistingDatabase::test_existing_db_with_operator_succeeds`

### 后端 mypy（严格模式）

```
Found 45 errors in 31 files (checked 251 source files)
```

典型错误包括：
- `no-any-return`：多处函数返回 `Any` 而非声明的具体类型
- `assignment`：`consultation_service.py:58` 参数默认值 `None` 与类型 `dict[Any, Any]` 不兼容（缺少 `Optional`）

### 后端 Ruff

```
Found 8 errors.
```

主要涉及 `backend/scripts/ci/v12_safety_probe.py` 中的延迟导入违规（`E402`）。

### 前端 Vitest

```
Test Files  15 passed (15)
Tests       100 passed (100)
```

**前端测试全部通过。**

### 前端 ESLint

```
eslint . — 无错误输出
```

**前端 lint 通过。**

### 前端 Build

```
✓ built in 11.70s
```

**构建成功**，但存在 chunk 体积警告（`vendor-antd` 1,258 kB > 600 kB 限制）。

## 评估指标基线

| 指标 | 值 | 说明 |
|------|------|------|
| macro-F1 | **0.0490** | 远低于验收阈值 |
| trace completeness | **0.1806** | 远低于验收阈值 |

## 远端证据声明

**远端不存在该分支的 PR、CI 运行记录和 Release Candidate 证据。**

- 无 PR 关联此分支
- 无 CI pipeline 运行记录
- 无 RC 构建产物

## 总结

本基线如实记录了从不可变基线 `3c75ac4` 创建修复分支时的项目状态：

- **后端测试存在 1 个失败**（迁移脚本测试）
- **mypy 严格检查存在 45 个错误**
- **Ruff 存在 8 个错误**
- **macro-F1 为 0.0490**，远未达标
- **trace completeness 为 0.1806**，远未达标
- **前端测试、lint、构建均通过**
- **Docker 环境不可用**
- **远端无 PR/CI/RC 证据**
