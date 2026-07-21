# Prompt 外置化 & LLM Provider 适配器

> 本文档说明两项基础设施增强：
> 1. **Prompt 外置化 + 版本管理**（`PromptManager`）
> 2. **LLM Provider 适配器抽象层 + failover 真实切换**（`ProviderAdapter`）
>
> 两项改造均遵循「零破坏、向后兼容、渐进式」原则，未引入任何新的第三方依赖。

---

## 目录

- [一、背景与动机](#一背景与动机)
- [二、问题 1：Prompt 外置化 + 版本管理](#二问题-1prompt-外置化--版本管理)
  - [2.1 目录结构](#21-目录结构)
  - [2.2 PromptManager API](#22-promptmanager-api)
  - [2.3 版本管理与灰度](#23-版本管理与灰度)
  - [2.4 Agent 迁移](#24-agent-迁移)
  - [2.5 如何新增 / 升级一个 Prompt](#25-如何新增--升级一个-prompt)
- [三、问题 2：Provider 适配器 + failover 真实切换](#三问题-2provider-适配器--failover-真实切换)
  - [3.1 修复的 Bug：failover「半接线」](#31-修复的-bugfailover半接线)
  - [3.2 适配器分层设计](#32-适配器分层设计)
  - [3.3 qwen_client 的真实切换](#33-qwen_client-的真实切换)
  - [3.4 通用 LLM_* 配置](#34-通用-llm_-配置)
  - [3.5 如何接入新的 Provider](#35-如何接入新的-provider)
- [四、配置项速查](#四配置项速查)
- [五、测试与回归](#五测试与回归)
- [六、兼容性与风险](#六兼容性与风险)

---

## 一、背景与动机

平台由多个评估智能体（诊断 / 治疗 / 建议 / 人文关怀 / 安全 / 反思 / 问诊 / 医学知识核对等）组成，
此前存在两个工程隐患：

| # | 问题 | 影响 |
|---|------|------|
| 1 | 所有 system prompt 以三引号字符串**硬编码**在各 agent 源码中 | 改 prompt 必须改代码、无法灰度 / A-B、无法追溯版本 |
| 2 | LLM 客户端在模块加载时固定创建，`failover` 切换 Provider 时只更新索引/计数，**不重建底层客户端** | 跨 Provider 熔断切换后请求仍打向原端点，failover 形同虚设（「半接线」） |

本次改造分别解决上述两点。

---

## 二、问题 1：Prompt 外置化 + 版本管理

将全部 agent 的 system prompt 从源码抽离为**独立文本文件**，并引入轻量级
`PromptManager` 负责加载、版本选择、变量渲染与缓存。

### 2.1 目录结构

```
backend/app/prompts/
├── manifest.json                 # 声明每个 prompt 的活跃版本 + 描述
├── diagnosis.system/
│   └── v1.txt
├── treatment.system/
│   └── v1.txt
├── suggestion.system/
│   └── v1.txt
├── humanistic.empathy_system/
│   └── v1.txt
├── humanistic.behavior_system/
│   └── v1.txt
├── safety.system/
│   └── v1.txt
├── reflection.system/
│   └── v1.txt
├── inquiry.slot_filling_system/
│   └── v1.txt
├── inquiry.logic_efficiency_system/
│   └── v1.txt
├── knowledge.consistency_system/
│   └── v1.txt
├── knowledge.tool_use_system/
│   └── v1.txt
├── knowledge.react_system/
│   └── v1.txt
└── scoring.summary_system/
    └── v1.txt
```

- **Prompt Key**：形如 `<domain>.<name>`（如 `diagnosis.system`），全局唯一。
- **版本文件**：`<key>/<version>.txt`，一个 key 可以有多个版本（`v1.txt`、`v2.txt` …）。
- **manifest.json**：登记每个 key 的 `active` 活跃版本与 `description`，是加载 prompt 的唯一「目录」。

`manifest.json` 片段：

```json
{
  "diagnosis.system": {
    "active": "v1",
    "description": "诊断结果评估智能体 system prompt"
  },
  "scoring.summary_system": {
    "active": "v1",
    "description": "五维综合评估摘要 system prompt（运行时追加分数约束）"
  }
}
```

### 2.2 PromptManager API

模块位置：`backend/app/services/prompts/`，对外导出：

```python
from app.services.prompts import (
    PromptManager,        # 管理器类（单例）
    get_prompt,           # 便捷函数：取活跃/指定版本的 prompt 文本
    get_prompt_manager,   # 获取全局单例
    render_prompt,        # 便捷函数：取 prompt 并用变量渲染
)
```

核心方法：

| 方法 | 说明 |
|------|------|
| `get_prompt(key, version=None, default=None)` | 返回 prompt 文本；`version` 为空时取活跃版本；文件缺失且给了 `default` 则降级返回 default，否则抛错 |
| `render_prompt(key, variables, version=None)` | 取 prompt 后用 `str.format_map` 渲染 `{占位符}`；**缺失的占位符原样保留**（`_SafeDict`），避免 `KeyError` |
| `get_active_version(key)` | 返回该 key 当前活跃版本 |
| `list_prompts()` / `list_versions(key)` | 列出所有 key / 某 key 的全部版本文件 |
| `reload()` | 清空缓存并重载 manifest（热更新，无需重启进程） |

特性：
- **单例 + 内存缓存**：首次读取后缓存文本，二次读取不再触碰磁盘；`reload()` 可强制刷新。
- **线程安全**：加载与缓存写入使用线程锁保护。
- **零新依赖**：仅用标准库 `json` + `str.format_map`。

### 2.3 版本管理与灰度

通过环境变量 / 配置项 `PROMPT_ACTIVE_VERSIONS`（JSON 字符串）**覆盖** manifest 中的活跃版本，
用于灰度发布 / A-B 对比，无需改动任何 prompt 文件或代码：

```bash
# .env
# 让 diagnosis.system 临时切到 v2，其余仍走 manifest.active
PROMPT_ACTIVE_VERSIONS={"diagnosis.system": "v2"}
```

- 值为非法 JSON 时**安全忽略**，回退到 manifest 声明的活跃版本。
- 空对象 `{}`（默认）表示全部使用 `manifest.active`。

优先级：`get_prompt(version=...)` 显式指定 > `PROMPT_ACTIVE_VERSIONS` 覆盖 > `manifest.active`。

### 2.4 Agent 迁移

8 个 agent 的 **13 个** system prompt 已全部外置。迁移模式（以诊断 agent 为例）：

```python
# 迁移前
DIAGNOSIS_SYSTEM_PROMPT = """你是资深临床诊断评估专家……"""

# 迁移后
from app.services.prompts import get_prompt
DIAGNOSIS_SYSTEM_PROMPT = get_prompt("diagnosis.system")
```

Key ↔ Agent 对照：

| Prompt Key | 所属 Agent |
|------------|-----------|
| `diagnosis.system` | 诊断评估 |
| `treatment.system` | 治疗方案评估 |
| `suggestion.system` | 建议指导（对比学习） |
| `humanistic.empathy_system` / `humanistic.behavior_system` | 人文关怀评估 |
| `safety.system` | 急危重安全门控 |
| `reflection.system` | 评估质量反思（ReAct） |
| `inquiry.slot_filling_system` / `inquiry.logic_efficiency_system` | 问诊技巧评估 |
| `knowledge.consistency_system` / `knowledge.tool_use_system` / `knowledge.react_system` | 医学知识核对 |
| `scoring.summary_system` | 五维综合评估摘要 |

> 注：`scoring_agent.py` 从 `scoring/summary.py` 导入 `SYSTEM_PROMPT`，只需迁移 `summary.py` 一处。

### 2.5 如何新增 / 升级一个 Prompt

**新增版本（升级）**：
1. 在 `backend/app/prompts/<key>/` 下新增 `v2.txt`。
2. 灰度：设 `PROMPT_ACTIVE_VERSIONS={"<key>": "v2"}` 验证；稳定后把 `manifest.json` 的 `active` 改为 `v2`。

**新增一个 Prompt**：
1. 新建目录 `backend/app/prompts/<new.key>/v1.txt`。
2. 在 `manifest.json` 登记 `{"<new.key>": {"active": "v1", "description": "..."}}`。
3. 代码中 `get_prompt("<new.key>")` 使用。

---

## 三、问题 2：Provider 适配器 + failover 真实切换

### 3.1 修复的 Bug：failover「半接线」

改造前：
- `qwen_client.py` 在模块加载时创建固定的 `client = AsyncOpenAI(QWEN 配置)`。
- `failover_manager.switch_to_next()` 只更新「当前 Provider 索引 / 失败计数」，**没有重建 `client`**。

后果：主 Provider 连续失败触发熔断切换后，`client` 仍指向原端点 → 请求继续打向已故障的 Provider，failover 名存实亡。

修复思路：引入 **ProviderAdapter 抽象层**统一管理底层客户端；failover 命中切换时，
依据新 Provider 配置**真实重建**底层客户端。

### 3.2 适配器分层设计

新增包：`backend/app/services/llm/`

```
app/services/llm/
├── __init__.py            # 统一导出
├── base.py                # ProviderConfig + ProviderAdapter(ABC)
├── openai_compatible.py   # OpenAICompatibleAdapter（内置实现）
└── registry.py            # 适配器注册表 + create_adapter 工厂
```

**`ProviderConfig`（不可变 dataclass）**
- 字段：`api_key` / `base_url` / `model` / `name` / `provider_type` / `timeout`。
- `from_dict(d)`：从 failover / settings 返回的 dict 构造，兼容 `type` 与 `provider_type` 两种键。
- `identity() -> (provider_type, api_key, base_url)`：决定「底层客户端身份」的三元组。
  **`model` 不纳入 identity** —— 同一端点切换模型无需重建连接池。

**`ProviderAdapter`（抽象基类）**
- 持有 `ProviderConfig`，对上层暴露统一的 `client` / `model` 接口。
- 抽象属性 `client`：由子类懒创建并返回底层客户端。

**`OpenAICompatibleAdapter`（内置实现）**
- `provider_type = "openai_compatible"`。
- 覆盖任意 OpenAI Chat Completions 兼容端点：阿里云百炼 compatible-mode、OpenAI 官方、DeepSeek、Moonshot 等。
- **懒创建**：首次访问 `.client` 时才建立 `httpx.AsyncClient` 连接池 + `AsyncOpenAI` 实例。
- `aclose()`：释放底层连接池（切换后可选调用）。

**`registry.py`（注册表 / 工厂）**
- `register_adapter(cls)`：按 `provider_type` 注册（可作装饰器）；拒绝空或 `base` 类型。
- `get_adapter_class(type)`：未知类型**安全回退** `OpenAICompatibleAdapter`（degrade-safe）。
- `create_adapter(config)`：按配置实例化适配器。
- 模块加载时内置注册 `OpenAICompatibleAdapter`。

### 3.3 qwen_client 的真实切换

`qwen_client.py` 关键改动：

```python
# 通过适配器创建底层客户端
_active_adapter = create_adapter(
    ProviderConfig.from_dict(failover_manager.get_current_provider())
)
_active_model: str = _active_adapter.model or settings.QWEN_MODEL
client: AsyncOpenAI = _active_adapter.client   # 模块级 client 保留（向后兼容 + 测试注入点）


def _refresh_active_provider(provider: dict) -> None:
    """failover 命中时刷新活跃适配器与模块级 client。"""
    global _active_adapter, _active_model, client
    new_config = ProviderConfig.from_dict(provider)
    if new_config.identity() != _active_adapter.config.identity():
        _active_adapter = create_adapter(new_config)   # 真实重建
        client = _active_adapter.client
        logger.warning(f"LLM 客户端已切换到 Provider: {_active_adapter.describe()}")
    _active_model = new_config.model or _active_model
```

在重试循环的失败切换分支：

```python
if failover_manager.should_switch():
    new_provider = failover_manager.switch_to_next()
    _refresh_active_provider(new_provider)   # ← 真实切换（而非仅记账）
    logger.warning(f"Failover: 切换到 Provider '{new_provider['name']}'")
```

设计要点：
- **保留模块级 `client`**：重试循环通过全局名 `client` 读取当前客户端；`failover` 命中时由 `_refresh_active_provider` 重新赋值。这同时保证既有测试（直接 `@patch("app.services.qwen_client.client")`）无需改动。
- **happy path 不触碰适配器**：正常调用路径**不会**调用 `_refresh_active_provider`，因此不会覆盖被 patch 的 mock 客户端。
- **按 identity 判定重建**：只有当 Provider 类型 / api_key / base_url 变化时才重建底层客户端；仅模型变化时只更新 `_active_model`，避免无谓重建连接池。
- **对外签名不变**：`call_qwen_chat` / `call_qwen_with_tools` 的参数与返回完全保持不变。

### 3.4 通用 LLM_* 配置

`config.py` 新增一组**框架/厂商无关**的通用配置，为空时回退到既有 `QWEN_*`：

```python
LLM_PROVIDER_TYPE: str = "openai_compatible"  # 适配器类型（对应注册表）
LLM_API_KEY: str = ""        # 为空回退 QWEN_API_KEY
LLM_API_BASE_URL: str = ""   # 为空回退 QWEN_API_BASE_URL
LLM_MODEL: str = ""          # 为空回退 QWEN_MODEL
```

`get_llm_providers()` 的默认项也带上了 `type` 字段并改用通用配置，从而在**不破坏**现有阿里云百炼配置的前提下，可无缝接入任意 OpenAI 兼容服务。

### 3.5 如何接入新的 Provider

**方式 A：多 Provider failover（推荐）**
在 `.env` 配置 `LLM_PROVIDERS`（JSON 数组），每项含 `type`：

```bash
LLM_PROVIDERS=[
  {"name":"primary","type":"openai_compatible","api_key":"k1","base_url":"https://dashscope.aliyuncs.com/compatible-mode/v1","model":"qwen-max"},
  {"name":"backup","type":"openai_compatible","api_key":"k2","base_url":"https://api.deepseek.com/v1","model":"deepseek-chat"}
]
LLM_CIRCUIT_BREAKER_THRESHOLD=3
```

主 Provider 连续失败达阈值即自动切换到备用，并**真实重建**客户端。

**方式 B：自定义非 OpenAI 兼容协议**
1. 继承 `ProviderAdapter`，实现 `client` 属性（返回具备 `.chat.completions.create(...)` 接口的对象）。
2. 设 `provider_type = "your-type"` 并 `register_adapter(YourAdapter)`。
3. 在 Provider 配置里用 `"type": "your-type"`。

---

## 四、配置项速查

| 配置项 | 默认值 | 说明 |
|--------|--------|------|
| `PROMPT_ACTIVE_VERSIONS` | `{}` | JSON，覆盖 manifest 活跃版本，用于灰度 / A-B |
| `LLM_PROVIDER_TYPE` | `openai_compatible` | 默认适配器类型 |
| `LLM_API_KEY` | `""` | 通用 Key，空回退 `QWEN_API_KEY` |
| `LLM_API_BASE_URL` | `""` | 通用 Base URL，空回退 `QWEN_API_BASE_URL` |
| `LLM_MODEL` | `""` | 通用模型，空回退 `QWEN_MODEL` |
| `LLM_PROVIDERS` | `[]` | 多 Provider failover 配置（JSON 数组） |
| `LLM_CIRCUIT_BREAKER_THRESHOLD` | `3` | 连续失败 N 次后切换备用 Provider |

---

## 五、测试与回归

新增单测：

| 测试文件 | 用例数 | 覆盖 |
|----------|--------|------|
| `tests/services/test_prompt_manager.py` | 18 | 加载 / 版本切换 / 渲染 / 缺失降级 / 覆盖开关 / 缓存 / reload 热更新 / 真实仓库 13 key 可加载 |
| `tests/services/test_llm_adapter.py` | 17 | ProviderConfig / identity / 注册表回退 / 懒创建 / **failover 真实切换 client** / 相同 identity 不重建但更新模型 |

回归结果：

```
全量：530 passed, 18 skipped
（既有 test_qwen_client_tools + test_llm_failover 全绿；
  test_call_qwen_chat_unchanged 直接 patch 模块级 client 的回归用例通过）
```

> 遗留的 warning（aiomysql `__del__`、mock 协程未 await）均为**既有**、与本次改造无关。

---

## 六、兼容性与风险

- **零新依赖**：Prompt 用标准库；适配器复用既有 `openai` / `httpx`。
- **向后兼容**：
  - 未配置 `LLM_*` 时行为与改造前完全一致（回退 `QWEN_*`）。
  - `call_qwen_chat` / `call_qwen_with_tools` 对外签名不变。
  - 模块级 `client` 仍存在，既有直接 patch 它的测试无需改动。
- **降级安全**：
  - Prompt 覆盖开关非法 JSON → 忽略回退 manifest。
  - 未知 `provider_type` → 回退 OpenAI 兼容适配器。
- **注意**：`_call_qwen_api_with_tools`（工具调用路径）本身不触发 failover 切换，
  其 Provider 切换依赖 chat 路径已触发的熔断（与改造前行为一致）。
