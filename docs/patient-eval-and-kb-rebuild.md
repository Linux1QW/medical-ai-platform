# 患者智能体评测闭环 与 向量库重建 — 使用说明

本文档说明两块相互独立、均为**离线运行**的能力：

1. **患者智能体评测闭环（P0）**：对虚拟患者智能体做三臂对照回放、软硬指标采集、Badcase 归档与回归红线门禁，用于量化「记忆账本」与「工具调用」两项升级带来的增量。
2. **向量库两阶段重建**：把医学指南 RAG 知识库的重建拆成「花钱的 embedding 落盘」与「免费的灌库」两阶段，并修复 ChromaDB 1.5.7 大集合 HNSW 段跨进程冷读崩溃的问题。

> 约束：所有脚本均**不改动生产请求路径**（零回归面）。评测脚本对真实 LLM 的调用会消耗 DashScope 额度；纯函数逻辑均有单测覆盖且 mock 掉 LLM，测试不产生费用。

---

## 目录

1. [患者智能体评测闭环](#1-患者智能体评测闭环)
   - [1.1 组件总览](#11-组件总览)
   - [1.2 评测集固化](#12-评测集固化)
   - [1.3 三臂回放](#13-三臂回放)
   - [1.4 LLM-as-Judge 软分](#14-llm-as-judge-软分)
   - [1.5 Badcase 归档](#15-badcase-归档)
   - [1.6 回归红线门禁](#16-回归红线门禁)
   - [1.7 完整操作流程](#17-完整操作流程)
2. [向量库两阶段重建](#2-向量库两阶段重建)
   - [2.1 两阶段设计](#21-两阶段设计)
   - [2.2 ChromaDB 1.5.7 HNSW 跨进程读修复](#22-chromadb-157-hnsw-跨进程读修复)
   - [2.3 rebuild_kb_from_cache.py 用法](#23-rebuild_kb_from_cachepy-用法)
3. [模型配置](#3-模型配置)
4. [隐私与提交约束](#4-隐私与提交约束)
5. [故障排除](#5-故障排除)

---

## 1. 患者智能体评测闭环

### 1.1 组件总览

| 组件 | 文件 | 职责 |
| --- | --- | --- |
| 评测集固化 | `backend/scripts/build_eval_set.py` | 分层抽样确定性生成评测集 |
| 评测集加载 | `backend/evaluation/patient_eval_set.py` | `load_eval_set()` 校验/去重/跳过 `_meta` |
| 三臂回放 | `backend/scripts/ab_patient_replay.py` | legacy / agent_ledger / agent_tool 三臂对照 |
| 软分裁判 | `backend/evaluation/patient_judge.py` | 同模型低温 LLM-as-Judge 四维评分 |
| 回归门禁 | `backend/evaluation/patient_regression.py`、`backend/scripts/eval_regression.py` | 报告 diff + 阈值破线检测 |
| 红线阈值 | `backend/evaluation/patient_ab_thresholds.json` | 各臂关键指标水位 |
| 评测集数据 | `backend/evaluation/patient_cases/patient_sim_v1.jsonl` | 版本化 18 例分层样本 |
| 报告产物 | `backend/evaluation/reports/patient_ab/` | `ab_*.json` / `badcase_*.jsonl`（已 gitignore） |

命令统一在 `backend/` 目录下、用虚拟环境解释器执行；PowerShell 请用 `;` 分隔语句，并建议设置 UTF-8 输出：

```powershell
cd backend
$env:PYTHONIOENCODING="utf-8"; $env:PYTHONUTF8="1"
```

### 1.2 评测集固化

`build_eval_set.py` 遍历 `dataset/`，读取每例的「人格·性格」与「主诊断」，按 (人格类型 × 诊断科室桶) 分层，用固定随机种子确定性抽样，跳过门诊对话为空的病例。

```powershell
.\venv\Scripts\python.exe scripts\build_eval_set.py
```

| 参数 | 默认 | 说明 |
| --- | --- | --- |
| `--n` | 18 | 目标抽样例数 |
| `--seed` | 42 | 随机种子（固定→可复现） |
| `--version` | v1 | 评测集版本号，写入 `_meta` |
| `--out` | `evaluation/patient_cases/patient_sim_v1.jsonl` | 输出路径 |

**确定性保证**：输入按 `case_id` 排序，人格组按 sorted 顺序，单一 `random.Random(seed)` 顺序消耗，诊断多样性为确定性贪心排序 —— 同 `(dataset, n, seed)` 重跑抽样结果完全一致（仅 `_meta.generated_at` 时间戳会变）。

输出 JSONL 首行为 `_meta`，其余每行一个样本，仅含 `case_id / personality / diagnosis / turns_available` 四个**元数据**字段，不含任何对话文本。加载侧：

```python
from evaluation.patient_eval_set import load_eval_set
cases = load_eval_set("evaluation/patient_cases/patient_sim_v1.jsonl")  # 校验 schema、去重、跳过 _meta 行
```

### 1.3 三臂回放

`ab_patient_replay.py` 对同一批病例依次跑三臂，隔离出「账本增量」和「工具增量」两段：

| 臂 | 含义 |
| --- | --- |
| `legacy` | 旧无记忆回复路径（基线） |
| `agent_ledger` | 披露账本 + 阶段状态机，**工具关** |
| `agent_tool` | 账本 + 患者专属工具，**工具开** |

```powershell
# 跑评测集全量（读 patient_sim_v1.jsonl 的 case_id 列表）
.\venv\Scripts\python.exe scripts\ab_patient_replay.py --cases @eval_set --turns-cap 10 --judge

# 冒烟：只跑前 3 例、省额度关掉 Judge
.\venv\Scripts\python.exe scripts\ab_patient_replay.py --cases @eval_set --limit 3 --turns-cap 8 --no-judge

# 断点续跑：配额中断后接力，已完成病例自动跳过
.\venv\Scripts\python.exe scripts\ab_patient_replay.py --cases @eval_set --resume evaluation\reports\patient_ab\ab_<ts>.json
```

| 参数 | 默认 | 说明 |
| --- | --- | --- |
| `--cases` | 空 | 逗号分隔 case_id；传 `@eval_set` 读评测集全部；空则默认取样本前 `--limit` 例 |
| `--limit` | 2 | 与 `--cases` / `@eval_set` 配合限制例数 |
| `--turns-cap` | 10 | 每例回放的最大医生轮数 |
| `--judge` / `--no-judge` | 开 | 是否跑 LLM-as-Judge 软分；`--no-judge` 省额度只出硬指标 |
| `--resume <report.json>` | 空 | 从既有报告续跑，跳过已完成病例 |

**健壮性**：每例回放包在独立 try/except 内，单例失败仅记入报告 `failed` 列表并继续；每例完成后**原子落盘**（`_flush()`），崩溃/中断也保留已完成进度。续跑时会剔除历史失败记录中将被重跑的病例，避免同一 case 既算成功又算失败的簿记矛盾。

**硬指标**（每臂 summary）：披露率 `disclosure_rate`、信任终值、情绪分布、阶段路径、重生成次数、工具降级率 `tool_degrade_rate`、延迟与 LLM 调用数。

### 1.4 LLM-as-Judge 软分

`patient_judge.py` 用**同模型低温**（temperature=0.0）rubric 对每轮患者回复打分，四个维度 1–5 分：

1. 角色一致性（是否泄露档案外信息 / AI 身份）
2. 医学合理性
3. 口语自然度
4. 信息披露时机是否符合人格策略

`overall` 为四维均分，另附一句理由。**降级策略**：LLM 调用异常或 JSON 解析失败时返回 `JudgeScore(degraded=True, scores=全None)`，不阻断回放；单维缺失只降该维（保留其余维度），`overall` 置 `None`；分数越界自动钳制到 [1,5]。

> Judge 走 `evaluation.patient_judge` 命名空间，不被各臂的延迟/调用计数器统计，避免污染硬指标归因。

### 1.5 Badcase 归档与失败模式归因

任一 Judge 维度 ≤2 或 `overall<3` 的轮次，连同上下文快照追加写入
`evaluation/reports/patient_ab/badcase_<ts>.jsonl`，字段含
`case_id / turn / arm / doctor / reply / scores / reason / attribution 归因占位`。

**归因标注**：`scripts/label_badcases.py`（纯离线、无 LLM 成本）按评委已给出的四维分
做**规则式归因**，为每条 badcase 回填 `attribution`（主失败模式）/ `attribution_label`
（中文）/ `attribution_modes`（全部破线维），并另出一份**去标识失败模式清单**
`badcase_summary_<ts>.json`（不含对话文本，可外发/入档）。判定阈值与归档口径一致
（维度分 ≤2 即破线）；主因取分最低维，并列按优先级 `role_consistency >
medical_plausibility > disclosure_timing > naturalness` 仲裁；无单维破线则归 `low_overall`。

```powershell
# 缺省取最新 badcase_*.jsonl，原地回填 attribution 并产出去标识清单
.\venv\Scripts\python.exe scripts\label_badcases.py
.\venv\Scripts\python.exe scripts\label_badcases.py --badcase <path> --no-inplace
```

四类失败模式 → 归因维度：角色一致性破坏 `role_consistency`、医学/上下文合理性不足
`medical_plausibility`、生硬失真(AI腔) `naturalness`、披露时机失当 `disclosure_timing`。

> 基线快照（`badcase_20260731_171321.jsonl`，18 例三臂，16 条 badcase）主因分布：
> 角色一致性破坏 7 / 披露时机失当 5 / 医学·上下文合理性 3 / 生硬失真 1；
> 按臂 agent_tool 8 条、agent_ledger 6 条、legacy 2 条。**角色一致性**是首要待改进方向。

### 1.6 回归红线门禁

红线定义在 `patient_ab_thresholds.json`，键格式 `<metric>_min` / `<metric>_max`，对应各臂 summary 字段（**回归引擎按病例取均值后比对**）；**缺失的指标标 SKIP，不算破线**。当前基线（`ab_20260731_171321.json`，qwen3.7-plus，18 例，`--judge`，0 失败）回填的水位：

```jsonc
{
  "agent_ledger": { "disclosure_rate_min": 0.45, "judge_overall_avg_min": 4.5 },
  "agent_tool":   { "disclosure_rate_min": 0.40, "judge_overall_avg_min": 4.4,
                    "tool_degrade_rate_max": 0.30 }
}
```

- `disclosure_rate_min` 取基线均值（judge 批 ledger 0.501 / tool 0.529）下浮留噪声容差，跨 `--no-judge` 与 `--judge` 两批均在其上，保守留档为 0.45 / 0.40。
- `tool_degrade_rate_max` 取 0.30 上限护栏（基线均值 0.0）。
- `judge_overall_avg_min` 由 Judge 基线均值（ledger 4.949 / tool 4.871）**下浮约 0.4~0.45** 校准而得（→ 4.5 / 4.4），比 disclosure 留更宽容差以吸收 LLM 评分的批间抖动；legacy 为对照臂不设门禁。

检测脚本：

```powershell
# 只校验最新报告是否破线（不传 --report 自动取最新）
.\venv\Scripts\python.exe scripts\eval_regression.py

# 与基线报告做 diff + 阈值对照
.\venv\Scripts\python.exe scripts\eval_regression.py --report <new>.json --baseline <old>.json
```

| 参数 | 说明 |
| --- | --- |
| `--report` | 待检报告 JSON；不传取 `patient_ab/` 下最新 |
| `--baseline` | 对比基线报告 JSON（可选，用于差值对比） |
| `--thresholds` | 阈值文件，默认 `patient_ab_thresholds.json` |

输出 PASS/FAIL 表；**破线以非零退出码**返回，已由 §1.8 的本地 pre-push 钩子挂钩（本项目不接 CI nightly，以避免在 CI 中消耗 LLM 额度）。

### 1.7 完整操作流程

```powershell
cd backend
$env:PYTHONIOENCODING="utf-8"; $env:PYTHONUTF8="1"

# 0) 纯函数测试（不耗额度）
.\venv\Scripts\python.exe -m pytest tests/evaluation -q

# 1) 生成评测集（不耗额度）
.\venv\Scripts\python.exe scripts\build_eval_set.py

# 2) 冒烟 3 例，验证管道 + Judge 采集
.\venv\Scripts\python.exe scripts\ab_patient_replay.py --cases @eval_set --limit 3 --turns-cap 8 --judge

# 3) 全量 18 例三臂回放
.\venv\Scripts\python.exe scripts\ab_patient_replay.py --cases @eval_set --turns-cap 10 --judge

# 4) 回填阈值后跑回归自检
.\venv\Scripts\python.exe scripts\eval_regression.py

# 5) （一次性）安装 pre-push 回归护栏
.\venv\Scripts\python.exe scripts\install_git_hooks.py
```

### 1.8 pre-push 回归护栏

本项目不接 CI nightly（避免在 CI 消耗 LLM 额度），改用**本地 pre-push 钩子**做轻量护栏：
push 前自动跑一次离线 `eval_regression.py`（无 LLM 调用），校验最新回放报告不破线。

钩子源 `scripts/hooks/pre-push` 随仓库版本化，由 `scripts/install_git_hooks.py` 拷贝到
`.git/hooks/pre-push`（`.git` 不入库，故需每个克隆手动装一次）：

```powershell
.\venv\Scripts\python.exe scripts\install_git_hooks.py            # 安装
.\venv\Scripts\python.exe scripts\install_git_hooks.py --check    # 查看状态
.\venv\Scripts\python.exe scripts\install_git_hooks.py --uninstall # 卸载
```

拦截语义（保守优先，不误伤正常提交）：

- 仅当**确有报告且破线（FAIL，退出码 1）**时拦截 push；
- 无报告（退出码 2）/ 缺解释器等基础设施缺失一律放行；
- 临时跳过：`SKIP_EVAL_REGRESSION=1 git push`；
- 安装器用标记行（`QODER-MANAGED-HOOK`）辨认托管钩子，绝不覆盖/删除用户自定义钩子（除非 `--force`）。

### 1.9 角色一致性强化（针对首要失败模式）

基于 §1.5 归因结论（role_consistency 是首要失败模式，16 条 badcase 中占 7 条），将其拆为两个子问题分别处置：

1. **身份/人称跳戏（代问场景）**：旧版硬编码“你就是患者本人、用第一人称”，与“代替家人咨询”类档案直接冲突。
   `prompts.py` 新增纯函数 `is_proxy_consult()` 从档案主诉/现病史启发式识别代问语义（短语级关键词，避开“代谢/替代”等医学词误撞），`build_role_prompt()` 据此在**本人/代述人**身份行间切换；两条生产路径（`PatientAgent._build_system_prompt` 与 legacy 回退 `_legacy_generate_patient_reply`）改为共用 `build_role_prompt`。`seed_patients.generate_system_prompt` 同步条件化第一人称要求，保持与身份判定一致。
2. **人格漂移**（对抗型越问越顺从、配合型变不耐烦）：`PATIENT_ROLE_WRAPPER` 人格块新增反漂移约束（两路径均受益）；`_build_system_prompt` 末尾【本轮风格】段锚定具体人格名（利用近因效应强化长对话一致性）。

> 冒烟复测（`ab_20260731_214040.json`，3 例×三臂带 judge）：代问病例 patient120_25 的 role_consistency
> 从基线 rc=2（身份跳戏“是我自己”/“我是化疗”）提升到 legacy/agent_ledger 逐轮全 5、agent_tool min 3，
> 回复改为第三人称代家人作答（“他没做过放疗、只做了化疗”）；两例人格漂移（122_25 / 129_26.5）Judge overall
> 均回升至 4.88~5.00。这 3 例的 badcase 从基线多条降到仅 1 条（naturalness 生硬，非角色一致性）。

---

## 2. 向量库两阶段重建

### 2.1 两阶段设计

RAG 医学知识库重建拆成两阶段，避免昂贵的 embedding 费用因 Chroma 崩溃而白花：

| 阶段 | 花费 | 频率 | 产物 |
| --- | --- | --- | --- |
| 阶段一：抽取 + 分块 + embedding 落盘 | **花钱** | PDF 变更时才跑 | `backend/data/embed_cache/*.npz` |
| 阶段二：读 npz 灌入 Chroma 集合 | **免费** | Chroma 损坏就重跑 | 集合 `medical_guidelines_<version>` |

阶段二由 `rebuild_kb_from_cache.py` 完成，幂等：已入库的 source 自动跳过；集合再损坏，删掉重跑本脚本即可，**不花一分钱**。

### 2.2 ChromaDB 1.5.7 HNSW 跨进程读修复

**症状**：灌库后进程内 `count()` 正常，但换一个进程冷读 `count()/get()/search()` 全部报
`Error loading hnsw index`。

**根因**：一旦集合规模越过 `hnsw:sync_threshold`（默认 1000），ChromaDB 1.5.7 会把 HNSW 索引落盘为独立段，而该版本自身的段读取器无法再把它加载回来。向量真值本身安全地持久化在 `chroma.sqlite3`，只是坏在段读取路径上。

**修复**（`backend/app/services/rag/medical_store.py` 的 `COLLECTION_METADATA`）：

```python
_HNSW_SYNC_THRESHOLD = 1_000_000
COLLECTION_METADATA = {
    "hnsw:space": "cosine",
    "hnsw:sync_threshold": _HNSW_SYNC_THRESHOLD,
    "embedding_dim": EMBEDDING_DIM,
}
```

把阈值设为极大值，索引始终留在 WAL，进程首次查询时从 `chroma.sqlite3` 内存重建，绕开坏段读取路径。**数据零丢失，仅首查有一次性重建开销**（约数秒冷读）。

> ⚠️ **重要**：该 metadata **仅在集合创建时生效**；`get_or_create_collection` 对已存在集合会忽略 `metadata`。存量旧集合仍是默认阈值，必须经 `rebuild_kb_from_cache.py --fresh` 删除重建才真正继承此配置。

### 2.3 rebuild_kb_from_cache.py 用法

```powershell
cd backend
# 增量灌库（已入库 source 自动跳过）
.\venv\Scripts\python.exe scripts\rebuild_kb_from_cache.py --version rag-v1

# 从零重建（先删同名集合，确保继承 sync_threshold 修复）
.\venv\Scripts\python.exe scripts\rebuild_kb_from_cache.py --version rag-v1 --fresh
```

| 参数 | 说明 |
| --- | --- |
| `--version` | 索引版本，默认取 `settings.ACTIVE_INDEX_VERSION` |
| `--fresh` | 重灌前删除同名集合（从零重建）；默认增量 |

脚本末尾会做进程内自检：真实向量检索 + `count()`，顺带把 HNSW 建到内存。

---

## 3. 模型配置

默认对话模型为 **`qwen3.7-plus`**（`backend/app/core/config.py` 默认值 + `backend/.env`）。Embedding 走 DashScope compatible-mode，`EMBEDDING_DIM=1024`。DashScope 账户需保证有可用额度、且未开启「仅免费额度」模式，否则真实回放会失败。

---

## 4. 隐私与提交约束

- `dataset/` 为真实医院门诊记录，含隐私，**禁止提交**（已 gitignore）。
- `backend/evaluation/reports/*`（含 `ab_*.json`、`badcase_*.jsonl`，二者嵌入真实问诊对话）**已 gitignore**，不入库。
- `backend/data/medical_kb/`（向量库，~1GB）、`backend/data/embed_cache/`（~220MB）体积大且可重建，**已 gitignore**。
- 可提交的 `patient_sim_v1.jsonl` 仅含元数据（case_id/人格/诊断/轮数），无姓名与对话文本。
- 纪律：只 `commit`，**推送需显式确认**并走安全审查交接流程。

---

## 5. 故障排除

| 现象 | 原因 / 处理 |
| --- | --- |
| 冷读报 `Error loading hnsw index` | 存量集合未继承 sync_threshold 修复；跑 `rebuild_kb_from_cache.py --fresh` 重建 |
| PowerShell 打印中文报 `UnicodeEncodeError` | 设 `$env:PYTHONIOENCODING="utf-8"; $env:PYTHONUTF8="1"` |
| 回放中途配额耗尽 | 用 `--resume <report>.json` 续跑；或 `--no-judge` 省额度 |
| 某些 PDF 灌库产出空 chunk | 扫描/图片型 PDF 无文本层且 OCR 关闭，属数据属性，非管道故障 |
| Judge 分数全为 null（degraded） | LLM 调用或 JSON 解析失败已降级，不影响硬指标；检查 DashScope 额度与网络 |
| `eval_regression.py` 非零退出 | 有指标破线，查 PASS/FAIL 表定位；确认是真实回归还是阈值过紧 |
