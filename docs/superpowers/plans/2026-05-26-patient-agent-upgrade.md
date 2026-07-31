# 患者模拟智能体深度升级 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 为虚拟患者构建披露账本记忆管理、专属工具调用与多轮对话规划三大子系统，使患者回复长对话一致、医学合理、人格动态。

**Architecture:** 新增 `app/services/agents/patient/` 包（memory 披露账本 / planner 阶段状态机 / guard 一致性守卫 / agent 编排 / dynamics 信任动力学 / strategy 策略矩阵 / coverage 覆盖报告），`consultation_service` 将回复生成委托给 `PatientAgent`（异常回退旧逻辑）；患者专属工具复用既有 `BaseTool`/`ToolRegistry`/白名单框架，注册于 `tools/patient/`；记忆状态以 JSON 持久化在 `consultations.memory_state` 列。

**Tech Stack:** Python + FastAPI + SQLAlchemy(async, MySQL) + Pydantic v2 + pytest/pytest-asyncio；LLM 统一走 `app.services.qwen_client.call_qwen_chat`。

## Global Constraints

- 工作目录：`d:\Q123\PyCharm\PycharmProjects\基于多智能体的医生临床问诊评估平台\medical-ai-platform`；后端命令均在 `backend\` 下执行，解释器用 `venv\Scripts\python.exe`
- Shell 为 PowerShell：语句分隔用 `;`，禁止 `&&`
- 测试中**必须 mock 所有 LLM 调用**（`call_qwen_chat`），不允许产生真实 API 费用
- 禁止触碰 ChromaDB / 向量索引相关文件（`backend/data/medical_kb/`、`rag/medical_store.py` 等）
- 只 commit，**绝不 push**（推送需用户显式确认，走 security-scan 交接流程）
- 新建 Python 文件带 `# -*- coding: utf-8 -*-` 头，注释与 docstring 用中文，风格对齐现有代码
- LLM 调用签名固定：`await call_qwen_chat(messages: List[Dict[str,str]], model=None, temperature=0.7, max_tokens=2000) -> str`
- JSON 解析统一用 `app.utils.json_parser.extract_json_dict_from_text(text) -> dict`（顶层非对象抛 ValueError）
- 所有新逻辑必须有异常兜底：患者智能体任何一步失败都不能中断问诊（回退旧无记忆路径）

## File Structure

```
backend/app/services/agents/patient/     ← 新包
  __init__.py          导出 PatientAgent / MemoryState / Fact / extract_facts
  memory.py            Fact、MemoryState、extract_facts（LLM+规则兜底）   [Task 1,2]
  planner.py           STAGES、classify_stage 关键词阶段分类              [Task 3]
  guard.py             update_ledger、check_contradiction                [Task 4]
  prompts.py           PATIENT_ROLE_WRAPPER（从 consultation_service 迁入）[Task 5]
  agent.py             PatientAgent.respond 编排                         [Task 5, 11改, 12改, 13改]
  dynamics.py          信任动力学与敏感事实解锁                           [Task 12]
  strategy.py          人格×阶段策略矩阵                                  [Task 13]
  coverage.py          披露账本 → 问诊覆盖报告                            [Task 14]

backend/app/services/tools/patient/      ← 新包
  __init__.py          register_patient_tools、PATIENT_TOOL_BUDGETS      [Task 11]
  plausible_symptom.py QueryPlausibleSymptom（RAG裁决档案外症状）          [Task 8]
  physiology.py        PhysiologyCalculator（确定性生命体征）             [Task 9]
  emotion.py           EmotionEngine + classify_doctor_behavior          [Task 10]

修改：
  backend/app/models/consultation.py     加 memory_state 列              [Task 6]
  backend/scripts/migrate_patient_memory.py  迁移脚本（新建）             [Task 6]
  backend/app/services/consultation_service.py  委托 PatientAgent        [Task 7]
  backend/app/services/tools/policy.py   加 patient_agent 白名单         [Task 11]
  backend/app/services/tools/__init__.py 注册患者工具                    [Task 11]
  backend/app/services/evaluation_service.py  注入覆盖报告               [Task 14]

测试：
  backend/tests/agents/patient/{__init__,test_memory,test_planner,test_guard,test_agent,test_dynamics,test_strategy,test_coverage}.py
  backend/tests/tools/test_patient_tools.py
  backend/tests/services/test_consultation_service.py（追加用例）
```

---

## P1 记忆账本 + 状态机

### Task 1: 披露账本数据模型（memory.py）

**Files:**
- Create: `backend/app/services/agents/patient/__init__.py`
- Create: `backend/app/services/agents/patient/memory.py`
- Create: `backend/tests/agents/patient/__init__.py`
- Test: `backend/tests/agents/patient/test_memory.py`

**Interfaces:**
- Produces: `Fact(fact_id, category, content, status, disclosed_at_turn, disclosure_condition)`；`MemoryState(facts, trust=0.5, emotion="平静", stage="greeting", stage_history, turn=0, tool_calls)`，方法 `to_json() -> str`、`MemoryState.from_json(raw) -> MemoryState|None`、`facts_by_status(status) -> list[Fact]`、`find_fact(fact_id) -> Fact|None`、`mark(fact_ids, status) -> None`

- [ ] **Step 1: 写失败测试**

`backend/tests/agents/patient/__init__.py` 内容为一行注释 `# -*- coding: utf-8 -*-`。`backend/tests/agents/patient/test_memory.py`：

```python
# -*- coding: utf-8 -*-
"""memory.py 单元测试：披露账本模型与序列化"""
import pytest

from app.services.agents.patient.memory import Fact, MemoryState


def _make_memory():
    return MemoryState(facts=[
        Fact(fact_id="sym_001", category="symptom", content="上腹隐痛"),
        Fact(fact_id="his_001", category="history", content="十年前胃溃疡"),
    ])


class TestMemoryState:
    def test_json_roundtrip(self):
        m = _make_memory()
        m.turn = 3
        restored = MemoryState.from_json(m.to_json())
        assert restored is not None
        assert restored.turn == 3
        assert [f.fact_id for f in restored.facts] == ["sym_001", "his_001"]

    def test_from_json_invalid_returns_none(self):
        assert MemoryState.from_json(None) is None
        assert MemoryState.from_json("") is None
        assert MemoryState.from_json("{broken json") is None

    def test_mark_disclosed_records_turn(self):
        m = _make_memory()
        m.turn = 5
        m.mark(["sym_001"], "disclosed")
        fact = m.find_fact("sym_001")
        assert fact.status == "disclosed"
        assert fact.disclosed_at_turn == 5

    def test_mark_unknown_id_ignored(self):
        m = _make_memory()
        m.mark(["nonexistent"], "denied")  # 不抛异常
        assert all(f.status == "undisclosed" for f in m.facts)

    def test_facts_by_status(self):
        m = _make_memory()
        m.mark(["his_001"], "denied")
        assert [f.fact_id for f in m.facts_by_status("denied")] == ["his_001"]
        assert [f.fact_id for f in m.facts_by_status("undisclosed")] == ["sym_001"]
```

- [ ] **Step 2: 运行测试确认失败**

Run: `cd backend; venv\Scripts\python.exe -m pytest tests\agents\patient\test_memory.py -v`
Expected: FAIL（`ModuleNotFoundError: app.services.agents.patient`）

- [ ] **Step 3: 最小实现**

`backend/app/services/agents/patient/__init__.py`：

```python
# -*- coding: utf-8 -*-
"""患者模拟智能体包 — 披露账本记忆、阶段规划与一致性回复生成"""
from .memory import Fact, MemoryState

__all__ = ["Fact", "MemoryState"]
```

`backend/app/services/agents/patient/memory.py`：

```python
# -*- coding: utf-8 -*-
"""患者智能体记忆管理 — 披露账本（Disclosure Ledger）与会话记忆状态

三层记忆中的 L2 情节记忆：把"患者说过什么/否认过什么"从 LLM 软记忆
升级为结构化状态，保证长对话一致性。序列化后存于 consultations.memory_state。
"""
import logging
from typing import Literal, Optional

from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)

FactCategory = Literal["symptom", "history", "medication", "exam", "lifestyle"]
FactStatus = Literal["undisclosed", "disclosed", "denied"]
DisclosureCondition = Literal["direct_ask", "empathy_unlock", "never_volunteer"]


class Fact(BaseModel):
    """患者档案中的一条原子事实"""
    fact_id: str
    category: FactCategory = "symptom"
    content: str
    status: FactStatus = "undisclosed"
    disclosed_at_turn: Optional[int] = None
    disclosure_condition: DisclosureCondition = "direct_ask"


class MemoryState(BaseModel):
    """会话级记忆状态：披露账本 + 信任 + 情绪 + 问诊阶段"""
    facts: list[Fact] = Field(default_factory=list)
    trust: float = 0.5
    emotion: str = "平静"
    stage: str = "greeting"
    stage_history: list[str] = Field(default_factory=list)
    turn: int = 0
    tool_calls: dict[str, int] = Field(default_factory=dict)

    def to_json(self) -> str:
        return self.model_dump_json()

    @classmethod
    def from_json(cls, raw: Optional[str]) -> Optional["MemoryState"]:
        """解析持久化 JSON，失败返回 None（调用方走初始化路径）"""
        if not raw:
            return None
        try:
            return cls.model_validate_json(raw)
        except Exception as e:
            logger.warning(f"memory_state 解析失败，将重新初始化: {e}")
            return None

    def facts_by_status(self, status: FactStatus) -> list[Fact]:
        return [f for f in self.facts if f.status == status]

    def find_fact(self, fact_id: str) -> Optional[Fact]:
        for f in self.facts:
            if f.fact_id == fact_id:
                return f
        return None

    def mark(self, fact_ids: list[str], status: FactStatus) -> None:
        """批量更新事实状态；置为 disclosed 时记录披露轮次。未知 id 静默忽略"""
        for fid in fact_ids:
            fact = self.find_fact(fid)
            if fact is None:
                continue
            fact.status = status
            if status == "disclosed":
                fact.disclosed_at_turn = self.turn
```

- [ ] **Step 4: 运行测试确认通过**

Run: `cd backend; venv\Scripts\python.exe -m pytest tests\agents\patient\test_memory.py -v`
Expected: 5 passed

- [ ] **Step 5: Commit**

```powershell
git add backend/app/services/agents/patient/__init__.py backend/app/services/agents/patient/memory.py backend/tests/agents/patient/__init__.py backend/tests/agents/patient/test_memory.py
git commit -m "feat(patient-agent): add disclosure ledger memory models"
```

### Task 2: 事实抽取（extract_facts，LLM + 规则兜底）

**Files:**
- Modify: `backend/app/services/agents/patient/memory.py`（文件末尾追加）
- Modify: `backend/app/services/agents/patient/__init__.py`（导出 extract_facts）
- Test: `backend/tests/agents/patient/test_memory.py`（追加）

**Interfaces:**
- Consumes: Task 1 的 `Fact`
- Produces: `async extract_facts(chief_complaint: str, medical_history: str, symptoms_raw: str) -> list[Fact]`；`_rule_based_facts(...) -> list[Fact]`（同参数，同步）

- [ ] **Step 1: 追加失败测试**

在 `test_memory.py` 末尾追加：

```python
from unittest.mock import AsyncMock, patch

from app.services.agents.patient.memory import _rule_based_facts, extract_facts


class TestExtractFacts:
    def test_rule_based_json_list_symptoms(self):
        facts = _rule_based_facts("头痛三天", "无特殊病史", '["头痛", "低热"]')
        contents = [f.content for f in facts]
        assert "头痛三天" in contents and "头痛" in contents and "低热" in contents
        assert all(f.status == "undisclosed" for f in facts)
        # "无特殊病史" 不应产生事实
        assert not [f for f in facts if f.category == "history"]

    def test_rule_based_plain_text_symptoms(self):
        facts = _rule_based_facts("", "十年前胃溃疡，青霉素过敏", "反酸，烧心")
        cats = {f.content: f.category for f in facts}
        assert cats["反酸"] == "symptom" and cats["烧心"] == "symptom"
        assert cats["十年前胃溃疡"] == "history" and cats["青霉素过敏"] == "history"

    @pytest.mark.asyncio
    async def test_extract_facts_llm_success(self):
        llm_out = '{"facts": [{"category": "symptom", "content": "上腹隐痛", "disclosure_condition": "direct_ask"}, {"category": "lifestyle", "content": "长期饮酒", "disclosure_condition": "empathy_unlock"}]}'
        with patch("app.services.agents.patient.memory.call_qwen_chat", new=AsyncMock(return_value=llm_out)):
            facts = await extract_facts("上腹痛", "无", "[]")
        assert [f.fact_id for f in facts] == ["sym_001", "lif_001"]
        assert facts[1].disclosure_condition == "empathy_unlock"

    @pytest.mark.asyncio
    async def test_extract_facts_llm_failure_falls_back(self):
        with patch("app.services.agents.patient.memory.call_qwen_chat", new=AsyncMock(side_effect=RuntimeError("boom"))):
            facts = await extract_facts("头痛三天", "无", '["头痛"]')
        assert [f.content for f in facts] == ["头痛三天", "头痛"]
```

- [ ] **Step 2: 运行测试确认失败**

Run: `cd backend; venv\Scripts\python.exe -m pytest tests\agents\patient\test_memory.py::TestExtractFacts -v`
Expected: FAIL（`ImportError: cannot import name 'extract_facts'`）

- [ ] **Step 3: 实现**

`memory.py` 顶部 import 区追加：

```python
import json

from app.services.qwen_client import call_qwen_chat
from app.utils.json_parser import extract_json_dict_from_text
```

`memory.py` 文件末尾追加：

```python
# ── 事实抽取 ─────────────────────────────────────────────────────────────────

_SPLIT_CHARS = "，,、;；\n"
_EMPTY_MARKERS = ("无", "无特殊病史", "无特殊", "没有")

_CATEGORY_PREFIX = {"symptom": "sym", "history": "his", "medication": "med", "exam": "exm", "lifestyle": "lif"}

_FACT_EXTRACT_SYSTEM = (
    "你是医学病历结构化助手。请把患者档案拆分为原子事实列表，每条事实只含一个信息点。\n"
    "category 取值：symptom(症状)/history(病史)/medication(用药)/exam(检查)/lifestyle(生活史)。\n"
    "disclosure_condition 取值：direct_ask(被直接问到即回答)/"
    "empathy_unlock(敏感隐私信息，需医生共情建立信任后才愿意说)/never_volunteer(绝不主动提)。\n"
    '只输出 JSON：{"facts": [{"category": "...", "content": "...", "disclosure_condition": "..."}]}'
)


def _split_items(text: str) -> list[str]:
    """按中文/英文标点拆分为条目，过滤空值与'无'类占位"""
    items = [text]
    for ch in _SPLIT_CHARS:
        items = [seg for item in items for seg in item.split(ch)]
    return [s.strip() for s in items if s.strip() and s.strip() not in _EMPTY_MARKERS]


def _rule_based_facts(chief_complaint: str, medical_history: str, symptoms_raw: str) -> list[Fact]:
    """规则兜底：主诉一条 + 症状逐项 + 病史逐句"""
    facts: list[Fact] = []
    symptom_items: list[str] = []
    try:
        parsed = json.loads(symptoms_raw or "[]")
        if isinstance(parsed, list):
            symptom_items = [str(x).strip() for x in parsed if str(x).strip()]
        elif isinstance(parsed, dict):
            symptom_items = [f"{k}: {v}" for k, v in parsed.items()]
        else:
            symptom_items = _split_items(str(parsed))
    except (json.JSONDecodeError, TypeError):
        symptom_items = _split_items(symptoms_raw or "")

    if chief_complaint and chief_complaint.strip():
        facts.append(Fact(fact_id="sym_000", category="symptom", content=chief_complaint.strip()))
    for i, item in enumerate(symptom_items, start=1):
        facts.append(Fact(fact_id=f"sym_{i:03d}", category="symptom", content=item))
    for i, item in enumerate(_split_items(medical_history or ""), start=1):
        facts.append(Fact(fact_id=f"his_{i:03d}", category="history", content=item))
    return facts


async def extract_facts(chief_complaint: str, medical_history: str, symptoms_raw: str) -> list[Fact]:
    """患者档案 → 原子事实列表。优先 LLM 结构化抽取，失败降级为规则拆分"""
    profile = f"主诉：{chief_complaint}\n病史：{medical_history}\n症状：{symptoms_raw}"
    try:
        raw = await call_qwen_chat(
            [{"role": "system", "content": _FACT_EXTRACT_SYSTEM},
             {"role": "user", "content": profile}],
            temperature=0.1, max_tokens=1500,
        )
        data = extract_json_dict_from_text(raw)
        facts: list[Fact] = []
        counters: dict[str, int] = {}
        for item in data.get("facts") or []:
            content = str(item.get("content", "")).strip()
            if not content:
                continue
            category = item.get("category", "symptom")
            if category not in _CATEGORY_PREFIX:
                category = "symptom"
            condition = item.get("disclosure_condition", "direct_ask")
            if condition not in ("direct_ask", "empathy_unlock", "never_volunteer"):
                condition = "direct_ask"
            counters[category] = counters.get(category, 0) + 1
            facts.append(Fact(
                fact_id=f"{_CATEGORY_PREFIX[category]}_{counters[category]:03d}",
                category=category, content=content, disclosure_condition=condition,
            ))
        if facts:
            return facts
        logger.warning("LLM 事实抽取结果为空，降级为规则拆分")
    except Exception as e:
        logger.warning(f"LLM 事实抽取失败，降级为规则拆分: {e}")
    return _rule_based_facts(chief_complaint, medical_history, symptoms_raw)
```

`__init__.py` 改为：

```python
# -*- coding: utf-8 -*-
"""患者模拟智能体包 — 披露账本记忆、阶段规划与一致性回复生成"""
from .memory import Fact, MemoryState, extract_facts

__all__ = ["Fact", "MemoryState", "extract_facts"]
```

- [ ] **Step 4: 运行测试确认通过**

Run: `cd backend; venv\Scripts\python.exe -m pytest tests\agents\patient\test_memory.py -v`
Expected: 9 passed

- [ ] **Step 5: Commit**

```powershell
git add backend/app/services/agents/patient/memory.py backend/app/services/agents/patient/__init__.py backend/tests/agents/patient/test_memory.py
git commit -m "feat(patient-agent): add fact extraction with rule-based fallback"
```

### Task 3: 问诊阶段分类器（planner.py）

**Files:**
- Create: `backend/app/services/agents/patient/planner.py`
- Test: `backend/tests/agents/patient/test_planner.py`

**Interfaces:**
- Produces: `STAGES: list[str]`（8 阶段）；`classify_stage(doctor_message: str, current_stage: str) -> str`

- [ ] **Step 1: 写失败测试**

`backend/tests/agents/patient/test_planner.py`：

```python
# -*- coding: utf-8 -*-
"""planner.py 单元测试：问诊阶段关键词分类"""
from app.services.agents.patient.planner import STAGES, classify_stage


class TestClassifyStage:
    def test_stage_list_complete(self):
        assert STAGES == [
            "greeting", "chief_complaint", "hpi", "past_history",
            "personal_family_history", "physical_exam",
            "assessment_communication", "closing",
        ]

    def test_chief_complaint(self):
        assert classify_stage("您好，哪里不舒服？", "greeting") == "chief_complaint"

    def test_hpi(self):
        assert classify_stage("疼了多久了？什么时候开始的？", "chief_complaint") == "hpi"

    def test_past_history(self):
        assert classify_stage("以前得过什么病吗？有没有药物过敏？", "hpi") == "past_history"

    def test_physical_exam(self):
        assert classify_stage("来，量一下体温和血压", "hpi") == "physical_exam"

    def test_no_hit_keeps_current(self):
        assert classify_stage("嗯。", "hpi") == "hpi"

    def test_empty_message_keeps_current(self):
        assert classify_stage("", "greeting") == "greeting"
```

- [ ] **Step 2: 运行测试确认失败**

Run: `cd backend; venv\Scripts\python.exe -m pytest tests\agents\patient\test_planner.py -v`
Expected: FAIL（`ModuleNotFoundError`）

- [ ] **Step 3: 实现**

`backend/app/services/agents/patient/planner.py`：

```python
# -*- coding: utf-8 -*-
"""问诊阶段规划 — 基于关键词规则的医生问题分类与阶段跟踪

阶段状态机只做记录与策略查询，不强制对话走向（约束患者行为而非医生行为）。
纯规则实现，零 LLM 成本、完全确定可测。
"""

STAGES = [
    "greeting", "chief_complaint", "hpi", "past_history",
    "personal_family_history", "physical_exam",
    "assessment_communication", "closing",
]

_STAGE_KEYWORDS: dict[str, tuple[str, ...]] = {
    "greeting": ("请坐", "早上好", "下午好"),
    "chief_complaint": ("哪里不舒服", "哪儿不舒服", "哪不舒服", "怎么了", "什么问题", "为什么来", "看什么"),
    "hpi": ("多久", "多长时间", "什么时候开始", "加重", "缓解", "诱因", "什么样的疼",
            "怎么个疼", "一天几次", "什么性质", "什么部位", "伴随", "还有别的"),
    "past_history": ("以前", "既往", "得过", "手术", "过敏", "住过院", "老毛病", "病史",
                     "吃什么药", "用过什么药", "慢性病"),
    "personal_family_history": ("抽烟", "吸烟", "喝酒", "饮酒", "家里人", "家族", "父母",
                                "职业", "做什么工作", "结婚", "月经"),
    "physical_exam": ("量一下", "测一下", "体温", "血压", "心率", "听诊", "按一下",
                      "压痛", "查体", "张嘴", "看一下舌头"),
    "assessment_communication": ("诊断", "考虑是", "可能是", "建议你", "开点药", "做个检查",
                                 "化验", "拍个片", "初步判断"),
    "closing": ("再见", "注意休息", "按时吃药", "复诊", "有问题再来", "先这样"),
}


def classify_stage(doctor_message: str, current_stage: str) -> str:
    """按关键词命中数将医生消息归入问诊阶段；无命中保持当前阶段"""
    text = (doctor_message or "").strip()
    if not text:
        return current_stage
    best_stage, best_hits = current_stage, 0
    for stage in STAGES:
        hits = sum(1 for kw in _STAGE_KEYWORDS[stage] if kw in text)
        if hits > best_hits:
            best_stage, best_hits = stage, hits
    return best_stage
```

- [ ] **Step 4: 运行测试确认通过**

Run: `cd backend; venv\Scripts\python.exe -m pytest tests\agents\patient\test_planner.py -v`
Expected: 7 passed

- [ ] **Step 5: Commit**

```powershell
git add backend/app/services/agents/patient/planner.py backend/tests/agents/patient/test_planner.py
git commit -m "feat(patient-agent): add consultation stage classifier"
```

### Task 4: 一致性守卫（guard.py — 账本更新与矛盾检测）

**Files:**
- Create: `backend/app/services/agents/patient/guard.py`
- Test: `backend/tests/agents/patient/test_guard.py`

**Interfaces:**
- Consumes: Task 1 的 `MemoryState`、`Fact`
- Produces: `async update_ledger(memory: MemoryState, doctor_message: str, patient_reply: str) -> None`（绝不抛异常）；`check_contradiction(memory: MemoryState, reply: str) -> bool`；`_rule_based_update(memory, patient_reply)`

- [ ] **Step 1: 写失败测试**

`backend/tests/agents/patient/test_guard.py`：

```python
# -*- coding: utf-8 -*-
"""guard.py 单元测试：账本更新（LLM+规则兜底）与矛盾检测"""
from unittest.mock import AsyncMock, patch

import pytest

from app.services.agents.patient.guard import (
    _rule_based_update,
    check_contradiction,
    update_ledger,
)
from app.services.agents.patient.memory import Fact, MemoryState


def _memory():
    return MemoryState(turn=2, facts=[
        Fact(fact_id="sym_001", content="上腹隐痛"),
        Fact(fact_id="sym_002", content="反酸烧心"),
        Fact(fact_id="his_001", category="history", content="青霉素过敏"),
    ])


class TestUpdateLedger:
    @pytest.mark.asyncio
    async def test_llm_marks_disclosed_and_denied(self):
        m = _memory()
        out = '{"disclosed": ["sym_001"], "denied": ["his_001"]}'
        with patch("app.services.agents.patient.guard.call_qwen_chat", new=AsyncMock(return_value=out)):
            await update_ledger(m, "哪里不舒服？有过敏吗？", "肚子上面隐隐地疼。没有过敏。")
        assert m.find_fact("sym_001").status == "disclosed"
        assert m.find_fact("sym_001").disclosed_at_turn == 2
        assert m.find_fact("his_001").status == "denied"
        assert m.find_fact("sym_002").status == "undisclosed"

    @pytest.mark.asyncio
    async def test_llm_failure_falls_back_to_rules(self):
        m = _memory()
        with patch("app.services.agents.patient.guard.call_qwen_chat", new=AsyncMock(side_effect=RuntimeError("boom"))):
            await update_ledger(m, "还有什么症状？", "有点反酸烧心。")  # 不抛异常
        assert m.find_fact("sym_002").status == "disclosed"

    @pytest.mark.asyncio
    async def test_no_pending_facts_skips_llm(self):
        m = _memory()
        for f in m.facts:
            f.status = "disclosed"
        mock_llm = AsyncMock()
        with patch("app.services.agents.patient.guard.call_qwen_chat", new=mock_llm):
            await update_ledger(m, "嗯", "嗯。")
        mock_llm.assert_not_called()


class TestRuleBasedUpdate:
    def test_token_match_marks_disclosed(self):
        m = _memory()
        _rule_based_update(m, "就是反酸烧心，晚上厉害。")
        assert m.find_fact("sym_002").status == "disclosed"
        assert m.find_fact("sym_001").status == "undisclosed"


class TestCheckContradiction:
    def test_denied_fact_reasserted_is_contradiction(self):
        m = _memory()
        m.mark(["his_001"], "denied")
        assert check_contradiction(m, "对，我青霉素过敏。") is True

    def test_denied_fact_with_negation_ok(self):
        m = _memory()
        m.mark(["his_001"], "denied")
        assert check_contradiction(m, "没有，我没有青霉素过敏。") is False

    def test_no_denied_facts_never_contradicts(self):
        m = _memory()
        assert check_contradiction(m, "青霉素过敏。") is False
```

- [ ] **Step 2: 运行测试确认失败**

Run: `cd backend; venv\Scripts\python.exe -m pytest tests\agents\patient\test_guard.py -v`
Expected: FAIL（`ModuleNotFoundError`）

- [ ] **Step 3: 实现**

`backend/app/services/agents/patient/guard.py`：

```python
# -*- coding: utf-8 -*-
"""一致性守卫 — 披露账本更新与矛盾检测

账本更新优先用一次低温 LLM 调用做对话状态跟踪，失败降级为规则匹配；
矛盾检测为纯规则（已否认事实在回复中被再次承认）。
"""
import logging

from app.services.qwen_client import call_qwen_chat
from app.utils.json_parser import extract_json_dict_from_text

from .memory import MemoryState

logger = logging.getLogger(__name__)

_NEGATION_WORDS = ("没有", "没", "不", "无", "从来没", "记不清", "不清楚")

_LEDGER_SYSTEM = (
    "你是医患对话状态跟踪器。给定患者档案事实清单与本轮对话，判断：\n"
    "1. disclosed：患者本轮回复中披露（承认/描述）了哪些事实\n"
    "2. denied：患者本轮回复中明确否认了哪些事实\n"
    '只输出 JSON：{"disclosed": ["fact_id"], "denied": ["fact_id"]}，没有则给空数组。'
)


def _split_tokens(text: str) -> list[str]:
    tokens = [text]
    for ch in "，,、;；。 ：:":
        tokens = [seg for tk in tokens for seg in tk.split(ch)]
    return [t.strip() for t in tokens if t.strip()]


async def update_ledger(memory: MemoryState, doctor_message: str, patient_reply: str) -> None:
    """回复后更新披露账本。LLM 判定失败时降级为规则匹配，绝不向上抛异常"""
    pending = memory.facts_by_status("undisclosed")
    if not pending:
        return
    fact_lines = "\n".join(f"{f.fact_id}: {f.content}" for f in pending)
    try:
        raw = await call_qwen_chat(
            [{"role": "system", "content": _LEDGER_SYSTEM},
             {"role": "user", "content": (
                 f"事实清单：\n{fact_lines}\n\n医生：{doctor_message}\n患者：{patient_reply}"
             )}],
            temperature=0.0, max_tokens=200,
        )
        data = extract_json_dict_from_text(raw)
        disclosed = [x for x in data.get("disclosed", []) if isinstance(x, str)]
        denied = [x for x in data.get("denied", []) if isinstance(x, str)]
        memory.mark(disclosed, "disclosed")
        memory.mark(denied, "denied")
    except Exception as e:
        logger.warning(f"账本 LLM 更新失败，降级为规则匹配: {e}")
        _rule_based_update(memory, patient_reply)


def _rule_based_update(memory: MemoryState, patient_reply: str) -> None:
    """规则兜底：事实内容的关键片段（≥2字）出现在回复中即视为已披露"""
    reply = patient_reply or ""
    for fact in memory.facts:
        if fact.status != "undisclosed":
            continue
        tokens = [t for t in _split_tokens(fact.content) if len(t) >= 2]
        if tokens and any(t in reply for t in tokens):
            fact.status = "disclosed"
            fact.disclosed_at_turn = memory.turn


def check_contradiction(memory: MemoryState, reply: str) -> bool:
    """检测回复是否承认了已否认（denied）的事实——即前后矛盾"""
    text = reply or ""
    for fact in memory.facts_by_status("denied"):
        tokens = [t for t in _split_tokens(fact.content) if len(t) >= 2]
        if not tokens:
            continue
        if any(t in text for t in tokens) and not any(neg in text for neg in _NEGATION_WORDS):
            return True
    return False
```

- [ ] **Step 4: 运行测试确认通过**

Run: `cd backend; venv\Scripts\python.exe -m pytest tests\agents\patient\test_guard.py -v`
Expected: 7 passed

- [ ] **Step 5: Commit**

```powershell
git add backend/app/services/agents/patient/guard.py backend/tests/agents/patient/test_guard.py
git commit -m "feat(patient-agent): add ledger update and contradiction guard"
```

### Task 5: PatientAgent 编排（prompts.py + agent.py）

**Files:**
- Create: `backend/app/services/agents/patient/prompts.py`
- Create: `backend/app/services/agents/patient/agent.py`
- Modify: `backend/app/services/consultation_service.py`（常量改为 import）
- Modify: `backend/app/services/agents/patient/__init__.py`（导出 PatientAgent）
- Test: `backend/tests/agents/patient/test_agent.py`

**Interfaces:**
- Consumes: Task 1-4 的 `MemoryState`、`classify_stage`、`update_ledger`、`check_contradiction`
- Produces: `PATIENT_ROLE_WRAPPER`（迁至 prompts.py，含 `{system_prompt}` 占位符）；`PatientAgent(patient, memory)`，方法 `_build_system_prompt() -> str`、`async respond(doctor_message: str, chat_history: list[dict]) -> str`（内部失败向上抛，由调用方回退）

- [ ] **Step 1: 写失败测试**

`backend/tests/agents/patient/test_agent.py`：

```python
# -*- coding: utf-8 -*-
"""agent.py 单元测试：PatientAgent 编排（LLM 全 mock）"""
from types import SimpleNamespace
from unittest.mock import AsyncMock, patch

import pytest

from app.services.agents.patient.agent import PatientAgent
from app.services.agents.patient.memory import Fact, MemoryState
from app.services.agents.patient.prompts import PATIENT_ROLE_WRAPPER


def _agent():
    memory = MemoryState(facts=[
        Fact(fact_id="sym_001", content="上腹隐痛", status="disclosed"),
        Fact(fact_id="his_001", category="history", content="青霉素过敏", status="denied"),
        Fact(fact_id="sym_002", content="反酸烧心"),
    ])
    patient = SimpleNamespace(system_prompt="45岁男性，上腹痛两周", personality_type="配合型")
    return PatientAgent(patient, memory)


def test_wrapper_migrated_with_placeholder():
    assert "{system_prompt}" in PATIENT_ROLE_WRAPPER
    # 向后兼容：consultation_service 仍可导入同名常量
    from app.services import consultation_service
    assert consultation_service.PATIENT_ROLE_WRAPPER is PATIENT_ROLE_WRAPPER


class TestBuildSystemPrompt:
    def test_contains_ledger_sections(self):
        prompt = _agent()._build_system_prompt()
        assert "45岁男性，上腹痛两周" in prompt
        assert "你已经告诉过医生的信息" in prompt and "上腹隐痛" in prompt
        assert "绝对不能再承认" in prompt and "青霉素过敏" in prompt

    def test_empty_ledger_no_sections(self):
        agent = _agent()
        agent.memory = MemoryState()
        prompt = agent._build_system_prompt()
        assert "你已经告诉过医生的信息" not in prompt
        assert "绝对不能再承认" not in prompt


class TestRespond:
    @pytest.mark.asyncio
    async def test_normal_flow_updates_memory(self):
        agent = _agent()
        with patch("app.services.agents.patient.agent.call_qwen_chat", new=AsyncMock(return_value="有点反酸烧心。")) as llm, \
             patch("app.services.agents.patient.agent.update_ledger", new=AsyncMock()) as ledger:
            reply = await agent.respond("还有什么症状？", [])
        assert reply == "有点反酸烧心。"
        assert agent.memory.turn == 1
        assert agent.memory.stage_history == [agent.memory.stage]
        llm.assert_awaited_once()
        ledger.assert_awaited_once()

    @pytest.mark.asyncio
    async def test_stage_transition(self):
        agent = _agent()
        with patch("app.services.agents.patient.agent.call_qwen_chat", new=AsyncMock(return_value="肚子上面疼。")), \
             patch("app.services.agents.patient.agent.update_ledger", new=AsyncMock()):
            await agent.respond("您好，哪里不舒服？", [])
        assert agent.memory.stage == "chief_complaint"

    @pytest.mark.asyncio
    async def test_contradiction_triggers_regeneration(self):
        agent = _agent()
        llm = AsyncMock(side_effect=["对，我青霉素过敏。", "没有，我没有过敏。"])
        with patch("app.services.agents.patient.agent.call_qwen_chat", new=llm), \
             patch("app.services.agents.patient.agent.update_ledger", new=AsyncMock()):
            reply = await agent.respond("你有青霉素过敏吗？", [])
        assert llm.await_count == 2
        assert reply == "没有，我没有过敏。"
```

- [ ] **Step 2: 运行测试确认失败**

Run: `cd backend; venv\Scripts\python.exe -m pytest tests\agents\patient\test_agent.py -v`
Expected: FAIL（`ModuleNotFoundError: app.services.agents.patient.prompts`）

- [ ] **Step 3: 实现**

(a) 创建 `backend/app/services/agents/patient/prompts.py`：文件头为

```python
# -*- coding: utf-8 -*-
"""患者角色扮演提示词 — 从 consultation_service 迁入，供 PatientAgent 与回退路径共用"""
```

然后将 `backend/app/services/consultation_service.py` 第 21-85 行（`# 虚拟患者角色扮演约束…` 注释 + 完整 `PATIENT_ROLE_WRAPPER = """…{system_prompt}"""` 常量）**逐字剪切**到该文件（不做任何文字改动）。

(b) 在 `consultation_service.py` 原位置删除常量，并在 import 区（`from app.services.qwen_client import call_qwen_chat` 之后）加入：

```python
from app.services.agents.patient.prompts import PATIENT_ROLE_WRAPPER
```

（模块内所有 `PATIENT_ROLE_WRAPPER` 用法不变，旧测试无感知。）

(c) 创建 `backend/app/services/agents/patient/agent.py`：

```python
# -*- coding: utf-8 -*-
"""患者智能体编排 — 账本注入、阶段跟踪、矛盾重生成

每次请求新建实例，全部会话状态存于传入的 MemoryState（由调用方持久化）。
respond 内部失败会向上抛出，由 consultation_service 回退旧无记忆路径。
"""
import logging

from app.services.qwen_client import call_qwen_chat

from .guard import check_contradiction, update_ledger
from .memory import MemoryState
from .planner import classify_stage
from .prompts import PATIENT_ROLE_WRAPPER

logger = logging.getLogger(__name__)

_REGEN_INSTRUCTION = (
    "注意：你上一版回复与你此前已否认的信息矛盾。"
    "请重新回答医生的问题，绝对不能承认下面列出的【已否认信息】。"
)


class PatientAgent:
    """基于披露账本的患者回复生成器"""

    def __init__(self, patient, memory: MemoryState):
        self.patient = patient
        self.memory = memory

    def _build_system_prompt(self) -> str:
        """角色包装 + 披露账本注入（已披露保持一致 / 已否认绝不翻供）"""
        sections = [PATIENT_ROLE_WRAPPER.format(system_prompt=self.patient.system_prompt or "")]
        disclosed = self.memory.facts_by_status("disclosed")
        if disclosed:
            lines = "\n".join(f"- {f.content}" for f in disclosed)
            sections.append(
                "【你已经告诉过医生的信息】（再被问到时保持说法一致，不要当作新信息重复展开）\n" + lines
            )
        denied = self.memory.facts_by_status("denied")
        if denied:
            lines = "\n".join(f"- {f.content}" for f in denied)
            sections.append("【你已明确否认过的信息（绝对不能再承认）】\n" + lines)
        return "\n\n".join(sections)

    async def respond(self, doctor_message: str, chat_history: list[dict]) -> str:
        """生成一条患者回复并更新记忆状态"""
        self.memory.turn += 1
        stage = classify_stage(doctor_message, self.memory.stage)
        self.memory.stage = stage
        self.memory.stage_history.append(stage)

        messages = [{"role": "system", "content": self._build_system_prompt()}]
        messages.extend(chat_history)
        messages.append({"role": "user", "content": doctor_message})

        reply = await call_qwen_chat(messages, temperature=0.3)

        if check_contradiction(self.memory, reply):
            logger.warning("患者回复与已否认事实矛盾，触发一次重生成")
            denied_lines = "\n".join(f"- {f.content}" for f in self.memory.facts_by_status("denied"))
            retry_messages = messages + [
                {"role": "assistant", "content": reply},
                {"role": "user", "content": (
                    f"{_REGEN_INSTRUCTION}\n【已否认信息】\n{denied_lines}\n\n"
                    f"医生刚才的问题是：{doctor_message}"
                )},
            ]
            regenerated = await call_qwen_chat(retry_messages, temperature=0.2)
            if not check_contradiction(self.memory, regenerated):
                reply = regenerated

        await update_ledger(self.memory, doctor_message, reply)
        return reply
```

(d) `__init__.py` 改为：

```python
# -*- coding: utf-8 -*-
"""患者模拟智能体包 — 披露账本记忆、阶段规划与一致性回复生成"""
from .agent import PatientAgent
from .memory import Fact, MemoryState, extract_facts

__all__ = ["Fact", "MemoryState", "PatientAgent", "extract_facts"]
```

- [ ] **Step 4: 运行测试确认通过**

Run: `cd backend; venv\Scripts\python.exe -m pytest tests\agents\patient\test_agent.py tests\services\test_consultation_service.py -v`
Expected: test_agent 6 passed；consultation_service 旧用例全部保持通过

- [ ] **Step 5: Commit**

```powershell
git add backend/app/services/agents/patient/prompts.py backend/app/services/agents/patient/agent.py backend/app/services/agents/patient/__init__.py backend/app/services/consultation_service.py backend/tests/agents/patient/test_agent.py
git commit -m "feat(patient-agent): add PatientAgent orchestration with ledger-aware prompts"
```

---

## P2 持久化与主流程接入

### Task 6: memory_state 列 + 迁移脚本

**Files:**
- Modify: `backend/app/models/consultation.py`
- Create: `backend/scripts/migrate_patient_memory.py`

**Interfaces:**
- Produces: `Consultation.memory_state: Optional[str]`（Text 列，nullable，JSON 序列化的 MemoryState）

- [ ] **Step 1: 修改模型**

`backend/app/models/consultation.py` 在 `max_rounds` 字段（40-42 行）之后、`created_at` 之前插入：

```python
    memory_state: Mapped[Optional[str]] = mapped_column(
        Text, nullable=True, comment="患者智能体记忆状态（JSON 序列化的 MemoryState）"
    )
```

（`Optional` 与 `Text` 已在文件头部导入，无需新增 import。）

- [ ] **Step 2: 写迁移脚本**

`backend/scripts/migrate_patient_memory.py`（仿 `migrate_personality_type.py` 范式，SHOW COLUMNS 幂等）：

```python
# -*- coding: utf-8 -*-
"""数据库迁移脚本：为 consultations 表增加 memory_state 列（患者智能体记忆）。

幂等：列已存在时直接跳过。
"""
import asyncio
import sys
from pathlib import Path

# 将 backend 目录加入路径以便导入 app
sys.path.insert(0, str(Path(__file__).parent.parent))

from sqlalchemy import text  # noqa: E402

from app.db.session import AsyncSessionLocal  # noqa: E402


async def migrate():
    async with AsyncSessionLocal() as db:
        r = await db.execute(text("SHOW COLUMNS FROM consultations LIKE 'memory_state'"))
        if r.fetchone():
            print("memory_state 列已存在，跳过迁移。")
            return
        print("[1/1] ALTER TABLE consultations ADD COLUMN memory_state ...")
        await db.execute(text(
            "ALTER TABLE consultations ADD COLUMN memory_state TEXT NULL "
            "COMMENT '患者智能体记忆状态（JSON 序列化的 MemoryState）' AFTER max_rounds"
        ))
        await db.commit()
        print("      Done.")

        # 验证
        r2 = await db.execute(text("SHOW COLUMNS FROM consultations LIKE 'memory_state'"))
        col = r2.fetchone()
        print(f"\n=== Verification ===\nColumn def: {col[1] if col else 'MISSING'}")

    print("\nMigration completed successfully!")


asyncio.run(migrate())
```

- [ ] **Step 3: 执行迁移并验证**

Run: `cd backend; venv\Scripts\python.exe scripts\migrate_patient_memory.py`
Expected: 输出 `Migration completed successfully!` 且 Column def 为 `text`；重复执行输出“已存在，跳过”

- [ ] **Step 4: 回归现有测试**

Run: `cd backend; venv\Scripts\python.exe -m pytest tests\services\test_consultation_service.py -v`
Expected: 全部通过（新列 nullable，旧逻辑无感知）

- [ ] **Step 5: Commit**

```powershell
git add backend/app/models/consultation.py backend/scripts/migrate_patient_memory.py
git commit -m "feat(patient-agent): add consultations.memory_state column with migration"
```

### Task 7: consultation_service 接入 PatientAgent（异常回退旧路径）

**Files:**
- Modify: `backend/app/services/consultation_service.py`
- Test: `backend/tests/services/test_consultation_service.py`（追加用例）

**Interfaces:**
- Consumes: `PatientAgent`、`MemoryState`、`extract_facts`
- Produces: `async _build_history(messages, patient_prompt: str) -> list[dict]`（提取的共享滑窗+摘要逻辑，返回不含角色包装 system 头的消息列表）；`async _generate_patient_reply(consultation, patient, messages, content) -> str`（账本路径，异常时回退 `_legacy_generate_patient_reply`）

- [ ] **Step 1: 写失败测试**

在 `backend/tests/services/test_consultation_service.py` 末尾追加（复用既有 fixtures `mock_db`/`sample_consultation`/`sample_patient`/`sample_messages`）：

```python
class TestGeneratePatientReplyWithAgent:
    """_generate_patient_reply：账本路径 + 异常回退"""

    @pytest.mark.asyncio
    async def test_agent_path_persists_memory(self, sample_consultation, sample_patient, sample_messages):
        sample_consultation.memory_state = None
        mock_facts = [Fact(fact_id="sym_001", content="上腹隐痛")]
        with patch("app.services.consultation_service.extract_facts", new=AsyncMock(return_value=mock_facts)), \
             patch.object(consultation_service.PatientAgent, "respond", new=AsyncMock(return_value="肚子上面疼。")):
            reply = await consultation_service._generate_patient_reply(
                sample_consultation, sample_patient, sample_messages, "哪里不舒服？"
            )
        assert reply == "肚子上面疼。"
        restored = MemoryState.from_json(sample_consultation.memory_state)
        assert restored is not None
        assert [f.fact_id for f in restored.facts] == ["sym_001"]

    @pytest.mark.asyncio
    async def test_agent_failure_falls_back_to_legacy(self, sample_consultation, sample_patient, sample_messages):
        sample_consultation.memory_state = None
        with patch("app.services.consultation_service.extract_facts", new=AsyncMock(side_effect=RuntimeError("boom"))), \
             patch("app.services.consultation_service.call_qwen_chat", new=AsyncMock(return_value="旧路径回复")):
            reply = await consultation_service._generate_patient_reply(
                sample_consultation, sample_patient, sample_messages, "哪里不舒服？"
            )
        assert reply == "旧路径回复"

    @pytest.mark.asyncio
    async def test_existing_memory_reused_not_reextracted(self, sample_consultation, sample_patient, sample_messages):
        existing = MemoryState(facts=[Fact(fact_id="his_001", category="history", content="胃溃疡")], turn=4)
        sample_consultation.memory_state = existing.to_json()
        extract_mock = AsyncMock()
        with patch("app.services.consultation_service.extract_facts", new=extract_mock), \
             patch.object(consultation_service.PatientAgent, "respond", new=AsyncMock(return_value="嗯。")):
            await consultation_service._generate_patient_reply(
                sample_consultation, sample_patient, sample_messages, "嗯"
            )
        extract_mock.assert_not_called()
```

同时在该测试文件头部 import 区追加：

```python
from app.services.agents.patient.memory import Fact, MemoryState
```

（`patch`/`AsyncMock`/`consultation_service` 已在既有 import 中，若缺则补。）

- [ ] **Step 2: 运行测试确认失败**

Run: `cd backend; venv\Scripts\python.exe -m pytest tests\services\test_consultation_service.py::TestGeneratePatientReplyWithAgent -v`
Expected: FAIL（`AttributeError: no attribute '_generate_patient_reply'`）

- [ ] **Step 3: 实现**

(a) `consultation_service.py` import 区追加：

```python
from app.services.agents.patient import MemoryState, PatientAgent, extract_facts
```

(b) 在 `_summarize_early_messages` 之后新增三个函数：

```python
async def _build_history(messages: List[ConsultationMessage], patient_prompt: str) -> List[Dict[str, str]]:
    """滑动窗口 + 早期摘要，返回不含角色包装头的对话历史（两条生成路径共享）"""
    history: List[Dict[str, str]] = []
    recent_window = MEMORY_RECENT_TURNS * 2  # 每轮 2 条消息
    compress_threshold = MEMORY_COMPRESS_THRESHOLD * 2

    if len(messages) > compress_threshold:
        early_messages = messages[:-recent_window]
        recent_messages = messages[-recent_window:]
        summary = await _summarize_early_messages(early_messages, patient_prompt)
        if summary:
            history.append({
                "role": "system",
                "content": f"《早期问诊记录摘要》（口述展示的症状和对话要点，请保持与此一致）\n{summary}",
            })
    else:
        recent_messages = messages[-recent_window:] if len(messages) > recent_window else messages

    for msg in recent_messages:
        history.append({
            "role": "user" if msg.role == "doctor" else "assistant",
            "content": msg.content,
        })
    return history


async def _legacy_generate_patient_reply(
    patient: VirtualPatient, messages: List[ConsultationMessage], content: str
) -> str:
    """无记忆旧路径（回退兼容）：角色包装 + 滑窗历史直接调 LLM"""
    wrapped_prompt = PATIENT_ROLE_WRAPPER.format(system_prompt=patient.system_prompt or "")
    chat_history = [{"role": "system", "content": wrapped_prompt}]
    chat_history.extend(await _build_history(messages, patient.system_prompt or ""))
    chat_history.append({"role": "user", "content": content})
    return await call_qwen_chat(chat_history, temperature=0.3)


async def _generate_patient_reply(
    consultation: Consultation,
    patient: VirtualPatient,
    messages: List[ConsultationMessage],
    content: str,
) -> str:
    """患者回复生成主入口：优先走披露账本智能体，任意异常回退旧路径，绝不中断问诊"""
    try:
        memory = MemoryState.from_json(consultation.memory_state)
        if memory is None:
            facts = await extract_facts(
                patient.chief_complaint or "",
                patient.medical_history or "",
                patient.symptoms or "",
            )
            memory = MemoryState(facts=facts)
        agent = PatientAgent(patient, memory)
        history = await _build_history(messages, patient.system_prompt or "")
        reply = await agent.respond(content, history)
        consultation.memory_state = memory.to_json()  # 调用方统一 commit
        return reply
    except Exception as e:
        logger.warning(f"患者智能体路径失败，回退无记忆旧路径: {e}", exc_info=True)
        return await _legacy_generate_patient_reply(patient, messages, content)
```

(c) `send_doctor_message` 中将“构建上下文 + 调 LLM”段（原 269-295 行，从 `wrapped_prompt = PATIENT_ROLE_WRAPPER.format(...)` 到 `patient_reply = await call_qwen_chat(chat_history, temperature=0.3)`）整段替换为：

```python
    patient_reply = await _generate_patient_reply(consultation, patient, messages, content)
```

(d) `send_doctor_message_stream` 中 Step 4-6 的同段逻辑（原 366-403 行，从 `wrapped_prompt = ...` 到 `patient_reply = await call_qwen_chat(...)`，**保留** `building_context`/`generating_reply` 两个 progress 事件，删除 `compressing_memory` 事件块）替换为：

```python
        yield _make_sse_event("progress", {
            "step": "generating_reply",
            "message": "患者正在思考回复...",
            "progress": 60,
        })
        patient_reply = await _generate_patient_reply(consultation, patient, messages, content)
```

- [ ] **Step 4: 运行测试确认通过**

Run: `cd backend; venv\Scripts\python.exe -m pytest tests\services\test_consultation_service.py -v`
Expected: 新增 3 条通过；旧用例若因生成段重构断言失效，按“mock `_generate_patient_reply` 或 `call_qwen_chat` 后断言消息落库”的语义最小修复，不弱化断言

- [ ] **Step 5: Commit**

```powershell
git add backend/app/services/consultation_service.py backend/tests/services/test_consultation_service.py
git commit -m "feat(patient-agent): route patient reply generation through PatientAgent with legacy fallback"
```

---

## P3 患者专属工具

### Task 8: QueryPlausibleSymptom（RAG 裁决档案外症状）

**Files:**
- Create: `backend/app/services/tools/patient/__init__.py`（本 Task 先建空包，Task 11 补注册函数）
- Create: `backend/app/services/tools/patient/plausible_symptom.py`
- Create: `backend/tests/tools/test_patient_tools.py`

**Interfaces:**
- Consumes: `BaseTool`/`ToolContext`（tools.base）、`tiered_retrieve(queries, top_k_per_query)`、`RetrievalQuery(query_type="diagnosis", text=..., source="clinical_facts")`、`call_qwen_chat`、`extract_json_dict_from_text`
- Produces: `QueryPlausibleSymptom`（name=`query_plausible_symptom`），`execute` 返回 `{"verdict": "present|absent|uncertain", "reason": str, "degraded": bool}`；任何异常降级为 `{"verdict": "uncertain", "degraded": True}`

- [ ] **Step 1: 写失败测试**

`backend/tests/tools/test_patient_tools.py`：

```python
# -*- coding: utf-8 -*-
"""患者专属工具单元测试（RAG/LLM 全 mock）"""
from types import SimpleNamespace
from unittest.mock import AsyncMock, patch

import pytest

from app.services.tools.base import ToolContext
from app.services.tools.patient.plausible_symptom import (
    QueryPlausibleSymptom,
    QueryPlausibleSymptomArgs,
)


def _context():
    return ToolContext(run_id="t-1", agent_name="patient_agent")


def _bundle(texts):
    items = [SimpleNamespace(text=t, source="内科学", heading_path="消化系统", rrf_score=0.9) for t in texts]
    return SimpleNamespace(candidates=items, level_used="base", trace=None)


class TestQueryPlausibleSymptom:
    @pytest.mark.asyncio
    async def test_present_verdict(self):
        tool = QueryPlausibleSymptom()
        args = QueryPlausibleSymptomArgs(symptom="夜间痛醒", diagnosis="十二指肠溃疡")
        with patch("app.services.tools.patient.plausible_symptom.tiered_retrieve", new=AsyncMock(return_value=_bundle(["十二指肠溃疡典型表现为夜间痛、饥饿痛"]))), \
             patch("app.services.tools.patient.plausible_symptom.call_qwen_chat", new=AsyncMock(return_value='{"verdict": "present", "reason": "夜间痛是典型伴随症状"}')):
            result = await tool.execute(args, _context())
        assert result["verdict"] == "present"
        assert result["degraded"] is False

    @pytest.mark.asyncio
    async def test_invalid_verdict_coerced_to_uncertain(self):
        tool = QueryPlausibleSymptom()
        args = QueryPlausibleSymptomArgs(symptom="头疼", diagnosis="胃溃疡")
        with patch("app.services.tools.patient.plausible_symptom.tiered_retrieve", new=AsyncMock(return_value=_bundle(["…"]))), \
             patch("app.services.tools.patient.plausible_symptom.call_qwen_chat", new=AsyncMock(return_value='{"verdict": "maybe", "reason": "?"}')):
            result = await tool.execute(args, _context())
        assert result["verdict"] == "uncertain"

    @pytest.mark.asyncio
    async def test_retrieval_failure_degrades(self):
        tool = QueryPlausibleSymptom()
        args = QueryPlausibleSymptomArgs(symptom="头疼", diagnosis="胃溃疡")
        with patch("app.services.tools.patient.plausible_symptom.tiered_retrieve", new=AsyncMock(side_effect=RuntimeError("boom"))):
            result = await tool.execute(args, _context())
        assert result == {"verdict": "uncertain", "reason": "知识库裁决失败，保守处理", "degraded": True}
```

同时确保 `backend/tests/tools/__init__.py` 存在（若无则创建，内容一行 `# -*- coding: utf-8 -*-`）。

- [ ] **Step 2: 运行测试确认失败**

Run: `cd backend; venv\Scripts\python.exe -m pytest tests\tools\test_patient_tools.py -v`
Expected: FAIL（`ModuleNotFoundError: app.services.tools.patient`）

- [ ] **Step 3: 实现**

`backend/app/services/tools/patient/__init__.py`（暂时仅包声明，Task 11 扩充）：

```python
# -*- coding: utf-8 -*-
"""患者智能体专属工具包"""
```

`backend/app/services/tools/patient/plausible_symptom.py`：

```python
# -*- coding: utf-8 -*-
"""档案外症状裁决工具 — 医生问到档案未写明的症状时，用 RAG 检索该诊断的
典型临床表现，由低温 LLM 裁决该症状应否存在，避免患者乱编或一律否认。
任何异常降级为 uncertain（患者回答“记不清/不确定”）。
"""
import logging

from pydantic import BaseModel, Field

from app.services.qwen_client import call_qwen_chat
from app.services.rag.retriever import tiered_retrieve
from app.services.rag.types import RetrievalQuery
from app.services.tools.base import BaseTool, ToolContext
from app.utils.json_parser import extract_json_dict_from_text

logger = logging.getLogger(__name__)

_VALID_VERDICTS = ("present", "absent", "uncertain")

_VERDICT_SYSTEM = (
    "你是临床医学知识裁决助手。根据检索到的医学证据，判断某症状在给定诊断下"
    "是否为合理伴随症状。present=典型/常见伴随；absent=医学上不相符；"
    "uncertain=证据不足。只输出 JSON："
    '{"verdict": "present|absent|uncertain", "reason": "一句话理由"}'
)


class QueryPlausibleSymptomArgs(BaseModel):
    symptom: str = Field(description="医生问到的、患者档案未写明的症状")
    diagnosis: str = Field(description="患者的预期诊断或主要病情")


class QueryPlausibleSymptom(BaseTool):
    name = "query_plausible_symptom"
    description = "裁决档案外症状在该诊断下是否合理存在（present/absent/uncertain）"
    args_schema = QueryPlausibleSymptomArgs
    timeout_seconds = 30
    critical = False

    async def execute(self, args: QueryPlausibleSymptomArgs, context: ToolContext) -> dict:
        try:
            query = RetrievalQuery(
                query_type="diagnosis",
                text=f"{args.diagnosis} 典型症状 临床表现 {args.symptom}",
                source="clinical_facts",
            )
            bundle = await tiered_retrieve(queries=[query], top_k_per_query=3)
            evidence_text = "\n".join(
                f"[{item.source}] {item.text[:300]}" for item in (bundle.candidates or [])[:3]
            ) or "（未检索到相关证据）"

            raw = await call_qwen_chat(
                [{"role": "system", "content": _VERDICT_SYSTEM},
                 {"role": "user", "content": (
                     f"诊断：{args.diagnosis}\n待裁决症状：{args.symptom}\n\n医学证据：\n{evidence_text}"
                 )}],
                temperature=0.1, max_tokens=300,
            )
            data = extract_json_dict_from_text(raw)
            verdict = data.get("verdict", "uncertain")
            if verdict not in _VALID_VERDICTS:
                verdict = "uncertain"
            return {"verdict": verdict, "reason": str(data.get("reason", "")), "degraded": False}
        except Exception as e:
            logger.warning(f"query_plausible_symptom 裁决失败，降级 uncertain: {e}")
            return {"verdict": "uncertain", "reason": "知识库裁决失败，保守处理", "degraded": True}
```

- [ ] **Step 4: 运行测试确认通过**

Run: `cd backend; venv\Scripts\python.exe -m pytest tests\tools\test_patient_tools.py -v`
Expected: 3 passed

- [ ] **Step 5: Commit**

```powershell
git add backend/app/services/tools/patient/ backend/tests/tools/test_patient_tools.py
git commit -m "feat(patient-tools): add query_plausible_symptom RAG verdict tool"
```

### Task 9: PhysiologyCalculator（确定性生命体征）

**Files:**
- Create: `backend/app/services/tools/patient/physiology.py`
- Test: `backend/tests/tools/test_patient_tools.py`（追加）

**Interfaces:**
- Produces: `PhysiologyCalculator`（name=`physiology_calculator`），`execute(args={vital, consultation_id, abnormal}) -> {"vital": str, "value": str, "unit": str}`；同一 `consultation_id+vital+abnormal` 结果确定不变（seed 固定）

- [ ] **Step 1: 追加失败测试**

在 `test_patient_tools.py` 末尾追加：

```python
from app.services.tools.patient.physiology import (
    PhysiologyCalculator,
    PhysiologyCalculatorArgs,
)


class TestPhysiologyCalculator:
    @pytest.mark.asyncio
    async def test_deterministic_same_seed(self):
        tool = PhysiologyCalculator()
        args = PhysiologyCalculatorArgs(vital="body_temperature", consultation_id=42, abnormal=False)
        r1 = await tool.execute(args, _context())
        r2 = await tool.execute(args, _context())
        assert r1 == r2
        assert r1["unit"] == "℃"
        assert 36.0 <= float(r1["value"]) <= 37.2

    @pytest.mark.asyncio
    async def test_abnormal_range_differs(self):
        tool = PhysiologyCalculator()
        normal = await tool.execute(PhysiologyCalculatorArgs(vital="body_temperature", consultation_id=42, abnormal=False), _context())
        fever = await tool.execute(PhysiologyCalculatorArgs(vital="body_temperature", consultation_id=42, abnormal=True), _context())
        assert float(fever["value"]) >= 37.8
        assert float(fever["value"]) != float(normal["value"])

    @pytest.mark.asyncio
    async def test_blood_pressure_format(self):
        tool = PhysiologyCalculator()
        r = await tool.execute(PhysiologyCalculatorArgs(vital="blood_pressure", consultation_id=7, abnormal=False), _context())
        assert "/" in r["value"] and r["unit"] == "mmHg"

    @pytest.mark.asyncio
    async def test_unknown_vital_rejected(self):
        tool = PhysiologyCalculator()
        r = await tool.execute(PhysiologyCalculatorArgs(vital="unknown_thing", consultation_id=1, abnormal=False), _context())
        assert r.get("error")
```

- [ ] **Step 2: 运行测试确认失败**

Run: `cd backend; venv\Scripts\python.exe -m pytest tests\tools\test_patient_tools.py::TestPhysiologyCalculator -v`
Expected: FAIL（`ModuleNotFoundError`）

- [ ] **Step 3: 实现**

`backend/app/services/tools/patient/physiology.py`：

```python
# -*- coding: utf-8 -*-
"""生理指标计算器 — 确定性生命体征生成

同一会话内同一指标多次测量结果稳定（seed = consultation_id:vital:abnormal），
避免 LLM 自由发挥导致“上次 38.5 这次 36.2”的不一致。纯本地计算，零 LLM 成本。
"""
import random

from pydantic import BaseModel, Field

from app.services.tools.base import BaseTool, ToolContext

# 正常/异常取值范围：(下限, 上限, 小数位数, 单位)
_BASELINES = {
    "body_temperature": (36.2, 37.0, 1, "℃"),
    "heart_rate": (62, 95, 0, "次/分"),
    "respiratory_rate": (14, 19, 0, "次/分"),
    "blood_pressure": ((105, 130), (65, 85), 0, "mmHg"),  # (收缩压范围, 舒张压范围)
}
_ABNORMAL = {
    "body_temperature": (37.8, 39.5, 1, "℃"),
    "heart_rate": (102, 130, 0, "次/分"),
    "respiratory_rate": (22, 30, 0, "次/分"),
    "blood_pressure": ((145, 175), (92, 110), 0, "mmHg"),
}


class PhysiologyCalculatorArgs(BaseModel):
    vital: str = Field(description="指标名: body_temperature/heart_rate/respiratory_rate/blood_pressure")
    consultation_id: int = Field(description="会话 ID（确定性种子）")
    abnormal: bool = Field(default=False, description="是否按异常（病情相关）范围生成")


class PhysiologyCalculator(BaseTool):
    name = "physiology_calculator"
    description = "按会话确定性生成生命体征数值（体温/心率/呼吸/血压）"
    args_schema = PhysiologyCalculatorArgs
    timeout_seconds = 5
    critical = False

    async def execute(self, args: PhysiologyCalculatorArgs, context: ToolContext) -> dict:
        table = _ABNORMAL if args.abnormal else _BASELINES
        if args.vital not in table:
            return {"error": f"未知指标: {args.vital}", "vital": args.vital}
        rng = random.Random(f"{args.consultation_id}:{args.vital}:{args.abnormal}")
        spec = table[args.vital]
        if args.vital == "blood_pressure":
            (sys_lo, sys_hi), (dia_lo, dia_hi), _, unit = spec
            value = f"{rng.randint(sys_lo, sys_hi)}/{rng.randint(dia_lo, dia_hi)}"
        else:
            lo, hi, digits, unit = spec
            value = str(round(rng.uniform(lo, hi), digits) if digits else rng.randint(int(lo), int(hi)))
        return {"vital": args.vital, "value": value, "unit": unit}
```

- [ ] **Step 4: 运行测试确认通过**

Run: `cd backend; venv\Scripts\python.exe -m pytest tests\tools\test_patient_tools.py -v`
Expected: 7 passed

- [ ] **Step 5: Commit**

```powershell
git add backend/app/services/tools/patient/physiology.py backend/tests/tools/test_patient_tools.py
git commit -m "feat(patient-tools): add deterministic physiology calculator"
```

### Task 10: EmotionEngine（行为分类 + 情绪转移）

**Files:**
- Create: `backend/app/services/tools/patient/emotion.py`
- Test: `backend/tests/tools/test_patient_tools.py`（追加）

**Interfaces:**
- Consumes: 行为四分类与 `humanistic_agent.BEHAVIOR_TYPES = ["comfort", "explain", "instruction", "ignore"]` 同一套模型（评估侧闭环）
- Produces: `classify_doctor_behavior(message: str) -> str`（纯规则，默认 ignore）；`update_emotion(current: str, behavior: str, personality: str) -> str`；`EmotionEngine`（name=`emotion_engine`）包装两者

- [ ] **Step 1: 追加失败测试**

在 `test_patient_tools.py` 末尾追加：

```python
from app.services.tools.patient.emotion import (
    EmotionEngine,
    EmotionEngineArgs,
    classify_doctor_behavior,
    update_emotion,
)


class TestClassifyDoctorBehavior:
    def test_comfort(self):
        assert classify_doctor_behavior("别担心，这个病不严重，我们一起想办法。") == "comfort"

    def test_explain(self):
        assert classify_doctor_behavior("这个病的原因是胃酸分泌过多，所以会反酸。") == "explain"

    def test_instruction(self):
        assert classify_doctor_behavior("哪里不舒服？疼了多久了？") == "instruction"

    def test_default_ignore(self):
        assert classify_doctor_behavior("嗯。") == "ignore"


class TestUpdateEmotion:
    def test_comfort_calms_anxious(self):
        assert update_emotion("焦虑", "comfort", "焦虑型") == "缓和"

    def test_ignore_worsens(self):
        assert update_emotion("平静", "ignore", "对抗型") == "不满"

    def test_unknown_state_stays(self):
        assert update_emotion("自定义情绪", "explain", "配合型") == "自定义情绪"


class TestEmotionEngineTool:
    @pytest.mark.asyncio
    async def test_tool_wraps_functions(self):
        tool = EmotionEngine()
        args = EmotionEngineArgs(doctor_message="别担心，慢慢说。", current_emotion="焦虑", personality="焦虑型")
        result = await tool.execute(args, _context())
        assert result == {"behavior": "comfort", "emotion": "缓和"}
```

- [ ] **Step 2: 运行测试确认失败**

Run: `cd backend; venv\Scripts\python.exe -m pytest tests\tools\test_patient_tools.py -v`
Expected: 新用例 FAIL（`ModuleNotFoundError`）

- [ ] **Step 3: 实现**

`backend/app/services/tools/patient/emotion.py`：

```python
# -*- coding: utf-8 -*-
"""情绪引擎 — 医生行为四分类（与 humanistic_agent 评估侧同模型）+ 情绪状态转移

纯规则实现：医生安慰/解释会缓和情绪，忽视会恶化情绪，
情绪字符串注入患者 prompt 影响语气（由 PatientAgent 使用）。
"""
from pydantic import BaseModel, Field

from app.services.tools.base import BaseTool, ToolContext

# 行为分类关键词（命中优先级：comfort > explain > instruction，均未命中为 ignore）
_BEHAVIOR_KEYWORDS = {
    "comfort": ("别担心", "不用担心", "别紧张", "别害怕", "放宽心", "不要焦虑", "慢慢说", "不严重", "一起想办法", "理解你"),
    "explain": ("原因是", "因为", "这个病", "意味着", "也就是说", "机制", "所以会", "解释", "通俗地讲"),
    "instruction": ("？", "?", "多久", "哪里", "什么时候", "有没有", "是不是", "量一下", "做个检查", "建议", "按时"),
}

# 情绪转移表：(当前情绪, 行为) -> 新情绪；未登记组合保持原情绪
_EMOTION_TRANSITION = {
    ("焦虑", "comfort"): "缓和",
    ("焦虑", "explain"): "缓和",
    ("焦虑", "ignore"): "恐慌",
    ("缓和", "ignore"): "焦虑",
    ("平静", "ignore"): "不满",
    ("平静", "comfort"): "安心",
    ("不满", "comfort"): "平静",
    ("不满", "explain"): "平静",
    ("不满", "ignore"): "愤怒",
    ("恐慌", "comfort"): "焦虑",
    ("愤怒", "comfort"): "不满",
}


def classify_doctor_behavior(message: str) -> str:
    """医生消息 -> comfort/explain/instruction/ignore（与评估侧 BEHAVIOR_TYPES 对齐）"""
    text = (message or "").strip()
    if not text:
        return "ignore"
    for behavior in ("comfort", "explain", "instruction"):
        if any(kw in text for kw in _BEHAVIOR_KEYWORDS[behavior]):
            return behavior
    return "ignore"


def update_emotion(current: str, behavior: str, personality: str) -> str:
    """查表转移情绪；未登记组合保持原情绪（personality 预留给后续差异化转移）"""
    return _EMOTION_TRANSITION.get((current, behavior), current)


class EmotionEngineArgs(BaseModel):
    doctor_message: str = Field(description="医生本轮消息")
    current_emotion: str = Field(description="患者当前情绪")
    personality: str = Field(description="患者人格类型")


class EmotionEngine(BaseTool):
    name = "emotion_engine"
    description = "分类医生行为并更新患者情绪状态"
    args_schema = EmotionEngineArgs
    timeout_seconds = 5
    critical = False

    async def execute(self, args: EmotionEngineArgs, context: ToolContext) -> dict:
        behavior = classify_doctor_behavior(args.doctor_message)
        emotion = update_emotion(args.current_emotion, behavior, args.personality)
        return {"behavior": behavior, "emotion": emotion}
```

- [ ] **Step 4: 运行测试确认通过**

Run: `cd backend; venv\Scripts\python.exe -m pytest tests\tools\test_patient_tools.py -v`
Expected: 15 passed

- [ ] **Step 5: Commit**

```powershell
git add backend/app/services/tools/patient/emotion.py backend/tests/tools/test_patient_tools.py
git commit -m "feat(patient-tools): add emotion engine with behavior classification"
```

### Task 11: 工具注册 + 白名单 + PatientAgent 接线

**Files:**
- Modify: `backend/app/services/tools/patient/__init__.py`
- Modify: `backend/app/services/tools/policy.py`
- Modify: `backend/app/services/tools/__init__.py`
- Modify: `backend/app/services/agents/patient/agent.py`
- Test: `backend/tests/tools/test_patient_tools.py`（追加）、`backend/tests/agents/patient/test_agent.py`（追加）

**Interfaces:**
- Produces: `register_patient_tools(registry: ToolRegistry) -> None`；`PATIENT_TOOL_BUDGETS = {"query_plausible_symptom": 5}`；policy 新增 `"patient_agent"` 白名单；`PatientAgent` 情绪前置路由（每轮用 emotion 函数更新 `memory.emotion`，并将情绪注入 system prompt）

- [ ] **Step 1: 写失败测试**

`test_patient_tools.py` 末尾追加：

```python
from app.services.tools.patient import PATIENT_TOOL_BUDGETS, register_patient_tools
from app.services.tools.policy import AGENT_TOOL_WHITELIST
from app.services.tools.registry import ToolRegistry


class TestPatientToolRegistration:
    def test_register_idempotent(self):
        registry = ToolRegistry()
        register_patient_tools(registry)
        register_patient_tools(registry)  # 幂等不报错
        names = {t.name for t in registry.list_tools()}
        assert {"query_plausible_symptom", "physiology_calculator", "emotion_engine"} <= names

    def test_whitelist_registered(self):
        assert AGENT_TOOL_WHITELIST["patient_agent"] == frozenset(
            {"query_plausible_symptom", "physiology_calculator", "emotion_engine"}
        )

    def test_budget_limits(self):
        assert PATIENT_TOOL_BUDGETS == {"query_plausible_symptom": 5}
```

`test_agent.py` 的 `TestRespond` 类内追加：

```python
    @pytest.mark.asyncio
    async def test_emotion_updated_each_turn(self):
        agent = _agent()
        agent.memory.emotion = "焦虑"
        with patch("app.services.agents.patient.agent.call_qwen_chat", new=AsyncMock(return_value="嗯。")), \
             patch("app.services.agents.patient.agent.update_ledger", new=AsyncMock()):
            await agent.respond("别担心，慢慢说。", [])
        assert agent.memory.emotion == "缓和"

    def test_emotion_injected_into_prompt(self):
        agent = _agent()
        agent.memory.emotion = "恐慌"
        assert "当前情绪状态：恐慌" in agent._build_system_prompt()
```

- [ ] **Step 2: 运行测试确认失败**

Run: `cd backend; venv\Scripts\python.exe -m pytest tests\tools\test_patient_tools.py tests\agents\patient\test_agent.py -v`
Expected: 新用例 FAIL（`ImportError: register_patient_tools` / 情绪断言失败）

- [ ] **Step 3: 实现**

(a) `backend/app/services/tools/patient/__init__.py` 改为：

```python
# -*- coding: utf-8 -*-
"""患者智能体专属工具包 — 注册函数与预算配置"""
from app.services.tools.registry import ToolRegistry

from .emotion import EmotionEngine, classify_doctor_behavior, update_emotion
from .physiology import PhysiologyCalculator
from .plausible_symptom import QueryPlausibleSymptom

# 会话级调用预算：未登记的工具不限次（本地确定性工具无需限制）
PATIENT_TOOL_BUDGETS: dict[str, int] = {"query_plausible_symptom": 5}


def register_patient_tools(registry: ToolRegistry) -> None:
    """注册患者专属工具（幂等）"""
    registry.register(QueryPlausibleSymptom())
    registry.register(PhysiologyCalculator())
    registry.register(EmotionEngine())


__all__ = [
    "EmotionEngine",
    "PATIENT_TOOL_BUDGETS",
    "PhysiologyCalculator",
    "QueryPlausibleSymptom",
    "classify_doctor_behavior",
    "register_patient_tools",
    "update_emotion",
]
```

(b) `policy.py` 的 `AGENT_TOOL_WHITELIST` 字典中 `"reflection_agent"` 条目之后追加：

```python
    "patient_agent": frozenset({
        "query_plausible_symptom",
        "physiology_calculator",
        "emotion_engine",
    }),
```

(c) `tools/__init__.py`：import 区追加 `from .patient import PATIENT_TOOL_BUDGETS, register_patient_tools`；`register_all_tools` 末尾追加 `register_patient_tools(registry)`；`__all__` 的“注册函数”区追加 `"register_patient_tools"` 与 `"PATIENT_TOOL_BUDGETS"`。

(d) `agent.py`：import 区追加：

```python
from app.services.tools.patient.emotion import classify_doctor_behavior, update_emotion
```

`respond` 中 `self.memory.stage_history.append(stage)` 之后插入情绪前置路由：

```python
        # 情绪前置路由：行为分类 + 情绪转移（纯规则，零成本）
        behavior = classify_doctor_behavior(doctor_message)
        self.memory.emotion = update_emotion(
            self.memory.emotion, behavior, self.patient.personality_type or ""
        )
```

`_build_system_prompt` 中 `sections = [...]` 之后插入：

```python
        sections.append(f"【当前情绪状态：{self.memory.emotion}】请在语气中自然体现这种情绪。")
```

（注意：Task 5 的 `test_empty_ledger_no_sections` 不受影响——它只断言账本两段不存在。）

- [ ] **Step 4: 运行测试确认通过**

Run: `cd backend; venv\Scripts\python.exe -m pytest tests\tools\test_patient_tools.py tests\agents\patient\ tests\services\test_consultation_service.py -v`
Expected: 全部通过（含既有工具/白名单回归：`venv\Scripts\python.exe -m pytest tests\tools\ -v`）

- [ ] **Step 5: Commit**

```powershell
git add backend/app/services/tools/patient/__init__.py backend/app/services/tools/policy.py backend/app/services/tools/__init__.py backend/app/services/agents/patient/agent.py backend/tests/tools/test_patient_tools.py backend/tests/agents/patient/test_agent.py
git commit -m "feat(patient-tools): register patient tools with whitelist and wire emotion routing"
```

---

## P4 人格动力学与评估闭环

### Task 12: 信任动力学（dynamics.py）

**Files:**
- Create: `backend/app/services/agents/patient/dynamics.py`
- Modify: `backend/app/services/agents/patient/agent.py`
- Test: `backend/tests/agents/patient/test_dynamics.py`

**Interfaces:**
- Produces: `INITIAL_TRUST = {"配合型": 0.7, "焦虑型": 0.5, "沉默型": 0.4, "对抗型": 0.2}`；`BEHAVIOR_TRUST_DELTA = {"comfort": 0.10, "explain": 0.05, "instruction": 0.0, "ignore": -0.10}`；`TRUST_UNLOCK_THRESHOLD = 0.5`；`apply_turn_dynamics(memory, behavior) -> None`（trust 限幅 [0,1]）；`locked_facts(memory) -> list[Fact]`（trust < 阈值时的 empathy_unlock 未披露事实）；`initial_trust(personality) -> float`

- [ ] **Step 1: 写失败测试**

`backend/tests/agents/patient/test_dynamics.py`：

```python
# -*- coding: utf-8 -*-
"""dynamics.py 单元测试：信任动力学与敏感事实解锁"""
from app.services.agents.patient.dynamics import (
    BEHAVIOR_TRUST_DELTA,
    INITIAL_TRUST,
    TRUST_UNLOCK_THRESHOLD,
    apply_turn_dynamics,
    initial_trust,
    locked_facts,
)
from app.services.agents.patient.memory import Fact, MemoryState


def _memory(trust=0.4):
    return MemoryState(trust=trust, facts=[
        Fact(fact_id="sym_001", content="上腹隐痛"),
        Fact(fact_id="lif_001", category="lifestyle", content="长期饮酒", disclosure_condition="empathy_unlock"),
        Fact(fact_id="lif_002", category="lifestyle", content="已披露隐私", disclosure_condition="empathy_unlock", status="disclosed"),
    ])


class TestTrustDynamics:
    def test_constants(self):
        assert INITIAL_TRUST == {"配合型": 0.7, "焦虑型": 0.5, "沉默型": 0.4, "对抗型": 0.2}
        assert BEHAVIOR_TRUST_DELTA == {"comfort": 0.10, "explain": 0.05, "instruction": 0.0, "ignore": -0.10}
        assert TRUST_UNLOCK_THRESHOLD == 0.5

    def test_initial_trust_fallback(self):
        assert initial_trust("配合型") == 0.7
        assert initial_trust("未知人格") == 0.5

    def test_comfort_raises_trust(self):
        m = _memory(trust=0.4)
        apply_turn_dynamics(m, "comfort")
        assert abs(m.trust - 0.5) < 1e-9

    def test_trust_clamped(self):
        m = _memory(trust=0.05)
        apply_turn_dynamics(m, "ignore")
        assert m.trust == 0.0
        m2 = _memory(trust=0.95)
        apply_turn_dynamics(m2, "comfort")
        assert m2.trust == 1.0


class TestLockedFacts:
    def test_low_trust_locks_undisclosed_sensitive(self):
        m = _memory(trust=0.3)
        assert [f.fact_id for f in locked_facts(m)] == ["lif_001"]

    def test_high_trust_unlocks_all(self):
        m = _memory(trust=0.6)
        assert locked_facts(m) == []
```

- [ ] **Step 2: 运行测试确认失败**

Run: `cd backend; venv\Scripts\python.exe -m pytest tests\agents\patient\test_dynamics.py -v`
Expected: FAIL（`ModuleNotFoundError`）

- [ ] **Step 3: 实现**

`backend/app/services/agents/patient/dynamics.py`：

```python
# -*- coding: utf-8 -*-
"""信任动力学 — 医生行为驱动患者信任度变化，信任解锁敏感事实

行为四分类与 humanistic_agent 评估侧同模型：医生安慰/解释提升信任，
忽视降低信任；trust >= 阈值时 empathy_unlock 事实才允许披露。
"""
from .memory import Fact, MemoryState

INITIAL_TRUST: dict[str, float] = {"配合型": 0.7, "焦虑型": 0.5, "沉默型": 0.4, "对抗型": 0.2}
BEHAVIOR_TRUST_DELTA: dict[str, float] = {"comfort": 0.10, "explain": 0.05, "instruction": 0.0, "ignore": -0.10}
TRUST_UNLOCK_THRESHOLD = 0.5


def initial_trust(personality: str) -> float:
    """人格 -> 初始信任度；未知人格取中性 0.5"""
    return INITIAL_TRUST.get(personality, 0.5)


def apply_turn_dynamics(memory: MemoryState, behavior: str) -> None:
    """按医生行为更新信任度，限幅 [0, 1]"""
    delta = BEHAVIOR_TRUST_DELTA.get(behavior, 0.0)
    memory.trust = max(0.0, min(1.0, memory.trust + delta))


def locked_facts(memory: MemoryState) -> list[Fact]:
    """信任不足时仍锁定的敏感事实（empathy_unlock 且未披露）"""
    if memory.trust >= TRUST_UNLOCK_THRESHOLD:
        return []
    return [
        f for f in memory.facts
        if f.disclosure_condition == "empathy_unlock" and f.status == "undisclosed"
    ]
```

`agent.py` 接线：import 区追加 `from .dynamics import apply_turn_dynamics, initial_trust, locked_facts`；`__init__` 末尾追加（仅首轮生效）：

```python
        if memory.turn == 0:
            memory.trust = initial_trust(patient.personality_type or "")
```

`respond` 情绪路由段之后追加 `apply_turn_dynamics(self.memory, behavior)`；`_build_system_prompt` 情绪段之后追加锁定事实注入：

```python
        locked = locked_facts(self.memory)
        if locked:
            lines = "\n".join(f"- {f.content}" for f in locked)
            sections.append(
                "【你暂时不愿意透露的隐私信息】（对医生信任不够，被问到时含糊回避，"
                "如“这个……不好说”；若医生安慰共情你，后续可以坦白）\n" + lines
            )
```

`test_agent.py` 的 `TestRespond` 内追加一条验证：

```python
    @pytest.mark.asyncio
    async def test_trust_rises_on_comfort(self):
        agent = _agent()
        agent.memory.trust = 0.4
        with patch("app.services.agents.patient.agent.call_qwen_chat", new=AsyncMock(return_value="嗯。")), \
             patch("app.services.agents.patient.agent.update_ledger", new=AsyncMock()):
            await agent.respond("别担心，慢慢说。", [])
        assert abs(agent.memory.trust - 0.5) < 1e-9
```

- [ ] **Step 4: 运行测试确认通过**

Run: `cd backend; venv\Scripts\python.exe -m pytest tests\agents\patient\ -v`
Expected: 全部通过

- [ ] **Step 5: Commit**

```powershell
git add backend/app/services/agents/patient/dynamics.py backend/app/services/agents/patient/agent.py backend/tests/agents/patient/test_dynamics.py backend/tests/agents/patient/test_agent.py
git commit -m "feat(patient-agent): add trust dynamics with sensitive fact unlocking"
```

### Task 13: 人格×阶段策略矩阵（strategy.py）

**Files:**
- Create: `backend/app/services/agents/patient/strategy.py`
- Modify: `backend/app/services/agents/patient/agent.py`
- Test: `backend/tests/agents/patient/test_strategy.py`

**Interfaces:**
- Produces: `DisclosureStrategy(reply_length: str, volunteer_info: bool, ask_back: bool, tone_hint: str)`（dataclass）；`get_strategy(personality: str, stage: str) -> DisclosureStrategy`（未登记组合回退 `_DEFAULTS[personality]`，再回退配合型默认）

- [ ] **Step 1: 写失败测试**

`backend/tests/agents/patient/test_strategy.py`：

```python
# -*- coding: utf-8 -*-
"""strategy.py 单元测试：人格×阶段策略查询"""
from app.services.agents.patient.planner import STAGES
from app.services.agents.patient.strategy import DisclosureStrategy, get_strategy


class TestGetStrategy:
    def test_anxious_hpi_asks_back(self):
        s = get_strategy("焦虑型", "hpi")
        assert s.ask_back is True and s.tone_hint

    def test_reticent_always_short(self):
        for stage in STAGES:
            assert get_strategy("沉默型", stage).reply_length == "极短"

    def test_unknown_combo_falls_back(self):
        s = get_strategy("未知人格", "未知阶段")
        assert isinstance(s, DisclosureStrategy)

    def test_cooperative_assessment_may_volunteer(self):
        assert get_strategy("配合型", "assessment_communication").volunteer_info is True
```

- [ ] **Step 2: 运行测试确认失败**

Run: `cd backend; venv\Scripts\python.exe -m pytest tests\agents\patient\test_strategy.py -v`
Expected: FAIL（`ModuleNotFoundError`）

- [ ] **Step 3: 实现**

`backend/app/services/agents/patient/strategy.py`：

```python
# -*- coding: utf-8 -*-
"""披露策略矩阵 — 人格×问诊阶段 决定回复风格提示

只产出 prompt 风格提示，不硬控制生成；未登记组合逐级回退，永不报错。
"""
from dataclasses import dataclass


@dataclass(frozen=True)
class DisclosureStrategy:
    reply_length: str      # 极短 / 简短 / 正常
    volunteer_info: bool   # 是否允许少量主动补充
    ask_back: bool         # 是否倾向反问医生
    tone_hint: str         # 语气提示（直接注入 prompt）


_DEFAULTS: dict[str, DisclosureStrategy] = {
    "配合型": DisclosureStrategy("简短", False, False, "态度友好，回答清楚"),
    "焦虑型": DisclosureStrategy("简短", False, True, "语气担忧，容易往坏处想"),
    "沉默型": DisclosureStrategy("极短", False, False, "惜字如金，语气冷淡"),
    "对抗型": DisclosureStrategy("简短", False, False, "不耐烦，语气带刺但不拒答"),
}

# 仅登记与默认行为不同的组合，其余回退 _DEFAULTS
_STRATEGY_MATRIX: dict[tuple[str, str], DisclosureStrategy] = {
    ("配合型", "chief_complaint"): DisclosureStrategy("简短", False, False, "清楚说出主要不适，不展开细节"),
    ("配合型", "assessment_communication"): DisclosureStrategy("正常", True, True, "关心诊断结果，可主动确认注意事项"),
    ("焦虑型", "greeting"): DisclosureStrategy("简短", False, True, "寒暄中带焦虑，急于说病情"),
    ("焦虑型", "hpi"): DisclosureStrategy("简短", False, True, "描述症状时追问“是不是很严重”"),
    ("焦虑型", "assessment_communication"): DisclosureStrategy("正常", False, True, "反复确认风险，担心预后"),
    ("沉默型", "physical_exam"): DisclosureStrategy("极短", False, False, "配合检查但不多说一字"),
    ("对抗型", "greeting"): DisclosureStrategy("简短", False, False, "开场就显不耐烦"),
    ("对抗型", "past_history"): DisclosureStrategy("简短", False, True, "质疑“问这些有什么用”但仍回答"),
    ("对抗型", "assessment_communication"): DisclosureStrategy("正常", False, True, "对诊断持怀疑态度，追问依据"),
}

_FALLBACK = _DEFAULTS["配合型"]


def get_strategy(personality: str, stage: str) -> DisclosureStrategy:
    """查询人格×阶段策略；组合未登记回退人格默认，人格未知回退配合型"""
    return _STRATEGY_MATRIX.get((personality, stage)) or _DEFAULTS.get(personality, _FALLBACK)
```

`agent.py` 接线：import 区追加 `from .strategy import get_strategy`；`_build_system_prompt` 末尾（return 前）追加：

```python
        strategy = get_strategy(self.patient.personality_type or "", self.memory.stage)
        sections.append(
            f"【本轮回复风格】长度：{strategy.reply_length}；语气：{strategy.tone_hint}"
            + ("；可少量主动补充相关信息" if strategy.volunteer_info else "")
            + ("；可向医生反问一个问题" if strategy.ask_back else "")
        )
```

- [ ] **Step 4: 运行测试确认通过**

Run: `cd backend; venv\Scripts\python.exe -m pytest tests\agents\patient\ -v`
Expected: 全部通过

- [ ] **Step 5: Commit**

```powershell
git add backend/app/services/agents/patient/strategy.py backend/app/services/agents/patient/agent.py backend/tests/agents/patient/test_strategy.py
git commit -m "feat(patient-agent): add personality-stage disclosure strategy matrix"
```

### Task 14: 问诊覆盖报告回馈评估侧（coverage.py）

**Files:**
- Create: `backend/app/services/agents/patient/coverage.py`
- Modify: `backend/app/services/evaluation_service.py`
- Test: `backend/tests/agents/patient/test_coverage.py`

**Interfaces:**
- Consumes: `MemoryState`；`evaluation_service` 中 `patient_info` 构建（542-549 行）与 `consultation` 对象
- Produces: `build_coverage_report(memory) -> dict`（`disclosure_rate`/`disclosed_count`/`total_facts`/`undisclosed_facts`/`stage_path`/`final_trust`/`final_emotion`）；`format_coverage_text(report) -> str`

- [ ] **Step 1: 写失败测试**

`backend/tests/agents/patient/test_coverage.py`：

```python
# -*- coding: utf-8 -*-
"""coverage.py 单元测试：披露账本 -> 问诊覆盖报告"""
from app.services.agents.patient.coverage import build_coverage_report, format_coverage_text
from app.services.agents.patient.memory import Fact, MemoryState


def _memory():
    return MemoryState(
        trust=0.65, emotion="缓和",
        stage_history=["greeting", "chief_complaint", "chief_complaint", "hpi"],
        facts=[
            Fact(fact_id="sym_001", content="上腹隐痛", status="disclosed", disclosed_at_turn=2),
            Fact(fact_id="sym_002", content="反酸烧心", status="disclosed", disclosed_at_turn=3),
            Fact(fact_id="his_001", category="history", content="胃溃疡史"),
            Fact(fact_id="his_002", category="history", content="青霉素过敏", status="denied"),
        ],
    )


class TestBuildCoverageReport:
    def test_report_fields(self):
        r = build_coverage_report(_memory())
        assert r["total_facts"] == 4 and r["disclosed_count"] == 2
        assert r["disclosure_rate"] == 0.5
        assert r["undisclosed_facts"] == ["胃溃疡史"]
        assert r["stage_path"] == ["greeting", "chief_complaint", "hpi"]  # 去重保序
        assert r["final_trust"] == 0.65 and r["final_emotion"] == "缓和"

    def test_empty_memory_no_division_error(self):
        r = build_coverage_report(MemoryState())
        assert r["disclosure_rate"] == 0.0 and r["total_facts"] == 0


class TestFormatCoverageText:
    def test_contains_key_numbers(self):
        text = format_coverage_text(build_coverage_report(_memory()))
        assert "50.0%" in text and "胃溃疡史" in text and "0.65" in text
```

- [ ] **Step 2: 运行测试确认失败**

Run: `cd backend; venv\Scripts\python.exe -m pytest tests\agents\patient\test_coverage.py -v`
Expected: FAIL（`ModuleNotFoundError`）

- [ ] **Step 3: 实现**

`backend/app/services/agents/patient/coverage.py`：

```python
# -*- coding: utf-8 -*-
"""问诊覆盖报告 — 披露账本汇总为客观统计，回馈评估侧

把“医生问出了多少/漏问了什么”从 LLM 主观判断变为系统客观数据，
注入病史采集评估 Agent 的 patient_info 提升评分准确性。
"""
from .memory import MemoryState


def build_coverage_report(memory: MemoryState) -> dict:
    """账本 -> 覆盖统计（纯计算，无 LLM）"""
    total = len(memory.facts)
    disclosed = memory.facts_by_status("disclosed")
    stage_path: list[str] = []
    for s in memory.stage_history:
        if not stage_path or stage_path[-1] != s:
            stage_path.append(s)
    return {
        "total_facts": total,
        "disclosed_count": len(disclosed),
        "disclosure_rate": round(len(disclosed) / total, 4) if total else 0.0,
        "undisclosed_facts": [f.content for f in memory.facts_by_status("undisclosed")],
        "stage_path": stage_path,
        "final_trust": memory.trust,
        "final_emotion": memory.emotion,
    }


def format_coverage_text(report: dict) -> str:
    """覆盖报告 -> 注入评估 prompt 的中文文本块"""
    undisclosed = report["undisclosed_facts"]
    lines = [
        f"事实披露率：{report['disclosure_rate'] * 100:.1f}%（{report['disclosed_count']}/{report['total_facts']}）",
        f"问诊阶段路径：{' -> '.join(report['stage_path']) or '（无）'}",
        f"结束时患者信任度：{report['final_trust']}，情绪：{report['final_emotion']}",
        "医生未问出的档案事实：" + ("、".join(undisclosed) if undisclosed else "（无，全部问出）"),
    ]
    return "\n".join(lines)
```

`evaluation_service.py` 接线：在 `patient_info = (...)` 赋值（542-549 行）之后、`doctor_diagnosis = ...` 之前插入：

```python
    # 患者智能体披露账本 -> 客观覆盖统计，增强病史采集评估准确性（失败静默跳过）
    try:
        from app.services.agents.patient.coverage import build_coverage_report, format_coverage_text
        from app.services.agents.patient.memory import MemoryState

        memory = MemoryState.from_json(consultation.memory_state)
        if memory is not None and memory.facts:
            coverage_text = format_coverage_text(build_coverage_report(memory))
            patient_info += f"\n\n【问诊信息披露账本（系统客观统计）】\n{coverage_text}"
    except Exception as e:
        logging.warning(f"披露账本注入评估失败，跳过: {e}")
```

- [ ] **Step 4: 运行测试确认通过**

Run: `cd backend; venv\Scripts\python.exe -m pytest tests\agents\patient\test_coverage.py tests\services\ -v`
Expected: coverage 3 passed；evaluation/consultation 既有用例无回归

- [ ] **Step 5: Commit**

```powershell
git add backend/app/services/agents/patient/coverage.py backend/app/services/evaluation_service.py backend/tests/agents/patient/test_coverage.py
git commit -m "feat(patient-agent): feed disclosure coverage report into evaluation"
```

---

## 整体验收

- [ ] **全量回归**：`cd backend; venv\Scripts\python.exe -m pytest tests\ -v --tb=short` — 全部通过（无 LLM 真实调用）
- [ ] **应用可启动**：`cd backend; venv\Scripts\python.exe -c "from app.main import app; print('OK')"` — 输出 OK（导入链无循环依赖）
- [ ] **人工冒烟**（可选，需启动前后端 + 真实 LLM）：新建问诊 → 问同一症状两次确认口径一致 → 否认后再问确认不翻供 → 结束后查 `consultations.memory_state` 非空
- [ ] **不 push**：全部工作仅本地 commit，推送需用户确认并走 security-scan 交接

## 风险与回滚

- 每个 Task 独立 commit，任意一步引入问题可 `git revert` 单点回滚
- 运行时双重保险：`_generate_patient_reply` 异常回退旧路径；`memory_state` 列 nullable，旧会话（列值 NULL）自动走初始化路径
- 迁移脚本幂等，可重复执行；如需回滚列：`ALTER TABLE consultations DROP COLUMN memory_state`（仅在用户明确要求时执行）
