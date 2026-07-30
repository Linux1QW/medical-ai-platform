# -*- coding: utf-8 -*-
"""患者模拟智能体 A/B 回放验证 — 旧无记忆 prompt 路径 vs 新披露账本智能体

医生侧刺激：直接回放 dataset/patientN_XX 主 JSON 中「门诊对话」的真实医生提问序列，
两臂收到完全相同的问题序列，保证可比性。

两臂均复用生产代码：
- legacy 臂：consultation_service._legacy_generate_patient_reply
- agent  臂：extract_facts 初始化账本 + PatientAgent.respond（与 _generate_patient_reply 等价）

指标：逐轮延迟、LLM 调用次数（按模块归因）、回复长度；agent 臂另出
披露率/漏问事实/信任-情绪轨迹/阶段路径（复用 build_coverage_report）与矛盾重生成次数。

用法（在 backend 目录下）：
    .\\venv\\Scripts\\python.exe scripts\\ab_patient_replay.py --cases patient100_21,patient11_9 --turns-cap 10
    .\\venv\\Scripts\\python.exe scripts\\ab_patient_replay.py --limit 20 --turns-cap 12   # 批量（按目录名排序取前 N）
"""
import argparse
import asyncio
import json
import os
import statistics
import sys
import time
import zlib
from datetime import datetime
from pathlib import Path

# Windows 下 httpx 会读系统注册表代理（127.0.0.1:7890），代理软件未运行时直连 dashscope
os.environ["NO_PROXY"] = "*"
os.environ["no_proxy"] = "*"
# 离线回放不需要 Redis：关 LLM 缓存，避免 temp=0 调用每次付 3s 连接超时污染延迟测量
os.environ["LLM_CACHE_ENABLED"] = "false"

sys.stdout.reconfigure(encoding="utf-8")
sys.path.insert(0, str(Path(__file__).parent.parent))

from seed_patients import PERSONALITY_MAP, generate_system_prompt  # noqa: E402

from app.services import consultation_service  # noqa: E402
from app.services import token_tracker as token_tracker_mod  # noqa: E402
from app.services.agents.patient import agent as patient_agent_mod  # noqa: E402
from app.services.agents.patient import guard as guard_mod  # noqa: E402
from app.services.agents.patient import memory as memory_mod  # noqa: E402
from app.services.agents.patient.coverage import build_coverage_report  # noqa: E402
from app.services.agents.patient.memory import MemoryState  # noqa: E402

DATASET_DIR = Path(__file__).parent.parent.parent / "dataset"
REPORT_DIR = Path(__file__).parent.parent / "evaluation" / "reports" / "patient_ab"


# ── LLM 调用打点 ────────────────────────────────────────────────────────────

class LLMCounter:
    """包装 call_qwen_chat / call_qwen_with_tools，按消费模块统计调用次数与耗时（from-import 需逐模块 patch）"""

    _TARGETS = [
        (consultation_service, "call_qwen_chat", "legacy"),        # legacy 臂主回复 / 早期摘要
        (patient_agent_mod, "call_qwen_chat", "agent_reply"),       # agent 臂纯文本回复 + 矛盾重生成
        (patient_agent_mod, "call_qwen_with_tools", "agent_tools"),  # agent 臂 function-calling 主回复
        (guard_mod, "call_qwen_chat", "ledger"),                    # 账本更新
        (memory_mod, "call_qwen_chat", "fact_extract"),             # 事实抽取
    ]

    def __init__(self):
        self.calls: list[dict] = []
        self._originals: list = []

    def _wrap(self, real_fn, tag: str):
        async def wrapper(*args, **kwargs):
            t0 = time.perf_counter()
            result = await real_fn(*args, **kwargs)
            self.calls.append({"module": tag, "elapsed_ms": round((time.perf_counter() - t0) * 1000)})
            return result
        return wrapper

    def install(self):
        for mod, attr, tag in self._TARGETS:
            real = getattr(mod, attr)
            self._originals.append((mod, attr, real))
            setattr(mod, attr, self._wrap(real, tag))

    def uninstall(self):
        for mod, attr, real in self._originals:
            setattr(mod, attr, real)
        self._originals.clear()

    def snapshot(self) -> int:
        return len(self.calls)

    def since(self, mark: int) -> list[dict]:
        return self.calls[mark:]


# ── dataset 病例加载 ────────────────────────────────────────────────────────

class ReplayPatient:
    """鸭子类型患者对象（PatientAgent / legacy 路径只读属性，不需要 ORM）"""

    def __init__(self, data: dict, case_id: str):
        basic = data.get("基础信息", {}) or {}
        record = data.get("门诊病历", {}) or {}
        persona = data.get("人格", {}) or {}

        self.case_id = case_id
        self.name = basic.get("姓名", case_id)
        age_str = str(basic.get("年龄", "30"))
        self.age = int(age_str) if age_str.isdigit() else 30
        self.gender = "female" if basic.get("性别") == "女" else "male"
        self.personality_type = PERSONALITY_MAP.get(
            str(persona.get("性格", "")).strip(), "配合型"
        )
        self.chief_complaint = _clean(record.get("主诉"))
        self.medical_history = _clean(record.get("既往史")) or "既往体健"
        self.symptoms = _clean(record.get("现病史")) or self.chief_complaint or "详见问诊"
        self.expected_diagnosis = _clean(data.get("主诊断"))
        self.system_prompt = generate_system_prompt(data, self.personality_type)


class ReplayMessage:
    """鸭子类型消息对象（_build_history 只读 role/content）"""

    def __init__(self, role: str, content: str):
        self.role = role
        self.content = content


def _clean(text) -> str:
    if not isinstance(text, str):
        return ""
    return text.replace("\u200b", "").replace("\ufeff", "").strip()


def load_case(case_id: str) -> tuple[ReplayPatient, list[str]] | None:
    """加载病例：患者档案 + 医生提问序列；对话为空返回 None"""
    main_json = DATASET_DIR / case_id / f"{case_id}.json"
    if not main_json.exists():
        print(f"[跳过] 主 JSON 缺失: {main_json}")
        return None
    with open(main_json, "r", encoding="utf-8") as f:
        data = json.load(f)
    dialogue = data.get("门诊对话") or []
    doctor_turns = [_clean(t.get("医生")) for t in dialogue if isinstance(t, dict) and _clean(t.get("医生"))]
    if not doctor_turns:
        print(f"[跳过] 门诊对话为空: {case_id}")
        return None
    return ReplayPatient(data, case_id), doctor_turns


# ── 两臂回放 ────────────────────────────────────────────────────────────────

async def run_legacy_arm(patient: ReplayPatient, doctor_turns: list[str], counter: LLMCounter) -> dict:
    """旧无记忆路径：逐轮调用生产回退函数"""
    messages: list[ReplayMessage] = []
    turns = []
    for i, doctor_text in enumerate(doctor_turns, start=1):
        mark = counter.snapshot()
        t0 = time.perf_counter()
        reply = await consultation_service._legacy_generate_patient_reply(patient, messages, doctor_text)
        latency_ms = round((time.perf_counter() - t0) * 1000)
        messages.append(ReplayMessage("doctor", doctor_text))
        messages.append(ReplayMessage("patient", reply))
        turns.append({
            "turn": i, "doctor": doctor_text, "reply": reply,
            "latency_ms": latency_ms, "llm_calls": len(counter.since(mark)),
        })
    return {"turns": turns}


async def run_agent_arm(patient: ReplayPatient, doctor_turns: list[str], counter: LLMCounter) -> dict:
    """新披露账本智能体路径：与 _generate_patient_reply 等价的离线复刻"""
    mark = counter.snapshot()
    t0 = time.perf_counter()
    facts = await memory_mod.extract_facts(
        patient.chief_complaint, patient.medical_history, patient.symptoms
    )
    init_ms = round((time.perf_counter() - t0) * 1000)
    memory = MemoryState(facts=facts)
    init_calls = len(counter.since(mark))

    messages: list[ReplayMessage] = []
    turns = []
    regen_count = 0
    for i, doctor_text in enumerate(doctor_turns, start=1):
        mark = counter.snapshot()
        t0 = time.perf_counter()
        # 工具路径 physiology 种子：用 case_id 的 CRC32 作确定性伪 consultation_id，避免跨病例同种子
        agent = patient_agent_mod.PatientAgent(
            patient, memory, consultation_id=zlib.crc32(patient.case_id.encode("utf-8"))
        )
        history = await consultation_service._build_history(messages, patient.system_prompt)
        reply = await agent.respond(doctor_text, history)
        latency_ms = round((time.perf_counter() - t0) * 1000)
        turn_calls = counter.since(mark)
        # agent_reply 模块一轮 >1 次调用 = 触发了矛盾重生成
        reply_calls = sum(1 for c in turn_calls if c["module"] == "agent_reply")
        if reply_calls > 1:
            regen_count += 1
        messages.append(ReplayMessage("doctor", doctor_text))
        messages.append(ReplayMessage("patient", reply))
        turns.append({
            "turn": i, "doctor": doctor_text, "reply": reply,
            "latency_ms": latency_ms, "llm_calls": len(turn_calls),
            "trust": round(memory.trust, 3), "emotion": memory.emotion, "stage": memory.stage,
        })
    return {
        "turns": turns,
        "fact_extract": {"latency_ms": init_ms, "llm_calls": init_calls, "fact_count": len(facts)},
        "regen_count": regen_count,
        "coverage": build_coverage_report(memory),
    }


# ── 指标汇总 ────────────────────────────────────────────────────────────────

def summarize_arm(arm: dict) -> dict:
    turns = arm["turns"]
    latencies = [t["latency_ms"] for t in turns]
    summary = {
        "turns": len(turns),
        "total_llm_calls": sum(t["llm_calls"] for t in turns),
        "latency_avg_ms": round(statistics.mean(latencies)) if latencies else 0,
        "latency_p95_ms": round(sorted(latencies)[max(0, int(len(latencies) * 0.95) - 1)]) if latencies else 0,
        "reply_chars_avg": round(statistics.mean(len(t["reply"]) for t in turns)) if turns else 0,
    }
    if "coverage" in arm:
        cov = arm["coverage"]
        summary.update({
            "fact_extract_calls": arm["fact_extract"]["llm_calls"],
            "fact_count": arm["fact_extract"]["fact_count"],
            "regen_count": arm["regen_count"],
            "disclosure_rate": cov["disclosure_rate"],
            "undisclosed_count": len(cov["undisclosed_facts"]),
            "final_trust": cov["final_trust"],
            "final_emotion": cov["final_emotion"],
            "stage_path": " -> ".join(cov["stage_path"]),
        })
    return summary


async def run_case(case_id: str, turns_cap: int, counter: LLMCounter) -> dict | None:
    loaded = load_case(case_id)
    if loaded is None:
        return None
    patient, doctor_turns = loaded
    doctor_turns = doctor_turns[:turns_cap]
    print(f"\n=== {case_id} | {patient.personality_type} | {len(doctor_turns)} 轮 | 主诉: {patient.chief_complaint[:30]} ===")

    legacy = await run_legacy_arm(patient, doctor_turns, counter)
    print(f"  [legacy] 完成 {len(legacy['turns'])} 轮")
    agent = await run_agent_arm(patient, doctor_turns, counter)
    print(f"  [agent ] 完成 {len(agent['turns'])} 轮，披露率 {agent['coverage']['disclosure_rate']:.0%}，重生成 {agent['regen_count']} 次")

    return {
        "case_id": case_id,
        "personality": patient.personality_type,
        "expected_diagnosis": patient.expected_diagnosis,
        "legacy": {"summary": summarize_arm(legacy), "detail": legacy},
        "agent": {"summary": summarize_arm(agent), "detail": agent},
    }


async def main():
    parser = argparse.ArgumentParser(description="患者智能体 A/B 回放验证")
    parser.add_argument("--cases", default="", help="逗号分隔的病例目录名；为空时按 --limit 取")
    parser.add_argument("--limit", type=int, default=2, help="未指定 --cases 时按目录名排序取前 N 例")
    parser.add_argument("--turns-cap", type=int, default=10, help="每例最多回放的医生提问轮数（控成本，≤14 可避开早期摘要调用）")
    args = parser.parse_args()

    # 离线环境无 Redis：禁用 token_tracker，避免每次 LLM 调用后 3s 连接超时
    async def _noop_record_usage(*_args, **_kwargs):
        return None
    token_tracker_mod.token_tracker.record_usage = _noop_record_usage

    if args.cases:
        case_ids = [c.strip() for c in args.cases.split(",") if c.strip()]
    else:
        case_ids = sorted(p.name for p in DATASET_DIR.iterdir() if p.is_dir())[: args.limit]

    from app.core.config import settings
    print(f"模型: {settings.llm_model} | 病例: {case_ids} | 轮数上限: {args.turns_cap}")

    counter = LLMCounter()
    counter.install()
    results = []
    try:
        for case_id in case_ids:
            result = await run_case(case_id, args.turns_cap, counter)
            if result is not None:
                results.append(result)
    finally:
        counter.uninstall()

    report = {
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "model": settings.llm_model,
        "turns_cap": args.turns_cap,
        "cases": results,
    }
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    out_path = REPORT_DIR / f"ab_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)

    # 控制台汇总表
    print("\n" + "=" * 100)
    print(f"{'case':<18}{'臂':<8}{'轮':>3}{'LLM调用':>8}{'均延迟ms':>10}{'P95ms':>8}{'均字数':>7}  账本指标")
    for r in results:
        for arm_name in ("legacy", "agent"):
            s = r[arm_name]["summary"]
            extra = ""
            if arm_name == "agent":
                extra = (f"披露率{s['disclosure_rate']:.0%} 漏问{s['undisclosed_count']} "
                         f"重生成{s['regen_count']} 信任{s['final_trust']} 情绪{s['final_emotion']}")
            print(f"{r['case_id']:<18}{arm_name:<8}{s['turns']:>3}{s['total_llm_calls']:>8}"
                  f"{s['latency_avg_ms']:>10}{s['latency_p95_ms']:>8}{s['reply_chars_avg']:>7}  {extra}")
    print(f"\n报告已写入: {out_path}")


if __name__ == "__main__":
    asyncio.run(main())
