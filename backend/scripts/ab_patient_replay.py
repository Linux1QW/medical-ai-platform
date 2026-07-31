# -*- coding: utf-8 -*-
"""患者模拟智能体三臂回放验证 — 旧无记忆 prompt / 新披露账本（工具关）/ 账本+工具（工具开）

医生侧刺激：直接回放 dataset/patientN_XX 主 JSON 中「门诊对话」的真实医生提问序列，
三臂收到完全相同的问题序列，保证可比性。

三臂均复用生产代码（零生产代码修改）：
- legacy 臂：consultation_service._legacy_generate_patient_reply
- agent_ledger 臂：extract_facts 初始化账本 + PatientAgent.respond（ENABLE_PATIENT_TOOL_USE=False）
- agent_tool 臂：同上，但临时置 ENABLE_PATIENT_TOOL_USE=True 开启工具路径

指标：逐轮延迟、LLM 调用次数（按模块归因）、回复长度；agent 臂另出披露率/漏问事实/
信任-情绪轨迹/阶段路径（复用 build_coverage_report）与矛盾重生成次数；agent_tool 臂另出工具使用/降级统计。
可选 LLM-as-Judge 软分（--judge，默认开）与 badcase 归档。

用法（在 backend 目录下）：
    .\\venv\\Scripts\\python.exe scripts\\ab_patient_replay.py --cases patient100_21,patient11_9 --turns-cap 10
    .\\venv\\Scripts\\python.exe scripts\\ab_patient_replay.py --cases @eval_set --turns-cap 10   # 固化评测集全量
    .\\venv\\Scripts\\python.exe scripts\\ab_patient_replay.py --cases @eval_set --limit 3 --no-judge   # 冒烟
    .\\venv\\Scripts\\python.exe scripts\\ab_patient_replay.py --cases @eval_set --no-judge --resume evaluation\\reports\\patient_ab\\ab_YYYYMMDD_HHMMSS.json   # 断点续跑
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

from app.core.config import settings  # noqa: E402
from app.services import consultation_service  # noqa: E402
from app.services import token_tracker as token_tracker_mod  # noqa: E402
from app.services.agents.patient import agent as patient_agent_mod  # noqa: E402
from app.services.agents.patient import guard as guard_mod  # noqa: E402
from app.services.agents.patient import memory as memory_mod  # noqa: E402
from app.services.agents.patient.coverage import build_coverage_report  # noqa: E402
from app.services.agents.patient.memory import MemoryState  # noqa: E402
from evaluation.patient_eval_set import load_eval_set  # noqa: E402
from evaluation.patient_judge import DIMENSIONS, judge_turn  # noqa: E402

DATASET_DIR = Path(__file__).parent.parent.parent / "dataset"
REPORT_DIR = Path(__file__).parent.parent / "evaluation" / "reports" / "patient_ab"
DEFAULT_EVAL_SET = Path(__file__).parent.parent / "evaluation" / "patient_cases" / "patient_sim_v1.jsonl"

# 三臂命名：legacy（无记忆）/ agent_ledger（账本，工具关）/ agent_tool（账本+工具）
ARMS = ("legacy", "agent_ledger", "agent_tool")


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


# ── 三臂回放 ────────────────────────────────────────────────────────────────

async def _judge_reply(patient, doctor_text: str, reply: str, history, enabled: bool) -> dict | None:
    """对单轮回复做 LLM-as-Judge 软分；enabled=False 或异常时返回 None（不阻断回放）。

    judge 在延迟测量之后调用，且走 evaluation.patient_judge 命名空间的 call_qwen_chat，
    不被 LLMCounter patch，故不计入本臂 LLM 成本 / 延迟。
    """
    if not enabled:
        return None
    profile = {
        "personality": patient.personality_type,
        "diagnosis": patient.expected_diagnosis,
        "chief_complaint": patient.chief_complaint,
    }
    score = await judge_turn(doctor_text, reply, profile, history)
    return score.model_dump()


async def run_legacy_arm(patient: ReplayPatient, doctor_turns: list[str], counter: LLMCounter,
                         judge_enabled: bool = False) -> dict:
    """旧无记忆路径：逐轮调用生产回退函数"""
    messages: list[ReplayMessage] = []
    turns = []
    for i, doctor_text in enumerate(doctor_turns, start=1):
        mark = counter.snapshot()
        t0 = time.perf_counter()
        reply = await consultation_service._legacy_generate_patient_reply(patient, messages, doctor_text)
        latency_ms = round((time.perf_counter() - t0) * 1000)
        judge = await _judge_reply(patient, doctor_text, reply, list(messages), judge_enabled)
        messages.append(ReplayMessage("doctor", doctor_text))
        messages.append(ReplayMessage("patient", reply))
        turns.append({
            "turn": i, "doctor": doctor_text, "reply": reply,
            "latency_ms": latency_ms, "llm_calls": len(counter.since(mark)),
            "judge": judge,
        })
    return {"turns": turns}


async def run_agent_arm(patient: ReplayPatient, doctor_turns: list[str], counter: LLMCounter,
                        use_tools: bool, judge_enabled: bool = False) -> dict:
    """新披露账本智能体路径：与 _generate_patient_reply 等价的离线复刻。

    use_tools 临时覆写 settings.ENABLE_PATIENT_TOOL_USE（try/finally 复位），
    以此拆出 agent_ledger（工具关）与 agent_tool（工具开）两臂。
    """
    original_tool_flag = settings.ENABLE_PATIENT_TOOL_USE
    settings.ENABLE_PATIENT_TOOL_USE = use_tools
    try:
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
        tool_used_turns = 0
        tool_degraded_turns = 0
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
            reply_calls = sum(1 for c in turn_calls if c["module"] == "agent_reply")
            tool_calls_n = sum(1 for c in turn_calls if c["module"] == "agent_tools")
            # agent_reply 一轮 >1 次 = 矛盾重生成（工具成功后的重生成因 reply_calls==1 无法识别，已知局限）
            if reply_calls > 1:
                regen_count += 1
            # 工具使用/降级归因：工具主回复 + 文本回退同现 = 降级回退
            if tool_calls_n > 0:
                tool_used_turns += 1
                if reply_calls > 0:
                    tool_degraded_turns += 1
            judge = await _judge_reply(patient, doctor_text, reply, list(messages), judge_enabled)
            messages.append(ReplayMessage("doctor", doctor_text))
            messages.append(ReplayMessage("patient", reply))
            turns.append({
                "turn": i, "doctor": doctor_text, "reply": reply,
                "latency_ms": latency_ms, "llm_calls": len(turn_calls),
                "trust": round(memory.trust, 3), "emotion": memory.emotion, "stage": memory.stage,
                "judge": judge,
            })
        return {
            "turns": turns,
            "fact_extract": {"latency_ms": init_ms, "llm_calls": init_calls, "fact_count": len(facts)},
            "regen_count": regen_count,
            "coverage": build_coverage_report(memory),
            "tool_stats": {"tool_used_turns": tool_used_turns, "tool_degraded_turns": tool_degraded_turns},
        }
    finally:
        settings.ENABLE_PATIENT_TOOL_USE = original_tool_flag


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
    if "tool_stats" in arm:
        ts = arm["tool_stats"]
        used = ts["tool_used_turns"]
        summary["tool_used_turns"] = used
        summary["tool_degraded_turns"] = ts["tool_degraded_turns"]
        # 近似：工具降级轮 / 工具使用轮（regen-after-tool-success 会被漏计，已知局限）
        summary["tool_degrade_rate"] = round(ts["tool_degraded_turns"] / used, 3) if used else 0.0
    _summarize_judge(turns, summary)
    return summary


def _summarize_judge(turns: list[dict], summary: dict) -> None:
    """把逐轮 judge 分聚合进 summary（无 judge 数据则跳过）"""
    judged = [t["judge"] for t in turns if t.get("judge")]
    if not judged:
        return
    valid = [j for j in judged if not j.get("degraded")]
    summary["judge_degraded_count"] = sum(1 for j in judged if j.get("degraded"))
    summary["judge_scored_turns"] = len(valid)
    overalls = [j["overall"] for j in valid if j.get("overall") is not None]
    summary["judge_overall_avg"] = round(statistics.mean(overalls), 3) if overalls else None
    for dim in DIMENSIONS:
        vals = [j[dim] for j in valid if j.get(dim) is not None]
        summary[f"judge_{dim}_avg"] = round(statistics.mean(vals), 3) if vals else None


def _build_manifest(results: list[dict], args, ts: str) -> dict:
    """根据回放参数构建 ReportManifest 字典。"""
    from evaluation.report_schema import ReportKind

    n_cases = len(results)
    # 冒烟判定：关闭 judge 或病例数 < 18
    if not args.judge or n_cases < 18:
        kind = ReportKind.SMOKE
    else:
        kind = ReportKind.REGRESSION

    return {
        "report_kind": kind.value,
        "report_id": f"ab_{ts}",
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "case_count": n_cases,
        "dataset_version": "patient_sim_v1",
        "model_version": settings.llm_model,
        "prompt_version": "v1",
        "judge_version": "judge_v1" if args.judge else "disabled",
        "kb_version": getattr(settings, "ACTIVE_INDEX_VERSION", "unknown"),
        "scoring_policy_version": "scoring_v1",
        "seed": None,
    }


def _write_report(out_path: Path, results: list[dict], failed: list[dict], args, ts: str) -> None:
    """原子写回放报告（tmp + replace），进程中途被杀也不会写坏 JSON。"""
    report = {
        "manifest": _build_manifest(results, args, ts),
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "model": settings.llm_model,
        "turns_cap": args.turns_cap,
        "judge_enabled": args.judge,
        "arms": list(ARMS),
        "cases": results,
        "failed": failed,
    }
    out_path.parent.mkdir(parents=True, exist_ok=True)
    tmp = out_path.with_suffix(".json.tmp")
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)
    tmp.replace(out_path)


def _write_badcases(bad_path: Path, results: list[dict]) -> int:
    """原子写 badcase JSONL 归档；返回条数（0 条时删除残留文件）。"""
    all_badcases = [bc for r in results for bc in r.get("badcases", [])]
    if not all_badcases:
        if bad_path.exists():
            bad_path.unlink()
        return 0
    tmp = bad_path.with_suffix(".jsonl.tmp")
    with open(tmp, "w", encoding="utf-8") as f:
        for bc in all_badcases:
            f.write(json.dumps(bc, ensure_ascii=False) + "\n")
    tmp.replace(bad_path)
    return len(all_badcases)


def _collect_badcases(case_id: str, patient: ReplayPatient, arms_data: dict) -> list[dict]:
    """任一 judge 维度 ≤2 或 overall<3 的轮次归档（降级轮不纳入，分不可信）"""
    badcases = []
    for arm_name, arm in arms_data.items():
        for t in arm["turns"]:
            j = t.get("judge")
            if not j or j.get("degraded"):
                continue
            dim_vals = [j.get(d) for d in DIMENSIONS if j.get(d) is not None]
            overall = j.get("overall")
            is_bad = any(v <= 2 for v in dim_vals) or (overall is not None and overall < 3)
            if not is_bad:
                continue
            badcases.append({
                "case_id": case_id,
                "personality": patient.personality_type,
                "diagnosis": patient.expected_diagnosis,
                "turn": t["turn"],
                "arm": arm_name,
                "doctor": t["doctor"],
                "reply": t["reply"],
                "scores": {d: j.get(d) for d in DIMENSIONS},
                "overall": overall,
                "reason": j.get("reason", ""),
                "attribution": "",  # 归因标签占位，人工回填
            })
    return badcases


async def run_case(case_id: str, turns_cap: int, counter: LLMCounter,
                   judge_enabled: bool = False) -> dict | None:
    loaded = load_case(case_id)
    if loaded is None:
        return None
    patient, doctor_turns = loaded
    doctor_turns = doctor_turns[:turns_cap]
    print(f"\n=== {case_id} | {patient.personality_type} | {len(doctor_turns)} 轮 | 主诉: {patient.chief_complaint[:30]} ===")

    legacy = await run_legacy_arm(patient, doctor_turns, counter, judge_enabled)
    print(f"  [legacy      ] 完成 {len(legacy['turns'])} 轮")
    agent_ledger = await run_agent_arm(patient, doctor_turns, counter, use_tools=False, judge_enabled=judge_enabled)
    print(f"  [agent_ledger] 完成 {len(agent_ledger['turns'])} 轮，披露率 {agent_ledger['coverage']['disclosure_rate']:.0%}，重生成 {agent_ledger['regen_count']} 次")
    agent_tool = await run_agent_arm(patient, doctor_turns, counter, use_tools=True, judge_enabled=judge_enabled)
    ts = agent_tool["tool_stats"]
    print(f"  [agent_tool  ] 完成 {len(agent_tool['turns'])} 轮，披露率 {agent_tool['coverage']['disclosure_rate']:.0%}，工具用 {ts['tool_used_turns']} 轮，降级 {ts['tool_degraded_turns']} 轮")

    arms_data = {"legacy": legacy, "agent_ledger": agent_ledger, "agent_tool": agent_tool}
    badcases = _collect_badcases(case_id, patient, arms_data)

    return {
        "case_id": case_id,
        "personality": patient.personality_type,
        "expected_diagnosis": patient.expected_diagnosis,
        "legacy": {"summary": summarize_arm(legacy), "detail": legacy},
        "agent_ledger": {"summary": summarize_arm(agent_ledger), "detail": agent_ledger},
        "agent_tool": {"summary": summarize_arm(agent_tool), "detail": agent_tool},
        "badcases": badcases,
    }


async def main():  # noqa: C901  # CLI 编排入口，参数解析与分支较多，复杂度超标可接受
    parser = argparse.ArgumentParser(description="患者智能体三臂回放验证")
    parser.add_argument("--cases", default="", help="逗号分隔的病例目录名；传 @eval_set 读固化评测集；为空时按 --limit 取")
    parser.add_argument("--limit", type=int, default=2, help="未指定 --cases 时按目录名排序取前 N 例；配 @eval_set 时显式传入则裁前 N 例")
    parser.add_argument("--turns-cap", type=int, default=10, help="每例最多回放的医生提问轮数（控成本，≤14 可避开早期摘要调用）")
    parser.add_argument("--judge", dest="judge", action="store_true", default=True, help="启用 LLM-as-Judge 软分（默认开）")
    parser.add_argument("--no-judge", dest="judge", action="store_false", help="关闭 Judge 只跑硬指标（省额度）")
    parser.add_argument("--resume", default="", help="从已有报告 JSON 路径续跑：载入已完成病例并跳过，仅跑剩余病例")
    args = parser.parse_args()

    # 离线环境无 Redis：禁用 token_tracker，避免每次 LLM 调用后 3s 连接超时
    async def _noop_record_usage(*_args, **_kwargs):
        return None
    token_tracker_mod.token_tracker.record_usage = _noop_record_usage

    limit_explicit = any(a == "--limit" or a.startswith("--limit=") for a in sys.argv)
    if args.cases == "@eval_set":
        case_ids = [c.case_id for c in load_eval_set(DEFAULT_EVAL_SET)]
        if limit_explicit:
            case_ids = case_ids[: args.limit]
    elif args.cases:
        case_ids = [c.strip() for c in args.cases.split(",") if c.strip()]
    else:
        case_ids = sorted(p.name for p in DATASET_DIR.iterdir() if p.is_dir())[: args.limit]

    print(f"模型: {settings.llm_model} | 病例数: {len(case_ids)} | 轮数上限: {args.turns_cap} | Judge: {'开' if args.judge else '关'}")

    counter = LLMCounter()
    counter.install()

    ts = datetime.now().strftime('%Y%m%d_%H%M%S')
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    out_path = REPORT_DIR / f"ab_{ts}.json"
    bad_path = REPORT_DIR / f"badcase_{ts}.jsonl"

    results: list[dict] = []
    failed: list[dict] = []
    done_ids: set[str] = set()

    # 续跑：从既有报告载入已完成病例，跳过重复（配额不稳时可多次接力跑完）
    if args.resume:
        resume_path = Path(args.resume)
        if resume_path.exists():
            prior = json.load(open(resume_path, encoding="utf-8"))
            results = prior.get("cases", [])
            failed = prior.get("failed", [])
            done_ids = {r["case_id"] for r in results}
            # 剔除历史失败记录中将被重跑的病例，避免重跑成功后仍挂在 failed 里
            # （成功即进 done_ids/cases；仍失败会重新 append，簿记不重复）
            failed = [f for f in failed if f["case_id"] in done_ids]
            print(f"[续跑] 从 {resume_path.name} 载入 {len(results)} 例已完成，将跳过它们")
        else:
            print(f"[续跑] 指定报告不存在，忽略: {resume_path}")

    def _flush():
        _write_report(out_path, results, failed, args, ts)
        _write_badcases(bad_path, results)

    try:
        for case_id in case_ids:
            if case_id in done_ids:
                print(f"[跳过] 已完成: {case_id}")
                continue
            try:
                result = await run_case(case_id, args.turns_cap, counter, judge_enabled=args.judge)
            except Exception as exc:  # 单例失败不拖垮整批：记录后继续，已完成进度已落盘
                import traceback
                print(f"  [错误] {case_id} 回放失败: {type(exc).__name__}: {exc}")
                traceback.print_exc()
                failed.append({"case_id": case_id, "error": f"{type(exc).__name__}: {exc}"})
                _flush()
                continue
            if result is not None:
                results.append(result)
            _flush()  # 每例落盘，崩溃/中断也保留已完成进度
    finally:
        counter.uninstall()

    _flush()
    n_bad = _write_badcases(bad_path, results)

    # 控制台汇总表
    print("\n" + "=" * 108)
    print(f"{'case':<16}{'臂':<14}{'轮':>3}{'LLM':>5}{'均延迟ms':>9}{'P95ms':>7}{'均字数':>7}{'Judge':>7}  账本/工具指标")
    for r in results:
        for arm_name in ARMS:
            s = r[arm_name]["summary"]
            judge_str = f"{s['judge_overall_avg']:.2f}" if s.get("judge_overall_avg") is not None else "-"
            extra = ""
            if arm_name in ("agent_ledger", "agent_tool"):
                extra = f"披露{s['disclosure_rate']:.0%} 漏{s['undisclosed_count']} 重生成{s['regen_count']}"
                if arm_name == "agent_tool":
                    extra += f" 工具用{s.get('tool_used_turns', 0)} 降级率{s.get('tool_degrade_rate', 0):.0%}"
            print(f"{r['case_id']:<16}{arm_name:<14}{s['turns']:>3}{s['total_llm_calls']:>5}"
                  f"{s['latency_avg_ms']:>9}{s['latency_p95_ms']:>7}{s['reply_chars_avg']:>7}{judge_str:>7}  {extra}")
    print(f"\n报告已写入: {out_path}（完成 {len(results)} 例，失败 {len(failed)} 例）")
    if n_bad:
        print(f"Badcase 归档: {bad_path}（{n_bad} 条）")
    if failed:
        print("失败病例（可用 --resume 接着跑）:")
        for fc in failed:
            print(f"  - {fc['case_id']}: {fc['error']}")


if __name__ == "__main__":
    asyncio.run(main())
