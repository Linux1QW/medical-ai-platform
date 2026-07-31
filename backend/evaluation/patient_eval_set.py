# -*- coding: utf-8 -*-
"""患者模拟评测集：分层抽样 + 加载校验。

评测集固化为 JSONL：首行 {"_meta": {...}}，其余每行一条 EvalCase。
分层抽样按 (人格类型) 轮转 + (诊断) 多样性排序，确定性种子可复现。
"""
import json
import logging
from pathlib import Path
from typing import Any, Optional

from pydantic import BaseModel

logger = logging.getLogger(__name__)

# 人格归一化映射（与 scripts/seed_patients.PERSONALITY_MAP 保持一致；缺省 -> 配合型）
_PERSONALITY_MAP = {
    "合作": "配合型",
    "偏执": "对抗型",
    "啰嗦": "焦虑型",
    "怀疑": "沉默型",
}
_DEFAULT_PERSONALITY = "配合型"


class EvalCase(BaseModel):
    """评测集单条病例元数据。"""

    case_id: str
    personality: str
    diagnosis: str
    turns_available: int


def normalize_personality(raw: Any) -> str:
    """原始『性格』字段 -> 归一化人格类型。"""
    return _PERSONALITY_MAP.get(str(raw or "").strip(), _DEFAULT_PERSONALITY)


def scan_dataset(dataset_dir) -> list[dict]:
    """遍历 dataset 目录，抽取每例元数据；跳过缺主 JSON 或门诊对话为空的例。"""
    root = Path(dataset_dir)
    records: list[dict] = []
    for d in sorted(root.iterdir(), key=lambda p: p.name):
        if not d.is_dir():
            continue
        main_json = d / f"{d.name}.json"
        if not main_json.exists():
            continue
        try:
            data = json.loads(main_json.read_text(encoding="utf-8"))
        except Exception:
            logger.warning("跳过无法解析的病例: %s", d.name)
            continue
        dialogue = data.get("门诊对话") or []
        doctor_turns = [
            t for t in dialogue
            if isinstance(t, dict) and str(t.get("医生", "")).strip()
        ]
        if not doctor_turns:
            continue
        persona = data.get("人格") or {}
        records.append({
            "case_id": d.name,
            "personality": normalize_personality(persona.get("性格")),
            "diagnosis": str(data.get("主诊断", "")).strip() or "未知",
            "turns_available": len(doctor_turns),
        })
    return records


def _order_by_diagnosis_diversity(records: list[dict]) -> list[dict]:
    """在组内重排，使相邻选取尽量落在不同诊断上（贪心去重）。"""
    remaining = list(records)
    ordered: list[dict] = []
    seen_diag: set[str] = set()
    while remaining:
        # 优先取一个诊断尚未出现过的
        pick_idx = next(
            (i for i, r in enumerate(remaining) if r["diagnosis"] not in seen_diag),
            0,
        )
        chosen = remaining.pop(pick_idx)
        ordered.append(chosen)
        seen_diag.add(chosen["diagnosis"])
        if len(seen_diag) == len({r["diagnosis"] for r in ordered} | {r["diagnosis"] for r in remaining}):
            # 所有诊断已至少出现一次，重置以便下一轮继续多样化
            seen_diag = set()
    return ordered


def stratified_sample(records: list[dict], n: int, seed: int) -> list[dict]:
    """按人格分层轮转抽样，组内按诊断多样性排序；确定性可复现。

    - 先按 case_id 排序保证输入顺序确定
    - 各人格组内用固定种子洗牌后按诊断多样性重排
    - 跨人格轮转取样，保证每种人格尽量被覆盖
    - n 超过总数时返回全部（去重）
    """
    import random

    by_p: dict[str, list[dict]] = {}
    for r in sorted(records, key=lambda x: x["case_id"]):
        by_p.setdefault(r["personality"], []).append(r)

    rng = random.Random(seed)
    pools: dict[str, list[dict]] = {}
    for p in sorted(by_p):
        recs = by_p[p][:]
        rng.shuffle(recs)
        pools[p] = _order_by_diagnosis_diversity(recs)

    personalities = sorted(pools)
    idx = {p: 0 for p in personalities}
    selected: list[dict] = []
    total = sum(len(v) for v in pools.values())
    target = min(n, total)
    while len(selected) < target:
        progressed = False
        for p in personalities:
            if len(selected) >= target:
                break
            if idx[p] < len(pools[p]):
                selected.append(pools[p][idx[p]])
                idx[p] += 1
                progressed = True
        if not progressed:
            break
    return selected


def load_eval_set(path) -> list[EvalCase]:
    """加载评测集 JSONL：跳过 _meta 行，按 case_id 去重（保留首次），逐行校验 schema。"""
    cases: list[EvalCase] = []
    seen: set[str] = set()
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            if "_meta" in row:
                continue
            case = EvalCase(**row)
            if case.case_id in seen:
                continue
            seen.add(case.case_id)
            cases.append(case)
    return cases


def read_meta(path) -> Optional[dict]:
    """读取评测集首行的 _meta（若无则返回 None）。"""
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            return row.get("_meta") if "_meta" in row else None
    return None
