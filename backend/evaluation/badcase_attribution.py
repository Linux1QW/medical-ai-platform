# -*- coding: utf-8 -*-
"""Badcase 失败模式归因（规则式，无 LLM 成本、可复现）。

依据 patient_judge 的四维评分，把每条 badcase 归到一个「主失败模式」，并列出全部
触发模式。判定阈值与 ab_patient_replay._collect_badcases 保持一致：维度分
≤ BAD_DIM_THRESHOLD 即视为该维度破线。

之所以按评委已给出的维度分做规则归因（而非再调 LLM）：确定、可复现、零额度成本，
且与 badcase 的触发口径天然对齐，便于回归复盘。
"""

# 维度 -> 失败模式标签（键为英文 mode key，值为中文可读标签）
DIMENSION_MODE = {
    "role_consistency": "角色一致性破坏",
    "medical_plausibility": "医学/上下文合理性不足",
    "naturalness": "生硬失真(AI腔)",
    "disclosure_timing": "披露时机失当",
}

# 主模式仲裁优先级：分数并列时，越靠前越优先归为主因。
# 角色跳戏直接破坏模拟有效性最严重，其次医学/上下文矛盾，再次披露时机，最后自然度。
MODE_PRIORITY = ["role_consistency", "medical_plausibility", "disclosure_timing", "naturalness"]

BAD_DIM_THRESHOLD = 2
LOW_OVERALL = "low_overall"
LOW_OVERALL_LABEL = "综合分偏低(无单维破线)"


def classify_badcase(record: dict, threshold: int = BAD_DIM_THRESHOLD) -> dict:
    """归因单条 badcase：返回 {attribution, attribution_label, modes}。

    modes 为全部破线维度（按 MODE_PRIORITY 顺序）；attribution 取分最低者，
    并列时按 MODE_PRIORITY 仲裁。无单维破线（仅 overall 触发）时归 low_overall。
    """
    scores = record.get("scores") or {}
    triggered = [
        d for d in MODE_PRIORITY
        if isinstance(scores.get(d), (int, float)) and not isinstance(scores.get(d), bool)
        and scores[d] <= threshold
    ]
    if not triggered:
        return {"attribution": LOW_OVERALL, "attribution_label": LOW_OVERALL_LABEL, "modes": []}
    primary = min(triggered, key=lambda d: (scores[d], MODE_PRIORITY.index(d)))
    return {
        "attribution": primary,
        "attribution_label": DIMENSION_MODE[primary],
        "modes": triggered,
    }


def label_badcases(records: list[dict], threshold: int = BAD_DIM_THRESHOLD) -> list[dict]:
    """为每条 badcase 填充 attribution / attribution_label / attribution_modes（不改原对象）。"""
    out = []
    for rec in records:
        r = dict(rec)
        cls = classify_badcase(rec, threshold)
        r["attribution"] = cls["attribution"]
        r["attribution_label"] = cls["attribution_label"]
        r["attribution_modes"] = cls["modes"]
        out.append(r)
    return out


def summarize_badcases(records: list[dict], threshold: int = BAD_DIM_THRESHOLD) -> dict:
    """聚合失败模式清单（去标识：不含 doctor/reply 对话文本，可外发/入档）。"""
    labeled = label_badcases(records, threshold)
    by_mode: dict[str, int] = {}
    by_arm: dict[str, dict[str, int]] = {}
    by_personality: dict[str, int] = {}
    entries = []
    for r in labeled:
        attr = r["attribution"]
        by_mode[attr] = by_mode.get(attr, 0) + 1
        arm = r.get("arm", "?")
        by_arm.setdefault(arm, {})
        by_arm[arm][attr] = by_arm[arm].get(attr, 0) + 1
        p = r.get("personality", "?")
        by_personality[p] = by_personality.get(p, 0) + 1
        entries.append({
            "case_id": r.get("case_id"),
            "arm": arm,
            "turn": r.get("turn"),
            "personality": p,
            "attribution": attr,
            "attribution_label": r["attribution_label"],
            "modes": r["attribution_modes"],
            "scores": r.get("scores"),
            "overall": r.get("overall"),
        })
    return {
        "total": len(labeled),
        "threshold": threshold,
        "by_mode": dict(sorted(by_mode.items(), key=lambda kv: -kv[1])),
        "by_arm": by_arm,
        "by_personality": by_personality,
        "entries": entries,
    }
