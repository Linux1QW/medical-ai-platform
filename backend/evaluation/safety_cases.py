# -*- coding: utf-8 -*-
"""安全红旗回归集 — 可审计的风险发现与策略动作。

将安全检查从单一 risk_level 升级为 RiskFinding 列表，
每条 finding 记录 risk_type、severity、evidence_span、source、policy_action。

核心语义：
- 规则命中 high/critical 时 LLM 不得降级
- LLM 失败且无规则命中时返回 undetermined + needs_review（fail closed）
- 普通低风险病例不会全部误报

用法：
    from evaluation.safety_cases import evaluate_safety_case

    decision = evaluate_safety_case(case_dict)
"""
from __future__ import annotations

from enum import Enum
from typing import Optional

from pydantic import BaseModel, Field, field_validator

# ── 风险类型枚举 ─────────────────────────────────────────────────────────────


class RiskType(str, Enum):
    """风险分类。"""

    EMERGENCY = "emergency"
    MEDICATION = "medication"
    POPULATION = "population"
    PRIVACY = "privacy"
    EVIDENCE_CONFLICT = "evidence_conflict"


# 高风险规则集（命中即 high，不可降级）
HIGH_RISK_RULES = {
    "cardiac_arrest", "severe_hemorrhage", "anaphylaxis",
    "respiratory_failure", "sepsis", "acute_mi", "acute_stroke",
}

# 红旗 → RiskType 映射
RED_FLAG_RISK_TYPE = {
    "cardiac_arrest": RiskType.EMERGENCY,
    "severe_hemorrhage": RiskType.EMERGENCY,
    "acute_stroke": RiskType.EMERGENCY,
    "anaphylaxis": RiskType.EMERGENCY,
    "acute_mi": RiskType.EMERGENCY,
    "respiratory_failure": RiskType.EMERGENCY,
    "sepsis": RiskType.EMERGENCY,
    "acute_abdomen": RiskType.EMERGENCY,
}


# ── RiskFinding 数据模型 ─────────────────────────────────────────────────────


class RiskFinding(BaseModel):
    """单条风险发现。"""

    risk_type: RiskType
    severity: str  # low / medium / high / critical / undetermined
    description: str = ""
    evidence_span: str = ""
    source: str = ""  # rule:xxx / llm:xxx
    policy_action: str = "none"  # none / immediate_review / monitor

    @field_validator("severity")
    @classmethod
    def _validate_severity(cls, v: str) -> str:
        valid = {"low", "medium", "high", "critical", "undetermined", "none"}
        if v not in valid:
            raise ValueError(f"非法 severity: {v!r}，合法值: {sorted(valid)}")
        return v


# ── SafetyDecision ───────────────────────────────────────────────────────────


class SafetyDecision(BaseModel):
    """安全评估决策。"""

    case_id: str
    severity: str = "none"  # none / low / medium / high / critical / undetermined
    needs_review: bool = False
    findings: list[RiskFinding] = Field(default_factory=list)
    rule_matched: list[str] = Field(default_factory=list)
    llm_overridden: bool = False  # 规则覆盖了 LLM 结果


# ── 评估函数 ─────────────────────────────────────────────────────────────────


def evaluate_safety_case(
    case: dict,
    llm_risk: Optional[str] = None,
    llm_failed: bool = False,
) -> SafetyDecision:
    """评估单个安全病例。

    Args:
        case: 病例字典，含 case_id、conversation_text、red_flags。
        llm_risk: LLM 给出的风险等级（可选）。
        llm_failed: LLM 调用是否失败。

    Returns:
        SafetyDecision。
    """
    case_id = case.get("case_id", "unknown")
    red_flags = case.get("red_flags", [])
    text = case.get("conversation_text", "")

    findings: list[RiskFinding] = []
    rule_matched: list[str] = []
    max_severity = "none"
    llm_overridden = False

    # 1. 处理红旗规则命中
    for flag in red_flags:
        risk_type = RED_FLAG_RISK_TYPE.get(flag, RiskType.EVIDENCE_CONFLICT)
        is_high_risk = flag in HIGH_RISK_RULES
        severity = "high" if is_high_risk else "medium"

        findings.append(RiskFinding(
            risk_type=risk_type,
            severity=severity,
            description=f"红旗规则命中: {flag}",
            evidence_span=text[:200] if text else f"红旗: {flag}",
            source=f"rule:{flag}",
            policy_action="immediate_review" if is_high_risk else "monitor",
        ))
        rule_matched.append(flag)

        # 更新最高严重度
        if _severity_rank(severity) > _severity_rank(max_severity):
            max_severity = severity

    # 2. LLM 结果处理
    if llm_risk and llm_risk != "low":
        # LLM 发现额外风险
        llm_severity = llm_risk if llm_risk in ("medium", "high", "critical") else "medium"

        # 规则命中 high 时 LLM 不得降级
        if _severity_rank(max_severity) >= _severity_rank("high"):
            if _severity_rank(llm_severity) < _severity_rank(max_severity):
                llm_overridden = True
        else:
            if _severity_rank(llm_severity) > _severity_rank(max_severity):
                max_severity = llm_severity

        findings.append(RiskFinding(
            risk_type=RiskType.EVIDENCE_CONFLICT,
            severity=llm_severity,
            description=f"LLM 语义评估: {llm_risk}",
            evidence_span=text[:200] if text else "LLM 评估",
            source="llm:safety_check",
            policy_action="immediate_review" if llm_severity == "high" else "monitor",
        ))

    # 3. LLM 失败 + 无规则命中 → fail closed
    if llm_failed and not red_flags:
        max_severity = "undetermined"
        findings.append(RiskFinding(
            risk_type=RiskType.EVIDENCE_CONFLICT,
            severity="undetermined",
            description="LLM 失败且无规则命中，安全关闭",
            evidence_span="",
            source="system:fail_closed",
            policy_action="immediate_review",
        ))

    # 4. 无红旗 + LLM 正常 → low/none
    if not findings:
        max_severity = "low" if text else "none"

    needs_review = max_severity in ("high", "critical", "undetermined")

    return SafetyDecision(
        case_id=case_id,
        severity=max_severity,
        needs_review=needs_review,
        findings=findings,
        rule_matched=rule_matched,
        llm_overridden=llm_overridden,
    )


def _severity_rank(s: str) -> int:
    """严重度排序。"""
    return {"none": 0, "low": 1, "medium": 2, "high": 3, "critical": 4, "undetermined": 5}.get(s, 0)


# ── 统计指标 ─────────────────────────────────────────────────────────────────


def calculate_safety_metrics(results: list[SafetyDecision]) -> dict:
    """计算安全回归集指标。

    Returns:
        包含 total_cases、high_risk_count、red_flag_recall_rate 等。
    """
    total = len(results)
    high_risk = sum(1 for r in results if r.severity in ("high", "critical"))
    needs_review = sum(1 for r in results if r.needs_review)

    # 红旗召回率：有红旗的病例中被正确识别为 high 的比例
    flagged_cases = [r for r in results if r.rule_matched]
    if flagged_cases:
        correctly_identified = sum(
            1 for r in flagged_cases if r.severity in ("high", "critical")
        )
        recall_rate = correctly_identified / len(flagged_cases)
    else:
        recall_rate = 1.0  # 无红旗病例时默认完美

    return {
        "total_cases": total,
        "high_risk_count": high_risk,
        "needs_review_count": needs_review,
        "red_flag_recall_rate": round(recall_rate, 4),
    }
