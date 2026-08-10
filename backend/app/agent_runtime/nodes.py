"""Async node functions for the LangGraph Coach pipeline.

Each node receives the full CoachGraphState and returns a dict of partial
updates that LangGraph merges into the state.
"""
from __future__ import annotations

import re
from typing import Any
from uuid import uuid4

from app.agent_runtime.contracts import CoachSuggestion
from app.agent_runtime.context import COACH_SAFETY_POLICY, COACH_SYSTEM_PROMPT, compile_coach_context
from app.agent_runtime.critic import (
    DIAGNOSTIC_PATTERNS,
    HIDDEN_FACT_PATTERNS,
    CriticFinding,
    CriticResult,
)
from app.agent_runtime.state import (
    CoachGraphState,
    DraftSuggestion,
    EvidenceItem,
    IntentDecision,
)
from app.services.agents.coach.intent import INTENT_RULES, classify_intent
from app.services.agents.coach.planner import (
    INTENT_TO_STAGE,
    FollowupPlan,
    build_followup_plan,
)
from app.services.memory.working import WorkingMemoryState


# ── Intent node ──────────────────────────────────────────────────────────────


async def intent_node(state: CoachGraphState, *, gateway: Any = None) -> dict[str, Any]:
    """Classify intent: rules handle obvious cases, model fallback."""
    msg = state.latest_message
    # Rules-first pass (fast, deterministic)
    msg_lower = msg.lower()
    for keywords, intent in INTENT_RULES:
        if any(kw in msg_lower for kw in keywords):
            return {
                "intent_result": IntentDecision(intent=intent, confidence=0.95),
                "trace_refs": ["intent:rules"],
            }

    # Handle unsafe keyword blocklist
    unsafe_keywords = ["自杀", "杀人", "伤害", "自残"]
    if any(kw in msg_lower for kw in unsafe_keywords):
        return {
            "intent_result": IntentDecision(intent="unsafe", confidence=0.99),
            "trace_refs": ["intent:unsafe_rules"],
        }

    # Model fallback
    if gateway is not None:
        try:
            from app.agent_runtime.state import IntentDecision as ID

            decision = await gateway.complete_structured(
                messages=[
                    {"role": "system", "content": "分类医生消息的问诊意图。返回JSON: {\"intent\": \"...\", \"confidence\": 0.0}"},
                    {"role": "user", "content": msg},
                ],
                schema=ID,
                max_tokens=256,
            )
            return {"intent_result": decision, "trace_refs": ["intent:model"]}
        except Exception:
            pass

    return {
        "intent_result": IntentDecision(intent="off_topic", confidence=0.5),
        "trace_refs": ["intent:fallback"],
    }


# ── Planner node ─────────────────────────────────────────────────────────────


async def planner_node(state: CoachGraphState, **_: Any) -> dict[str, Any]:
    """Build a follow-up plan from context and working memory."""
    if state.intent_result is None:
        return {"plan": None}

    intent = state.intent_result.intent

    if intent == "unsafe" or intent == "off_topic":
        return {"plan": None, "trace_refs": ["planner:skip_unsafe"]}

    # Reconstruct working memory from asked_dimensions snapshot
    wm = WorkingMemoryState()
    wm.asked_dimensions = dict(state.asked_dimensions)

    plan = build_followup_plan(
        current_intent=intent,
        working_memory=wm,
        current_turn=state.turn,
    )
    return {"plan": plan, "trace_refs": ["planner:built"]}


# ── Evidence node ────────────────────────────────────────────────────────────


async def evidence_node(state: CoachGraphState, *, evidence_fn: Any = None, **_: Any) -> dict[str, Any]:
    """Retrieve evidence via skill executor (RAG)."""
    if evidence_fn is None or state.intent_result is None:
        return {"evidence": [], "trace_refs": ["evidence:skip"]}

    try:
        raw_items = evidence_fn(state.intent_result.intent, state.latest_message)
        items = [
            EvidenceItem(
                source=item.get("source", "unknown"),
                text=item.get("text", "")[:2000],
                score=float(item.get("score", 0.5)),
                doc_id=str(item.get("doc_id", "")),
            )
            for item in raw_items
        ]
        return {"evidence": items, "trace_refs": ["evidence:retrieved"]}
    except Exception:
        return {"evidence": [], "trace_refs": ["evidence:error"]}


# ── Draft node ───────────────────────────────────────────────────────────────


async def draft_node(state: CoachGraphState, *, gateway: Any = None, **_: Any) -> dict[str, Any]:
    """Draft a natural Chinese follow-up question."""
    if state.intent_result is None or state.plan is None:
        return {"draft": None, "trace_refs": ["draft:skip"]}

    intent = state.intent_result.intent
    stage = INTENT_TO_STAGE.get(intent, "off_topic")

    # Build context for the model
    compiled = compile_coach_context(state.context, state.budget)
    context_lines = [
        COACH_SYSTEM_PROMPT,
        COACH_SAFETY_POLICY,
    ]
    if compiled.dialogue_section:
        context_lines.append(f"对话历史:\n{compiled.dialogue_section}")
    if compiled.memory_section:
        context_lines.append(f"教学记忆:\n{compiled.memory_section}")

    recommended = state.plan.recommended_intent or intent
    evidence_text = ""
    if state.evidence:
        evidence_text = "\n参考证据:\n" + "\n".join(
            f"- [{e.source}] {e.text[:200]}" for e in state.evidence[:3]
        )

    prompt = (
        f"当前意图: {intent}\n"
        f"推荐方向: {recommended}\n"
        f"{evidence_text}\n\n"
        f"请用自然中文生成一个问诊建议问题。不要做诊断，不要开处方。\n"
        f"返回JSON: {{\"question\": \"...\", \"rationale\": \"...\", \"confidence\": 0.0, \"risk_level\": \"low\"}}"
    )

    # Model draft
    if gateway is not None:
        try:
            from pydantic import BaseModel, Field

            class _DraftOut(BaseModel):
                question: str = Field(max_length=240)
                rationale: str = Field(max_length=300)
                confidence: float = Field(ge=0.0, le=1.0, default=0.7)
                risk_level: str = Field(default="low")

            messages = [
                {"role": "system", "content": "\n".join(context_lines)},
                {"role": "user", "content": prompt},
            ]
            result = await gateway.complete_structured(
                messages=messages, schema=_DraftOut, max_tokens=512
            )
            draft = DraftSuggestion(
                intent=intent,
                stage=stage,
                suggested_question=result.question,
                rationale_summary=result.rationale[:300],
                confidence=result.confidence,
                risk_level=result.risk_level if result.risk_level in ("low", "medium", "high") else "low",
                citation_ids=[e.doc_id for e in state.evidence[:3] if e.doc_id],
            )
            return {"draft": draft, "trace_refs": ["draft:model"]}
        except Exception:
            pass

    # Deterministic fallback
    question = f"[Coach] 建议您进一步了解: {recommended}"
    draft = DraftSuggestion(
        intent=intent,
        stage=stage,
        suggested_question=question,
        rationale_summary=f"Intent: {intent} | Plan: {recommended}",
        confidence=0.6,
        risk_level="low",
        citation_ids=[e.doc_id for e in state.evidence[:3] if e.doc_id],
    )
    return {"draft": draft, "trace_refs": ["draft:deterministic"]}


# ── Critic node ──────────────────────────────────────────────────────────────


async def critic_node(state: CoachGraphState, **_: Any) -> dict[str, Any]:
    """Deterministic safety rules first, then structured review.

    Schema failure → non-diagnostic educational fallback.
    """
    if state.draft is None:
        return {"critic_result": None, "blocked": False}

    findings: list[CriticFinding] = []
    draft = state.draft
    text_to_check = f"{draft.suggested_question} {draft.rationale_summary}".lower()

    # Check 1: Hidden-fact leakage (hard block)
    for pattern in HIDDEN_FACT_PATTERNS:
        if pattern in text_to_check:
            findings.append(CriticFinding(severity="error", category="hidden_leak", message=f"Hidden fact: {pattern}"))

    # Check 2: Diagnostic phrasing (hard block)
    for pattern in DIAGNOSTIC_PATTERNS:
        if re.search(pattern, draft.suggested_question):
            findings.append(CriticFinding(severity="error", category="diagnostic", message="Diagnostic phrasing"))

    # Check 3: Repetition against pre-turn memory (penalize once, don't block)
    is_repeat = False
    dim_key = draft.intent
    last_asked = state.asked_dimensions.get(dim_key)
    if last_asked is not None and (state.turn - last_asked) <= 1:
        is_repeat = True
        findings.append(CriticFinding(severity="warning", category="repeated", message="Repeated dimension"))

    has_errors = any(f.severity == "error" for f in findings)

    if has_errors:
        return {
            "critic_result": CriticResult(passed=False, findings=findings),
            "blocked": True,
            "block_reason": f"Critic: {findings[0].category}",
        }

    # Adjust confidence for repetition (penalize once)
    adjusted_confidence = draft.confidence
    if is_repeat:
        adjusted_confidence = max(0.1, draft.confidence - 0.3)

    # Check 4: Unsupported citations
    citation_ids = draft.citation_ids
    if citation_ids and not state.evidence:
        citation_ids = []

    # Update draft with adjusted values
    updated_draft = draft.model_copy(update={
        "confidence": adjusted_confidence,
        "citation_ids": citation_ids,
    })

    return {
        "draft": updated_draft,
        "critic_result": CriticResult(
            passed=True,
            findings=findings,
            adjusted_confidence=adjusted_confidence,
        ),
        "trace_refs": ["critic:passed"],
    }


# ── Finalize + persist nodes ─────────────────────────────────────────────────


async def finalize_node(state: CoachGraphState, **_: Any) -> dict[str, Any]:
    """Build final CoachSuggestion from the approved draft."""
    if state.draft is None or state.blocked:
        return {"status": "degraded" if state.blocked else "done"}

    draft = state.draft
    final = CoachSuggestion(
        suggestion_id=uuid4(),
        session_id=state.session_id,
        turn_no=state.turn,
        intent=draft.intent,
        stage=draft.stage,
        suggested_question=draft.suggested_question,
        rationale_summary=draft.rationale_summary,
        targeted_slots=draft.targeted_slots,
        citation_ids=draft.citation_ids,
        confidence=draft.confidence,
        risk_level=draft.risk_level,
    )

    # Mark the selected dimension as asked in the snapshot
    new_asked = dict(state.asked_dimensions)
    new_asked[draft.intent] = state.turn

    return {
        "final_suggestion": final,
        "asked_dimensions": new_asked,
        "status": "done",
        "trace_refs": ["finalize:done"],
    }


async def persist_node(state: CoachGraphState, **_: Any) -> dict[str, Any]:
    """Record trace references. Persistence handled by the caller."""
    refs = list(state.trace_refs)
    if "persist:done" not in refs:
        refs.append("persist:done")
    return {"trace_refs": refs, "status": state.status or "done"}
