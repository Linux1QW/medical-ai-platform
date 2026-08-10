"""Deterministic multi-agent coach graph.

Pipeline: load_context → intent → memory → safety → planner → evidence → draft → critic → finalize → persist

Thread identity: coach:{session_uuid}
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any, Callable
from uuid import UUID, uuid4

from app.agent_runtime.contracts import (
    CoachContextView,
    CoachIntent,
    CoachSuggestion,
    InterviewStage,
)
from app.agent_runtime.context import CompiledContext, compile_coach_context
from app.agent_runtime.telemetry import AgentEventRecorder
from app.services.agents.coach.intent import classify_intent
from app.services.agents.coach.planner import (
    INTENT_TO_STAGE,
    FollowupPlan,
    build_followup_plan,
)
from app.services.memory.working import WorkingMemoryState, validate_memory_sources


@dataclass
class CoachGraphState:
    """Shared state flowing through the coach graph nodes."""

    # Input
    context_view: CoachContextView
    latest_message: str = ""
    turn_no: int = 1
    session_id: UUID = field(default_factory=uuid4)

    # Intermediate
    intent: CoachIntent | None = None
    compiled_context: CompiledContext | None = None
    working_memory: WorkingMemoryState = field(default_factory=WorkingMemoryState)
    followup_plan: FollowupPlan | None = None
    evidence_results: list[dict[str, Any]] = field(default_factory=list)
    draft_suggestion: CoachSuggestion | None = None

    # Output
    final_suggestion: CoachSuggestion | None = None
    blocked: bool = False
    block_reason: str = ""

    # Trace
    node_trace: list[dict[str, Any]] = field(default_factory=list)


class CoachGraph:
    """Deterministic multi-agent coach graph.

    Nodes are executed in fixed order. Each node receives the state
    and returns a (possibly modified) state. No LLM calls are made
    in the default implementation — this is a deterministic pipeline.
    """

    def __init__(
        self,
        *,
        recorder: AgentEventRecorder | None = None,
        evidence_fn: Callable[[str, str], list[dict[str, Any]]] | None = None,
    ) -> None:
        self.recorder = recorder
        self.evidence_fn = evidence_fn  # Injectable for testing
        self._nodes: list[tuple[str, Callable[[CoachGraphState], CoachGraphState]]] = [
            ("load_context", self._load_context),
            ("intent", self._intent_node),
            ("memory", self._memory_node),
            ("safety", self._safety_node),
            ("planner", self._planner_node),
            ("evidence", self._evidence_node),
            ("draft", self._draft_node),
            ("critic", self._critic_node),
            ("finalize", self._finalize_node),
            ("persist", self._persist_node),
        ]

    async def run(self, state: CoachGraphState) -> CoachGraphState:
        """Execute the full coach graph pipeline."""
        for node_name, node_fn in self._nodes:
            if state.blocked:
                break
            state.node_trace.append({"node": node_name, "status": "started"})
            try:
                state = node_fn(state)
                state.node_trace.append({"node": node_name, "status": "completed"})
            except Exception as e:
                state.node_trace.append(
                    {"node": node_name, "status": "error", "error": str(e)}
                )
                state.blocked = True
                state.block_reason = f"Node '{node_name}' failed: {e}"
        return state

    # ── Node implementations ──────────────────────────────────────────

    def _load_context(self, state: CoachGraphState) -> CoachGraphState:
        """Node: Compile context from the context view."""
        state.compiled_context = compile_coach_context(state.context_view)
        return state

    def _intent_node(self, state: CoachGraphState) -> CoachGraphState:
        """Node: Classify the doctor's intent."""
        state.intent = classify_intent(state.latest_message)
        return state

    def _memory_node(self, state: CoachGraphState) -> CoachGraphState:
        """Node: Update working memory with visible messages."""
        if state.intent:
            stage = INTENT_TO_STAGE.get(state.intent, "off_topic")
            if stage not in state.working_memory.stage_history:
                state.working_memory.stage_history.append(stage)
            state.working_memory.mark_asked(state.intent, state.turn_no)
        return state

    def _safety_node(self, state: CoachGraphState) -> CoachGraphState:
        """Node: Safety checks — block unsafe intents and hidden-fact leakage."""
        if state.intent == "unsafe":
            state.blocked = True
            state.block_reason = "Unsafe intent detected"
            return state

        # Check for hidden-fact leakage via working memory validation
        visible_sequences = {msg.sequence for msg in state.context_view.messages}
        try:
            validate_memory_sources(state.working_memory, visible_sequences)
        except Exception:
            state.blocked = True
            state.block_reason = "Hidden context violation in working memory"
        return state

    def _planner_node(self, state: CoachGraphState) -> CoachGraphState:
        """Node: Build follow-up plan."""
        if state.intent:
            state.followup_plan = build_followup_plan(
                current_intent=state.intent,
                working_memory=state.working_memory,
                current_turn=state.turn_no,
            )
        return state

    def _evidence_node(self, state: CoachGraphState) -> CoachGraphState:
        """Node: Gather evidence from skills (injectable)."""
        if self.evidence_fn and state.intent:
            state.evidence_results = self.evidence_fn(state.intent, state.latest_message)
        return state

    def _draft_node(self, state: CoachGraphState) -> CoachGraphState:
        """Node: Draft a suggestion based on intent, plan, and evidence."""
        if not state.intent or not state.followup_plan:
            return state

        recommended = state.followup_plan.recommended_intent or state.intent

        # Map intent to stage
        stage: InterviewStage = INTENT_TO_STAGE.get(state.intent, "off_topic")  # type: ignore[assignment]

        # Build rationale from plan
        rationale = f"Intent: {state.intent}"
        if state.followup_plan.candidates:
            rationale += f" | Plan: {state.followup_plan.candidates[0].rationale}"

        state.draft_suggestion = CoachSuggestion(
            suggestion_id=uuid4(),
            session_id=state.session_id,
            turn_no=state.turn_no,
            intent=state.intent,
            stage=stage,
            suggested_question=f"[Coach] Consider asking about: {recommended}",
            rationale_summary=rationale[:300],
            confidence=0.7,
            risk_level="low",
        )
        return state

    def _critic_node(self, state: CoachGraphState) -> CoachGraphState:
        """Node: Critic checks for hidden-fact leakage, diagnostic phrasing, repeated questions, unsupported citations."""
        if not state.draft_suggestion:
            return state

        suggestion = state.draft_suggestion

        # Check 1: Hidden-fact leakage
        hidden_patterns = ["expected_diagnosis", "gold_standard", "hidden_fact", "unrevealed"]
        for pattern in hidden_patterns:
            if (
                pattern in suggestion.suggested_question.lower()
                or pattern in suggestion.rationale_summary.lower()
            ):
                state.blocked = True
                state.block_reason = f"Critic: hidden-fact leakage detected (pattern: {pattern})"
                return state

        # Check 2: No diagnostic phrasing
        diagnostic_patterns = [r"你应该诊断", r"这是.*病", r"确诊为", r"处方"]
        for pattern in diagnostic_patterns:
            if re.search(pattern, suggestion.suggested_question):
                state.blocked = True
                state.block_reason = "Critic: diagnostic phrasing detected"
                return state

        # Check 3: No repeated questions (lower confidence, don't block)
        if state.intent and state.working_memory.is_repeated(
            state.intent, within_turns=1, current_turn=state.turn_no
        ):
            suggestion.confidence = max(0.1, suggestion.confidence - 0.3)

        # Check 4: Unsupported citations
        if suggestion.citation_ids and not state.evidence_results:
            suggestion.citation_ids = []

        return state

    def _finalize_node(self, state: CoachGraphState) -> CoachGraphState:
        """Node: Finalize the suggestion."""
        state.final_suggestion = state.draft_suggestion
        return state

    def _persist_node(self, state: CoachGraphState) -> CoachGraphState:
        """Node: Persist decision (telemetry recording)."""
        # Trace is already captured in node_trace
        return state
