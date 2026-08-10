"""LangGraph-based Coach Agent runtime.

Replaces the hand-written loop with a real LangGraph async StateGraph.
Pipeline: intent → planner → evidence → draft → critic → finalize → persist

Thread identity: coach:{session_uuid}
"""
from __future__ import annotations

import asyncio
import inspect
from typing import Any, TypedDict
from uuid import UUID, uuid4

from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END, StateGraph
from langgraph.graph.state import CompiledStateGraph

from app.agent_runtime.nodes import (
    critic_node,
    draft_node,
    evidence_node,
    finalize_node,
    intent_node,
    planner_node,
    persist_node,
)
from app.agent_runtime.state import CoachGraphState

# Re-export for backward compatibility
__all__ = [
    "CoachGraph",
    "CoachDependencies",
    "build_coach_graph",
    "invoke_coach_graph",
    "CoachGraphState",
]


# ── LangGraph TypedDict state schema ─────────────────────────────────────────


class CoachStateDict(TypedDict, total=False):
    """TypedDict schema for the LangGraph state channels.

    Each key becomes an independent channel, avoiding the __root__ conflict.
    """

    context: Any
    turn: int
    session_id: Any  # UUID
    latest_message: str
    asked_dimensions: dict
    intent_result: Any
    plan: Any
    evidence: list
    draft: Any
    critic_result: Any
    final_suggestion: Any
    budget: Any
    status: str
    trace_refs: list
    blocked: bool
    block_reason: str


# ── Dependency container ─────────────────────────────────────────────────────


class CoachDependencies:
    """Injectable dependencies for the coach graph."""

    def __init__(
        self,
        *,
        gateway: Any = None,
        evidence_fn: Any = None,
        timeout_seconds: int = 8,
    ) -> None:
        self.gateway = gateway
        self.evidence_fn = evidence_fn
        self.timeout_seconds = timeout_seconds


# ── Node wrappers (adapt async nodes to LangGraph dict-based state) ──────────


def _state_from_dict(d: dict[str, Any]) -> CoachGraphState:
    """Reconstruct a CoachGraphState from a LangGraph state dict."""
    from app.agent_runtime.context import ContextBudget

    raw_ctx: Any = d.get("context")
    raw_sid = d.get("session_id")
    state = CoachGraphState(
        context=raw_ctx,
        turn=d.get("turn", 1),
        session_id=raw_sid if raw_sid is not None else uuid4(),
        latest_message=d.get("latest_message", ""),
        asked_dimensions=d.get("asked_dimensions", {}),
        intent_result=d.get("intent_result"),
        plan=d.get("plan"),
        evidence=d.get("evidence", []),
        draft=d.get("draft"),
        critic_result=d.get("critic_result"),
        final_suggestion=d.get("final_suggestion"),
        budget=d.get("budget", ContextBudget()),
        status=d.get("status", "pending"),
        trace_refs=list(d.get("trace_refs", [])),
        blocked=d.get("blocked", False),
        block_reason=d.get("block_reason", ""),
    )
    return state


def _make_wrapper(node_fn: Any, deps: CoachDependencies) -> Any:
    """Create a LangGraph async wrapper for a node function.

    Inspects the node function signature to only pass accepted kwargs.
    """
    sig = inspect.signature(node_fn)
    params = sig.parameters
    has_var_keyword = any(
        p.kind == inspect.Parameter.VAR_KEYWORD for p in params.values()
    )
    accepted_kwargs = {
        name
        for name, p in params.items()
        if p.kind
        in (inspect.Parameter.KEYWORD_ONLY, inspect.Parameter.POSITIONAL_OR_KEYWORD)
        and name != "state"
    }

    async def wrapper(state: dict[str, Any]) -> dict[str, Any]:
        coach_state = _state_from_dict(state)
        all_kwargs: dict[str, Any] = {
            "gateway": deps.gateway,
            "evidence_fn": deps.evidence_fn,
        }
        if has_var_keyword:
            kwargs = all_kwargs
        else:
            kwargs = {k: v for k, v in all_kwargs.items() if k in accepted_kwargs}
        result = await node_fn(coach_state, **kwargs)
        # Merge trace_refs instead of replacing
        if "trace_refs" in result:
            existing = list(state.get("trace_refs", []))
            new_refs = result["trace_refs"]
            merged = existing + [r for r in new_refs if r not in existing]
            result["trace_refs"] = merged
        return result

    return wrapper


# ── Conditional edge: should we continue or short-circuit? ───────────────────


def _route_after_intent(state: dict[str, Any]) -> str:
    """If intent is unsafe/off_topic, skip to finalize."""
    intent_result = state.get("intent_result")
    if intent_result is None:
        return "finalize"
    if intent_result.intent in ("unsafe", "off_topic"):
        return "finalize"
    return "planner"


def _route_after_critic(state: dict[str, Any]) -> str:
    """Always go to finalize after critic."""
    return "finalize"


# ── Graph builder ────────────────────────────────────────────────────────────


def build_coach_graph(
    checkpointer: Any = None,
    dependencies: CoachDependencies | None = None,
) -> CompiledStateGraph:
    """Build and compile the LangGraph coach StateGraph.

    Args:
        checkpointer: LangGraph checkpointer for thread-based persistence.
                      If None, uses MemorySaver().
        dependencies: Injectable dependencies (gateway, evidence_fn, timeout).

    Returns:
        CompiledStateGraph ready for ainvoke().
    """
    if checkpointer is None:
        checkpointer = MemorySaver()
    if dependencies is None:
        dependencies = CoachDependencies()

    graph = StateGraph(CoachStateDict)

    # Add nodes with dependency injection
    graph.add_node("intent", _make_wrapper(intent_node, dependencies))
    graph.add_node("planner", _make_wrapper(planner_node, dependencies))
    graph.add_node("evidence", _make_wrapper(evidence_node, dependencies))
    graph.add_node("draft", _make_wrapper(draft_node, dependencies))
    graph.add_node("critic", _make_wrapper(critic_node, dependencies))
    graph.add_node("finalize", _make_wrapper(finalize_node, dependencies))
    graph.add_node("persist", _make_wrapper(persist_node, dependencies))

    # Entry point
    graph.set_entry_point("intent")

    # Conditional routing after intent
    graph.add_conditional_edges(
        "intent",
        _route_after_intent,
        {
            "planner": "planner",
            "finalize": "finalize",
        },
    )

    # Deterministic flow
    graph.add_edge("planner", "evidence")
    graph.add_edge("evidence", "draft")
    graph.add_edge("draft", "critic")
    graph.add_conditional_edges(
        "critic",
        _route_after_critic,
        {"finalize": "finalize"},
    )
    graph.add_edge("finalize", "persist")
    graph.add_edge("persist", END)

    return graph.compile(checkpointer=checkpointer)


# ── CoachGraph class wrapper ─────────────────────────────────────────────────


class CoachGraph:
    """Wrapper class for backward compatibility.

    coach_service.py imports CoachGraph as a class.
    This wraps build_coach_graph() to provide a class-based interface.
    """

    def __init__(self, checkpointer: Any = None, dependencies: CoachDependencies | None = None):
        self._compiled = build_coach_graph(checkpointer=checkpointer, dependencies=dependencies)

    async def ainvoke(self, state: dict[str, Any], config: Any = None) -> dict[str, Any]:
        return await self._compiled.ainvoke(state, config=config)

    @property
    def compiled(self) -> CompiledStateGraph:
        return self._compiled


# ── High-level invoke with hard timeout ──────────────────────────────────────


async def invoke_coach_graph(
    compiled_graph: CompiledStateGraph,
    *,
    context: Any,
    latest_message: str,
    turn: int = 1,
    session_id: UUID | None = None,
    asked_dimensions: dict[str, int] | None = None,
    thread_id: str | None = None,
    timeout_seconds: int = 8,
) -> dict[str, Any]:
    """Invoke the coach graph with a hard timeout.

    On timeout → status='degraded', no exception text in output.
    """
    if session_id is None:
        session_id = uuid4()
    if thread_id is None:
        thread_id = f"coach:{session_id}"

    initial_state: dict[str, Any] = {
        "context": context,
        "turn": turn,
        "session_id": session_id,
        "latest_message": latest_message,
        "asked_dimensions": asked_dimensions or {},
        "intent_result": None,
        "plan": None,
        "evidence": [],
        "draft": None,
        "critic_result": None,
        "final_suggestion": None,
        "status": "running",
        "trace_refs": [],
        "blocked": False,
        "block_reason": "",
    }

    config: Any = {"configurable": {"thread_id": thread_id}}

    try:
        result = await asyncio.wait_for(
            compiled_graph.ainvoke(initial_state, config=config),
            timeout=timeout_seconds,
        )
        return dict(result)
    except asyncio.TimeoutError:
        return {
            **initial_state,
            "status": "degraded",
            "blocked": True,
            "block_reason": "COACH_TIMEOUT",
            "trace_refs": ["timeout:hard"],
        }
    except Exception:
        return {
            **initial_state,
            "status": "degraded",
            "blocked": True,
            "block_reason": "COACH_ERROR",
            "trace_refs": ["error:unhandled"],
        }
