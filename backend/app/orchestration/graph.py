"""LangGraph 评估主图 — StateGraph 构建与编译"""

from __future__ import annotations

import asyncio
import logging
from typing import Any

from langgraph.graph import StateGraph, START, END

from app.orchestration.state import (
    EvaluationState,
    AgentResultEnvelope,
    DimensionResult as StateDimensionResult,
    ProgressEvent,
)
from app.services.agents.safety_agent import run_safety_check
from app.services.scoring.policies import get_default_policy
from app.services.scoring.calculator import ScoreCalculator, DimensionResult as CalcDimResult
from app.services.scoring.summary import SummaryGenerator

logger = logging.getLogger(__name__)

# ── 节点函数 ──────────────────────────────────────────────────────────────


async def load_context(state: EvaluationState) -> dict[str, Any]:
    """加载上下文 — 从 state 中已有的 context 字段获取，无需额外加载"""
    return {
        "progress_events": [
            ProgressEvent(
                progress=10,
                message="上下文加载完成",
                node_name="load_context",
            )
        ]
    }


async def classify_consultation(state: EvaluationState) -> dict[str, Any]:
    """分类问诊类型 — 从 context 和 consultation 获取"""
    consultation_type = state.get("consultation_type", "initial")
    return {
        "consultation_type": consultation_type,
        "progress_events": [
            ProgressEvent(
                progress=15,
                message="问诊类型已确认",
                node_name="classify_consultation",
            )
        ],
    }


async def safety_check(state: EvaluationState) -> dict[str, Any]:
    """执行 Safety 检查"""
    conversation_text = state["context"].conversation_text
    result = await run_safety_check(conversation_text)
    return {
        "safety_result": result,
        "progress_events": [
            ProgressEvent(
                progress=20,
                message=f"安全检查完成: {result.risk_level}",
                node_name="safety_check",
            )
        ],
    }


def safety_gate(state: EvaluationState) -> str:
    """Safety 条件边路由"""
    safety = state.get("safety_result")
    if safety is None:
        # fail closed: 无结果时走复核路径
        return "needs_review"
    if safety.risk_level in ("high", "undetermined") or safety.immediate_review_required:
        return "needs_review"
    return "continue"


async def build_route_plan_node(state: EvaluationState) -> dict[str, Any]:
    """构建路由计划"""
    from app.orchestration.routes import build_route_plan

    plan = build_route_plan(
        state.get("consultation_type", "initial"),
        state["submission_flags"],
    )
    return {
        "route_plan": plan,
        "progress_events": [
            ProgressEvent(
                progress=25,
                message=f"路由计划: {plan.selected_agents}",
                node_name="build_route_plan",
            )
        ],
    }


async def dispatch_and_run(state: EvaluationState) -> dict[str, Any]:
    """分发并并行运行选中 Agent（简化方案：asyncio.gather）"""
    from app.orchestration.adapters.registry import get_adapter

    plan = state["route_plan"]
    context = state["context"]

    async def run_one(agent_name: str) -> AgentResultEnvelope:
        adapter = get_adapter(agent_name)
        return await adapter.run(context)

    results = await asyncio.gather(
        *[run_one(name) for name in plan.selected_agents],
        return_exceptions=True,
    )

    envelopes: list[AgentResultEnvelope] = []
    for name, result in zip(plan.selected_agents, results):
        if isinstance(result, Exception):
            envelopes.append(
                AgentResultEnvelope(
                    agent_name=name,  # type: ignore[arg-type]
                    status="error",
                    analysis=f"执行异常: {result}",
                    human_review_needed=True,
                    review_reason=str(result)[:200],
                )
            )
        else:
            envelopes.append(result)

    return {
        "agent_results": envelopes,
        "progress_events": [
            ProgressEvent(
                progress=60,
                message=f"Agent执行完成: {len(envelopes)}个",
                node_name="dispatch_and_run",
            )
        ],
    }


async def aggregate_results(state: EvaluationState) -> dict[str, Any]:
    """聚合 Agent 结果为 dimension_results"""
    dimensions: dict[str, StateDimensionResult] = {}

    # 从 agent_results 转换
    for result in state.get("agent_results", []):
        dim = StateDimensionResult(
            dimension=result.agent_name,
            status="scored" if result.status == "success" else result.status,
            score=result.score,
            analysis=result.analysis,
        )
        dimensions[result.agent_name] = dim

    # 处理被跳过的 Agent
    route_plan = state.get("route_plan")
    if route_plan:
        for skipped in route_plan.skipped_agents:
            if skipped not in dimensions:
                dimensions[skipped] = StateDimensionResult(
                    dimension=skipped,
                    status="not_submitted",
                    score=None,
                    analysis=route_plan.skip_reasons.get(skipped, "未提交"),
                )

    return {
        "dimension_results": dimensions,
        "progress_events": [
            ProgressEvent(
                progress=70,
                message="结果聚合完成",
                node_name="aggregate_results",
            )
        ],
    }


async def deterministic_scoring(state: EvaluationState) -> dict[str, Any]:
    """确定性评分"""
    policy = get_default_policy()
    calculator = ScoreCalculator(policy)

    # 转换为 calculator 的 DimensionResult
    calc_dims: dict[str, CalcDimResult] = {}
    for name, dim in state.get("dimension_results", {}).items():
        calc_dims[name] = CalcDimResult(
            dimension=dim.dimension,
            status=dim.status,
            score=dim.score,
            analysis=dim.analysis,
        )

    calculation = calculator.calculate(calc_dims)

    return {
        "total_score": calculation.total_score,
        "scoring_policy_version": policy.version,
        "progress_events": [
            ProgressEvent(
                progress=80,
                message=f"评分完成: {calculation.total_score}",
                node_name="deterministic_scoring",
            )
        ],
    }


def review_gate(state: EvaluationState) -> str:
    """审核门控条件边"""
    # 检查是否有维度需要 review
    for dim in state.get("dimension_results", {}).values():
        if dim.status in ("error", "insufficient"):
            return "needs_review"

    # 检查 agent_results 中是否有 human_review_needed
    for result in state.get("agent_results", []):
        if result.human_review_needed:
            return "needs_review"

    return "completed"


async def generate_suggestion(state: EvaluationState) -> dict[str, Any]:
    """生成改进建议（确定性降级，本阶段不调用 LLM suggestion agent）"""
    suggestions: list[str] = []
    for name, dim in state.get("dimension_results", {}).items():
        if dim.status == "scored" and dim.score is not None and dim.score < 70:
            suggestions.append(f"{name}维度得分较低({dim.score}分)，建议重点改进")

    if not suggestions:
        suggestions = ["整体表现良好，继续保持"]

    return {
        "improvement_suggestions": suggestions,
        "progress_events": [
            ProgressEvent(
                progress=90,
                message="建议生成完成",
                node_name="generate_suggestion",
            )
        ],
    }


async def finalize_completed(state: EvaluationState) -> dict[str, Any]:
    """完成状态 — 生成摘要"""
    policy = get_default_policy()
    generator = SummaryGenerator(policy)

    summary = await generator.generate(
        state.get("dimension_results", {}),
        state.get("total_score"),
    )

    return {
        "overall_summary": summary,
        "evaluation_status": "completed",
        "human_review_needed": False,
        "progress_events": [
            ProgressEvent(
                progress=100,
                message="评估完成",
                node_name="finalize_completed",
            )
        ],
    }


async def finalize_needs_review(state: EvaluationState) -> dict[str, Any]:
    """需要人工复核状态"""
    reasons: list[str] = []

    safety = state.get("safety_result")
    if safety and safety.risk_level in ("high", "undetermined"):
        reasons.append(f"Safety: {safety.risk_level} - {safety.reasoning_summary}")

    for result in state.get("agent_results", []):
        if result.human_review_needed:
            reasons.append(f"{result.agent_name}: {result.review_reason}")

    for name, dim in state.get("dimension_results", {}).items():
        if dim.status in ("error", "insufficient"):
            reasons.append(f"{name}: {dim.status}")

    return {
        "evaluation_status": "needs_review",
        "human_review_needed": True,
        "review_reason": "; ".join(reasons) if reasons else "需要人工复核",
        "total_score": None,
        "overall_summary": "评估过程中出现需要人工复核的情况，综合评分暂未生成。",
        "progress_events": [
            ProgressEvent(
                progress=100,
                message="评估需要人工复核",
                node_name="finalize_needs_review",
            )
        ],
    }


# ── 图构建 ────────────────────────────────────────────────────────────────


def build_evaluation_graph() -> StateGraph:
    """构建评估状态图

    状态图结构：
    START → load_context → classify_consultation → safety_check → safety_gate
      → build_route_plan → dispatch_and_run → aggregate_results → deterministic_scoring
      → review_gate → generate_suggestion → finalize_completed → END
                  → finalize_needs_review → END
    """
    graph = StateGraph(EvaluationState)

    # 添加节点
    graph.add_node("load_context", load_context)
    graph.add_node("classify_consultation", classify_consultation)
    graph.add_node("safety_check", safety_check)
    graph.add_node("build_route_plan", build_route_plan_node)
    graph.add_node("dispatch_and_run", dispatch_and_run)
    graph.add_node("aggregate_results", aggregate_results)
    graph.add_node("deterministic_scoring", deterministic_scoring)
    graph.add_node("generate_suggestion", generate_suggestion)
    graph.add_node("finalize_completed", finalize_completed)
    graph.add_node("finalize_needs_review", finalize_needs_review)

    # 线性边
    graph.add_edge(START, "load_context")
    graph.add_edge("load_context", "classify_consultation")
    graph.add_edge("classify_consultation", "safety_check")

    # Safety 条件边
    graph.add_conditional_edges(
        "safety_check",
        safety_gate,
        {
            "continue": "build_route_plan",
            "needs_review": "finalize_needs_review",
        },
    )

    graph.add_edge("build_route_plan", "dispatch_and_run")
    graph.add_edge("dispatch_and_run", "aggregate_results")
    graph.add_edge("aggregate_results", "deterministic_scoring")

    # 评分后审核门控
    graph.add_conditional_edges(
        "deterministic_scoring",
        review_gate,
        {
            "completed": "generate_suggestion",
            "needs_review": "finalize_needs_review",
        },
    )

    graph.add_edge("generate_suggestion", "finalize_completed")
    graph.add_edge("finalize_completed", END)
    graph.add_edge("finalize_needs_review", END)

    return graph


# ── 编译与生命周期管理 ────────────────────────────────────────────────────

_compiled_graph = None


async def get_graph():
    """获取编译后的图（应用生命周期内只编译一次）"""
    global _compiled_graph
    if _compiled_graph is None:
        from app.orchestration.checkpointer import get_checkpointer

        checkpointer = get_checkpointer()
        graph = build_evaluation_graph()
        _compiled_graph = graph.compile(checkpointer=checkpointer)
        logger.info("LangGraph 评估图已编译")
    return _compiled_graph


async def close_graph():
    """关闭图（重置编译缓存）"""
    global _compiled_graph
    _compiled_graph = None
    logger.info("LangGraph 评估图缓存已清除")
