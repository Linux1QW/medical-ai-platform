"""LangGraph 评估主图 — StateGraph 构建与编译（Send fan-out/fan-in + Plan-Execute）"""

from __future__ import annotations

import asyncio
import json
import logging
from typing import Any, TypedDict

from langgraph.graph import END, START, StateGraph
from langgraph.types import Send

from app.orchestration.state import (
    AgentResultEnvelope,
    EvaluationContext,
    EvaluationPlan,
    EvaluationState,
    ExecutionResult,
    PlanStep,
    ProgressEvent,
    ReflectionIssue,
    ReflectionResult,
)
from app.orchestration.state import (
    DimensionResult as StateDimensionResult,
)
from app.services.agents.safety_agent import run_safety_check
from app.services.scoring.calculator import DimensionResult as CalcDimResult
from app.services.scoring.calculator import ScoreCalculator
from app.services.scoring.policies import get_default_policy
from app.services.scoring.summary import SummaryGenerator

logger = logging.getLogger(__name__)


# ── Send 工作器状态 ────────────────────────────────────────────────────────


class RunAgentState(TypedDict, total=False):
    """Send fan-out 工作器状态

    每个 Send 携带一个 agent 的执行上下文（含步骤信息），
    run_agent 节点读取此状态并返回 AgentResultEnvelope + ExecutionResult。
    """
    agent_name: str
    step_id: str
    context: EvaluationContext
    run_id: str
    # Knowledge Agent 检索到的指南证据（第二波执行时由 extract_knowledge_citations 注入）
    knowledge_citations: list[dict]


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


# ── Plan-Execute 节点 ────────────────────────────────────────────────────


async def plan_evaluation(state: EvaluationState) -> dict[str, Any]:
    """生成评估计划 — Plan-Execute 模式的 Plan 阶段

    基于问诊类型和提交标志，构建包含步骤、依赖关系和描述的完整评估计划。
    同时保留旧版 route_plan 以兼容下游引用。
    """
    from app.orchestration.routes import build_evaluation_plan, build_route_plan

    consultation_type = state.get("consultation_type", "initial")
    flags = state["submission_flags"]

    # 构建新版评估计划
    plan = build_evaluation_plan(consultation_type, flags)

    # 同时构建旧版 route_plan（向后兼容）
    route_plan = build_route_plan(consultation_type, flags)

    step_summary = ", ".join(
        f"{s.step_id}({s.agent_name})" for s in plan.agent_steps
    )
    return {
        "evaluation_plan": plan,
        "route_plan": route_plan,
        "progress_events": [
            ProgressEvent(
                progress=25,
                message=f"评估计划已生成: [{step_summary}]",
                node_name="plan_evaluation",
            )
        ],
    }


async def validate_plan(state: EvaluationState) -> dict[str, Any]:
    """校验评估计划 — 确保计划完整性和可执行性

    校验规则：
    1. 计划不为空（至少有一个 agent 步骤）
    2. 每个步骤的 agent_name 非空
    3. 步骤依赖引用的 step_id 存在于计划中

    校验失败时标记 plan_valid=False，后续走 needs_review 路径。
    """
    plan: EvaluationPlan | None = state.get("evaluation_plan")
    errors: list[str] = []

    if plan is None:
        errors.append("评估计划为空")
        return {
            "plan_valid": False,
            "plan_validation_errors": errors,
            "progress_events": [
                ProgressEvent(
                    progress=26,
                    message="评估计划校验失败: 计划为空",
                    node_name="validate_plan",
                )
            ],
        }

    # 规则 1: 至少一个 agent 步骤
    agent_steps = plan.agent_steps
    if not agent_steps:
        errors.append("计划中没有可执行的 agent 评估步骤")

    # 规则 2: 每个步骤 agent_name 非空
    all_step_ids = {s.step_id for s in plan.steps}
    for step in plan.steps:
        if step.step_type == "agent_evaluation" and not step.agent_name:
            errors.append(f"步骤 {step.step_id} 的 agent_name 为空")

        # 规则 3: 依赖引用有效
        for dep_id in step.depends_on:
            if dep_id not in all_step_ids:
                errors.append(f"步骤 {step.step_id} 依赖的 {dep_id} 不存在")

    is_valid = len(errors) == 0
    message = (
        f"评估计划校验通过: {len(agent_steps)} 个步骤"
        if is_valid
        else f"评估计划校验失败: {'; '.join(errors)}"
    )

    return {
        "plan_valid": is_valid,
        "plan_validation_errors": errors,
        "progress_events": [
            ProgressEvent(
                progress=27,
                message=message,
                node_name="validate_plan",
            )
        ],
    }


# 第一波执行的 Agent（knowledge 必须先完成，以便其 citations 可传递给 diagnosis/treatment）
_WAVE1_AGENTS = {"knowledge", "inquiry", "humanistic"}


def plan_valid_gate(state: EvaluationState) -> list[Send] | str:
    """计划校验条件边 + Fan-out 路由

    校验通过时，返回 Send 列表将每个 agent 步骤分发到 run_agent（fan-out）。
    校验失败时，返回 "needs_review" 字符串走复核路径。
    """
    if not state.get("plan_valid", False):
        return "needs_review"

    # 校验通过 → 构建 Wave-1 Send 列表（knowledge + inquiry + humanistic）
    context = state["context"]
    run_id = state.get("run_id", "unknown")

    plan: EvaluationPlan | None = state.get("evaluation_plan")
    if plan is not None:
        # Wave-1: 仅发送 knowledge/inquiry/humanistic
        steps = [s for s in plan.agent_steps if s.agent_name in _WAVE1_AGENTS]
    else:
        # 回退到旧版 route_plan
        route_plan = state.get("route_plan")
        if route_plan is None:
            return "needs_review"
        steps = [
            PlanStep(step_id=f"step_{name}", agent_name=name)
            for name in route_plan.selected_agents
            if name in _WAVE1_AGENTS
        ]

    # 如果没有 Wave-1 步骤（例如 communication 场景不需要 knowledge），直接跳到 Wave-2
    if not steps:
        return "skip_to_remaining"

    return [
        Send(
            "run_agent_wave1",
            {
                "agent_name": step.agent_name,
                "step_id": step.step_id,
                "context": context,
                "run_id": f"{run_id}_{step.agent_name}",
            },
        )
        for step in steps
    ]


# ── Execute 阶段（Send fan-out / fan-in） ────────────────────────────


async def extract_knowledge_citations(state: EvaluationState) -> dict[str, Any]:
    """从 Knowledge Agent 结果中提取指南证据，存入 state 供下游 Agent 使用

    在第一波 Agent 执行完成后，从 agent_results / execution_results 中
    找到 knowledge 的结果，提取 citations 写入 state.knowledge_citations。
    """
    citations: list[dict] = []

    # 优先从 execution_results 提取
    for exec_result in state.get("execution_results", []):
        if exec_result.agent_name == "knowledge" and exec_result.envelope:
            citations = exec_result.envelope.citations
            break

    # 回退：从 agent_results 提取
    if not citations:
        for result in state.get("agent_results", []):
            if result.agent_name == "knowledge":
                citations = result.citations
                break

    return {
        "knowledge_citations": citations,
        "progress_events": [
            ProgressEvent(
                progress=55,
                message=f"知识证据提取完成: {len(citations)} 条引用",
                node_name="extract_knowledge_citations",
            )
        ],
    }


def dispatch_remaining_agents(state: EvaluationState) -> list[Send]:
    """第二波 Fan-out 路由 — 发送非 Wave-1 的 Agent 步骤（携带 knowledge citations）

    从 evaluation_plan 中筛选出不在 _WAVE1_AGENTS 中的步骤，
    为每个步骤创建 Send，并将 knowledge_citations 注入 RunAgentState。
    """
    context = state["context"]
    run_id = state.get("run_id", "unknown")
    knowledge_citations = state.get("knowledge_citations", [])

    plan: EvaluationPlan | None = state.get("evaluation_plan")
    route_plan = state.get("route_plan")

    if plan is not None:
        steps = [s for s in plan.agent_steps if s.agent_name not in _WAVE1_AGENTS]
    elif route_plan is not None:
        steps = [
            PlanStep(step_id=f"step_{name}", agent_name=name)
            for name in route_plan.selected_agents
            if name not in _WAVE1_AGENTS
        ]
    else:
        return []

    return [
        Send(
            "run_agent",
            {
                "agent_name": step.agent_name,
                "step_id": step.step_id,
                "context": context,
                "run_id": f"{run_id}_{step.agent_name}",
                "knowledge_citations": knowledge_citations,
            },
        )
        for step in steps
    ]


async def run_agent_wave1(state: RunAgentState) -> dict[str, Any]:
    """Wave-1 工作器节点 — 执行第一波 Agent（knowledge/inquiry/humanistic）

    与 run_agent 逻辑相同，但作为独立节点以便图结构区分两波执行。
    """
    return await run_agent(state)


async def run_agent(state: RunAgentState) -> dict[str, Any]:
    """Send fan-out 工作器节点 — 执行单个 Agent（基于计划步骤）

    由 route_to_agents 条件边通过 Send 机制触发，
    每个 Send 携带 RunAgentState（agent_name + step_id + context）。
    返回的 agent_results 和 execution_results 通过 reducer（operator.add）累积到主状态。
    """
    from app.orchestration.adapters.registry import get_adapter

    agent_name = state["agent_name"]
    step_id = state.get("step_id", f"step_{agent_name}")
    context = state["context"]

    # 注入 knowledge_citations 到 context（如果 RunAgentState 中有提供）
    knowledge_citations = state.get("knowledge_citations")
    if knowledge_citations and agent_name != "knowledge":
        context = context.model_copy(update={"knowledge_citations": knowledge_citations})

    try:
        adapter = get_adapter(agent_name)
        envelope = await adapter.run(context)
    except Exception as e:
        logger.error(f"Send agent '{agent_name}' (step={step_id}) failed: {e}", exc_info=True)
        envelope = AgentResultEnvelope(
            agent_name=agent_name,  # type: ignore[arg-type]
            status="error",
            analysis=f"执行异常: {e}",
            human_review_needed=True,
            review_reason=str(e)[:200],
        )

    # 构建执行结果
    exec_result = ExecutionResult(
        step_id=step_id,
        agent_name=agent_name,
        status=envelope.status,
        score=envelope.score,
        analysis=envelope.analysis,
        envelope=envelope,
    )

    return {
        "agent_results": [envelope],
        "execution_results": [exec_result],
        "progress_events": [
            ProgressEvent(
                progress=50,
                message=f"Agent '{agent_name}' (步骤 {step_id}) 执行完成",
                node_name="run_agent",
            )
        ],
    }


def route_to_agents(state: EvaluationState) -> list[Send]:
    """Fan-out 条件路由 — 为评估计划中的每个 agent 步骤生成一个 Send

    读取 evaluation_plan.agent_steps，为每个步骤创建独立的
    RunAgentState 并通过 Send 机制分发到 run_agent 节点并行执行。
    回退到 route_plan.selected_agents（向后兼容）。
    """
    context = state["context"]
    run_id = state.get("run_id", "unknown")

    # 优先使用 evaluation_plan
    plan: EvaluationPlan | None = state.get("evaluation_plan")
    if plan is not None:
        steps = plan.agent_steps
    else:
        # 回退到旧版 route_plan
        route_plan = state.get("route_plan")
        if route_plan is None:
            logger.warning("No evaluation_plan or route_plan found, returning empty Send list")
            return []
        steps = [
            PlanStep(step_id=f"step_{name}", agent_name=name)
            for name in route_plan.selected_agents
        ]

    return [
        Send(
            "run_agent",
            {
                "agent_name": step.agent_name,
                "step_id": step.step_id,
                "context": context,
                "run_id": f"{run_id}_{step.agent_name}",
            },
        )
        for step in steps
    ]


async def aggregate_results(state: EvaluationState) -> dict[str, Any]:
    """聚合 Agent 结果为 dimension_results（Fan-in 汇聚点）

    在 Send fan-out/fan-in 模式下，所有 run_agent 并行执行后，
    其 agent_results 通过 reducer 累积到此节点，由本函数统一转换。
    优先从 execution_results 中提取（包含步骤信息），回退到 agent_results。
    """
    dimensions: dict[str, StateDimensionResult] = {}

    # 优先从 execution_results 转换（Plan-Execute 路径）
    exec_results = state.get("execution_results", [])
    if exec_results:
        for exec_result in exec_results:
            dim = StateDimensionResult(
                dimension=exec_result.agent_name,
                status="scored" if exec_result.status == "success" else exec_result.status,
                score=exec_result.score,
                analysis=exec_result.analysis,
            )
            dimensions[exec_result.agent_name] = dim
    else:
        # 回退：从 agent_results 转换（兼容旧路径）
        for result in state.get("agent_results", []):
            dim = StateDimensionResult(
                dimension=result.agent_name,
                status="scored" if result.status == "success" else result.status,
                score=result.score,
                analysis=result.analysis,
            )
            dimensions[result.agent_name] = dim

    # 处理被跳过的 Agent
    plan: EvaluationPlan | None = state.get("evaluation_plan")
    route_plan = state.get("route_plan")

    if plan:
        for skipped in plan.skipped_agents:
            if skipped not in dimensions:
                dimensions[skipped] = StateDimensionResult(
                    dimension=skipped,
                    status="not_submitted",
                    score=None,
                    analysis=plan.skip_reasons.get(skipped, "未提交"),
                )
    elif route_plan:
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


# ── 保留旧版 dispatch_and_run（向后兼容） ────────────────────────────────


async def dispatch_and_run(state: EvaluationState) -> dict[str, Any]:
    """分发并并行运行选中 Agent（保留向后兼容）

    注意：主流程已升级为 Send fan-out/fan-in 模式（route_to_agents → run_agent），
    此函数保留用于直接调用或测试场景。
    """
    from app.orchestration.adapters.registry import get_adapter

    plan = state.get("route_plan")
    context = state["context"]

    if plan is None:
        return {
            "agent_results": [],
            "progress_events": [
                ProgressEvent(
                    progress=60,
                    message="无路由计划，跳过执行",
                    node_name="dispatch_and_run",
                )
            ],
        }

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


# ── 确定性评分 ────────────────────────────────────────────────────────────


async def deterministic_scoring(state: EvaluationState) -> dict[str, Any]:
    """确定性评分 — ScoreCalculator 保证评分结果可复现，不允许 LLM 自主决定总分"""
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


async def reflection_check(state: EvaluationState) -> dict[str, Any]:
    """反思检查 — 评分完成后由 Reflection Agent 进行自我反思和验证

    反思结果不替代原始评分，仅作为辅助参考。
    如果 ENABLE_REACT_REFLECTION=false，则跳过反思直接返回。
    """
    from app.core.config import settings as app_settings
    from app.orchestration.adapters.reflection import run_reflection_from_context

    if not app_settings.ENABLE_REACT_REFLECTION:
        return {
            "reflection_result": ReflectionResult(disabled=True),
            "progress_events": [
                ProgressEvent(
                    progress=82,
                    message="反思智能体未启用，跳过",
                    node_name="reflection_check",
                )
            ],
        }

    dimension_results = state.get("dimension_results", {})
    total_score = state.get("total_score")

    try:
        raw_result = await run_reflection_from_context(
            dimension_results=dimension_results,
            total_score=total_score,
        )

        # 转换为 ReflectionResult Pydantic 模型
        issues = []
        for issue_data in raw_result.get("issues_found", []):
            issues.append(ReflectionIssue(
                issue_type=issue_data.get("type", "score_anomaly"),
                severity=issue_data.get("severity", "low"),
                description=issue_data.get("description", ""),
                affected_dimensions=issue_data.get("affected_dimensions", []),
                recommendation=issue_data.get("recommendation", ""),
            ))

        reflection = ReflectionResult(
            overall_quality=raw_result.get("overall_quality", "acceptable"),
            confidence=raw_result.get("confidence", 0.5),
            issues_found=issues,
            consistency_score=raw_result.get("consistency_score", 0.5),
            evidence_adequacy_score=raw_result.get("evidence_adequacy_score", 0.5),
            summary=raw_result.get("summary", ""),
            needs_review=raw_result.get("needs_review", False),
            review_reasons=raw_result.get("review_reasons", []),
            react_steps_count=raw_result.get("react_steps_count", 0),
            dimension_count=raw_result.get("dimension_count", 0),
        )

        quality_msg = f"反思检查完成: quality={reflection.overall_quality}, issues={len(issues)}"
        return {
            "reflection_result": reflection,
            "progress_events": [
                ProgressEvent(
                    progress=85,
                    message=quality_msg,
                    node_name="reflection_check",
                )
            ],
        }

    except Exception as e:
        logger.error(f"Reflection check failed: {e}", exc_info=True)
        return {
            "reflection_result": ReflectionResult(
                overall_quality="acceptable",
                confidence=0.3,
                summary=f"反思检查异常: {str(e)[:100]}",
            ),
            "progress_events": [
                ProgressEvent(
                    progress=85,
                    message=f"反思检查异常: {type(e).__name__}",
                    node_name="reflection_check",
                )
            ],
        }


async def review_gate_node(state: EvaluationState) -> dict[str, Any]:
    """人工复核门控节点

    检查是否需要人工复核。如果需要，标记为 pending_review 并保存 checkpoint 到 Redis，
    图执行将在此暂停，等待外部 API 调用恢复。
    """
    needs_review = False
    review_reasons: list[str] = []

    # 检查是否有维度需要 review
    for dim in state.get("dimension_results", {}).values():
        if dim.status in ("error", "insufficient"):
            needs_review = True
            review_reasons.append(f"{dim.dimension}: {dim.status}")

    # 检查 agent_results 中是否有 human_review_needed
    for result in state.get("agent_results", []):
        if result.human_review_needed:
            needs_review = True
            review_reasons.append(f"{result.agent_name}: {result.review_reason}")

    # 检查反思结果是否标记需要复核
    reflection = state.get("reflection_result")
    if reflection is not None and reflection.needs_review:
        needs_review = True
        review_reasons.extend(reflection.review_reasons)

    if needs_review:
        reason_text = "; ".join(review_reasons) if review_reasons else "需要人工复核"
        state_updates: dict[str, Any] = {
            "evaluation_status": "pending_review",
            "human_review_needed": True,
            "review_reason": reason_text,
        }

        # 保存 checkpoint 到 Redis
        eval_id = state.get("evaluation_id") or state.get("run_id")
        if eval_id:
            try:
                from app.services.llm_cache import _get_redis
                redis = await _get_redis()
                if redis:
                    key = f"eval_checkpoint:{eval_id}"
                    # 构建可序列化的状态快照
                    snapshot = {
                        "evaluation_id": eval_id,
                        "run_id": state.get("run_id"),
                        "consultation_id": state.get("consultation_id"),
                        "evaluation_status": "pending_review",
                        "needs_review": True,
                        "review_reason": reason_text,
                        "total_score": state.get("total_score"),
                        "dimension_results": {
                            k: {"dimension": v.dimension, "status": v.status, "score": v.score, "analysis": v.analysis}
                            for k, v in state.get("dimension_results", {}).items()
                        },
                        "human_review_needed": True,
                    }
                    await redis.set(
                        key,
                        json.dumps(snapshot, ensure_ascii=False, default=str),
                        ex=604800,  # 7 天 TTL
                    )
                    logger.info(f"Checkpoint saved for evaluation {eval_id}")
            except Exception as e:
                logger.warning(f"Failed to save checkpoint: {e}")

        state_updates["progress_events"] = [
            ProgressEvent(
                progress=88,
                message=f"需要人工复核: {reason_text[:80]}",
                node_name="review_gate",
            )
        ]
        return state_updates

    return {
        "progress_events": [
            ProgressEvent(
                progress=88,
                message="复核检查通过，无需人工介入",
                node_name="review_gate",
            )
        ],
    }


def review_gate_router(state: EvaluationState) -> str:
    """审核门控条件边路由

    根据 review_gate_node 设置的状态决定路由方向。
    """
    if state.get("evaluation_status") == "pending_review":
        return "needs_review"
    return "completed"


async def generate_suggestion(state: EvaluationState) -> dict[str, Any]:
    """生成改进建议 — 优先调用 LLM suggestion agent，失败时回退规则建议

    当 ENABLE_LLM_SUGGESTION=true 时，调用 suggestion_agent 进行对比学习分析。
    LLM 调用失败时自动降级到原有的规则建议方案。
    """
    import json

    from app.core.config import settings as app_settings
    from app.orchestration.adapters.suggestion import run_suggestion_from_context

    context = state.get("context")
    dimension_results = state.get("dimension_results", {})

    # 构建维度评估结果字符串
    inquiry_result = ""
    knowledge_result = ""
    humanistic_result = ""

    for name, dim in dimension_results.items():
        if name == "inquiry":
            inquiry_result = f"得分:{dim.score}分。{dim.analysis}" if dim.score is not None else dim.analysis
        elif name == "knowledge":
            knowledge_result = f"得分:{dim.score}分。{dim.analysis}" if dim.score is not None else dim.analysis
        elif name == "humanistic":
            humanistic_result = f"得分:{dim.score}分。{dim.analysis}" if dim.score is not None else dim.analysis

    # 构建患者信息
    patient_info_parts = []
    if context and context.patient_age is not None:
        patient_info_parts.append(f"年龄:{context.patient_age}岁")
    if context and context.patient_gender:
        patient_info_parts.append(f"性别:{context.patient_gender}")
    if context and context.chief_complaint:
        patient_info_parts.append(f"主诉:{context.chief_complaint}")
    if context and context.medical_history:
        patient_info_parts.append(f"病史:{context.medical_history}")
    patient_info = "，".join(patient_info_parts) if patient_info_parts else "无额外患者信息"

    conversation_text = context.conversation_text if context else ""

    # 尝试调用 LLM suggestion agent
    try:
        if app_settings.ENABLE_LLM_SUGGESTION:
            raw_result = await run_suggestion_from_context(
                conversation_text=conversation_text,
                patient_info=patient_info,
                inquiry_result=inquiry_result,
                knowledge_result=knowledge_result,
                humanistic_result=humanistic_result,
            )

            # 解析 JSON 响应
            raw_response = raw_result.get("raw_response", "{}")
            suggestion_data = json.loads(raw_response)

            # 提取 suggestions 字段
            suggestions_text = suggestion_data.get("suggestions", "")
            if suggestions_text:
                # 将文本按段落分割为列表
                suggestions = [s.strip() for s in suggestions_text.split("\n\n") if s.strip()]
                if not suggestions:
                    suggestions = [suggestions_text]
            else:
                # 如果没有 suggestions 字段，从其他字段构造
                missing = suggestion_data.get("missing_questions", [])
                improvements = suggestion_data.get("improvement_suggestions", [])
                suggestions = []
                if missing:
                    suggestions.append("缺失的关键问题：" + "；".join(missing[:5]))
                if improvements:
                    suggestions.append("改进措施：" + "；".join(improvements[:5]))
                if not suggestions:
                    suggestions = ["整体表现良好，继续保持"]

            logger.info(f"LLM suggestion generated successfully with {len(suggestions)} items")
            return {
                "improvement_suggestions": suggestions,
                "progress_events": [
                    ProgressEvent(
                        progress=90,
                        message="LLM 建议生成完成",
                        node_name="generate_suggestion",
                    )
                ],
            }
        else:
            logger.debug("ENABLE_LLM_SUGGESTION=false, using rule-based fallback")
    except Exception as e:
        logger.error(f"LLM suggestion failed, falling back to rule-based: {e}", exc_info=True)

    # 降级：规则建议（原有逻辑）
    suggestions: list[str] = []
    for name, dim in dimension_results.items():
        if dim.status == "scored" and dim.score is not None and dim.score < 70:
            suggestions.append(f"{name}维度得分较低({dim.score}分)，建议重点改进")

    # 整合反思结果中的建议
    reflection = state.get("reflection_result")
    if reflection is not None and not reflection.disabled:
        for issue in reflection.issues_found:
            if issue.severity in ("medium", "high") and issue.recommendation:
                suggestion_text = f"[反思发现] {issue.description}: {issue.recommendation}"
                if suggestion_text not in suggestions:
                    suggestions.append(suggestion_text)

        if reflection.consistency_score < 0.5:
            suggestions.append("各维度评分一致性较低，建议关注评估的整体协调性")
        if reflection.evidence_adequacy_score < 0.5:
            suggestions.append("证据充分性不足，建议补充更多临床信息")

    if not suggestions:
        suggestions = ["整体表现良好，继续保持"]

    return {
        "improvement_suggestions": suggestions,
        "progress_events": [
            ProgressEvent(
                progress=90,
                message="建议生成完成（规则降级）",
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

    # 检查计划校验错误
    plan_errors = state.get("plan_validation_errors", [])
    if plan_errors:
        reasons.append(f"计划校验失败: {'; '.join(plan_errors)}")

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

    状态图结构（Plan-Execute + Send fan-out/fan-in + ReAct Reflection + Checkpoint Review）：
    START → load_context → classify_consultation → safety_check → safety_gate
      → plan_evaluation → validate_plan → plan_valid_gate
        → [Send fan-out] → run_agent × N → [fan-in] → aggregate_results
        → deterministic_scoring → reflection_check → review_gate_node
        → review_gate_router → generate_suggestion → finalize_completed → END
                              → finalize_needs_review → END（暂停等待复核恢复）
    """
    graph = StateGraph(EvaluationState)

    # 添加节点
    graph.add_node("load_context", load_context)
    graph.add_node("classify_consultation", classify_consultation)
    graph.add_node("safety_check", safety_check)
    graph.add_node("plan_evaluation", plan_evaluation)
    graph.add_node("validate_plan", validate_plan)
    graph.add_node("run_agent_wave1", run_agent_wave1)
    graph.add_node("run_agent", run_agent)
    graph.add_node("dispatch_and_run", dispatch_and_run)
    graph.add_node("extract_knowledge_citations", extract_knowledge_citations)
    graph.add_node("aggregate_results", aggregate_results)
    graph.add_node("deterministic_scoring", deterministic_scoring)
    graph.add_node("reflection_check", reflection_check)
    graph.add_node("review_gate_node", review_gate_node)
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
            "continue": "plan_evaluation",
            "needs_review": "finalize_needs_review",
        },
    )

    # Plan → Validate → 校验条件边 + 两波 Fan-out/Fan-in
    graph.add_edge("plan_evaluation", "validate_plan")

    # validate_plan 的条件边：
    # - 校验通过 → 返回 list[Send] fan-out 到 run_agent_wave1（Wave-1: knowledge/inquiry/humanistic）
    # - 校验失败 → 返回 "needs_review" 字符串
    # - 无 Wave-1 步骤 → 返回 "skip_to_remaining" 直接到 Wave-2
    graph.add_conditional_edges(
        "validate_plan",
        plan_valid_gate,
        {
            "needs_review": "finalize_needs_review",
            "run_agent": "run_agent_wave1",
            "skip_to_remaining": "extract_knowledge_citations",
        },
    )
    # Wave-1 完成后 → 提取 knowledge citations
    graph.add_edge("run_agent_wave1", "extract_knowledge_citations")
    # 提取完成后 → Wave-2 fan-out (diagnosis/treatment，携带 citations)
    graph.add_conditional_edges(
        "extract_knowledge_citations",
        dispatch_remaining_agents,
        {
            "run_agent": "run_agent",
        },
    )
    # Wave-2 完成后 → 结果聚合
    graph.add_edge("run_agent", "aggregate_results")

    graph.add_edge("aggregate_results", "deterministic_scoring")

    # 评分后 → 反思检查
    graph.add_edge("deterministic_scoring", "reflection_check")

    # 反思检查后 → 复核门控节点
    graph.add_edge("reflection_check", "review_gate_node")

    # 复核门控节点 → 条件路由
    graph.add_conditional_edges(
        "review_gate_node",
        review_gate_router,
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
    """获取编译后的图（应用生命周期内只编译一次）

    Returns:
        编译后的图，如果 LANGGRAPH_ENABLED=false 则返回 None
    """
    global _compiled_graph

    from app.core.config import settings

    # LANGGRAPH_ENABLED=false 时不编译图
    if not settings.LANGGRAPH_ENABLED:
        logger.debug("LangGraph disabled, get_graph returning None")
        return None

    if _compiled_graph is None:
        from app.orchestration.checkpointer import get_checkpointer

        checkpointer = get_checkpointer()
        graph = build_evaluation_graph()

        if checkpointer is not None:
            _compiled_graph = graph.compile(checkpointer=checkpointer)
            logger.info("LangGraph 评估图已编译（带 checkpointer）")
        else:
            # LANGGRAPH_ENABLED=true 但 checkpointer 为 None 不应该发生
            # 因为 init_checkpointer 在 LANGGRAPH_ENABLED=true 时会抛异常
            _compiled_graph = graph.compile()
            logger.warning("LangGraph 评估图已编译（无 checkpointer）")

    return _compiled_graph


async def close_graph():
    """关闭图（重置编译缓存）"""
    global _compiled_graph
    _compiled_graph = None
    logger.info("LangGraph 评估图缓存已清除")
