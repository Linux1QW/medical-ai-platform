"""LangGraph 评估流程统一状态与数据契约"""

from __future__ import annotations

from operator import add
from typing import Annotated, Any, Literal, Optional, TypedDict

from pydantic import BaseModel, Field


# ── 输入上下文 ──────────────────────────────────────────────────────────────

class EvaluationContext(BaseModel):
    """评估输入上下文（去标识化，不含患者姓名等直接标识符）"""
    conversation_text: str = ""
    patient_age: int | None = None
    patient_gender: str | None = None
    chief_complaint: str = ""
    medical_history: str = ""
    symptoms: list[str] = Field(default_factory=list)
    doctor_diagnosis: str | None = None
    treatment_plan: str | None = None


# ── 提交标志 ────────────────────────────────────────────────────────────────

class SubmissionFlags(BaseModel):
    """医生提交状态标志"""
    has_diagnosis: bool = False
    has_treatment: bool = False


# ── 路由计划 ────────────────────────────────────────────────────────────────

class RoutePlan(BaseModel):
    """动态路由计划"""
    consultation_type: Literal["initial", "follow_up", "emergency", "communication"] = "initial"
    selected_agents: list[str] = Field(default_factory=list)
    skipped_agents: list[str] = Field(default_factory=list)
    skip_reasons: dict[str, str] = Field(default_factory=dict)


# ── Safety 结果 ─────────────────────────────────────────────────────────────

class SafetyResult(BaseModel):
    """Safety Agent 检查结果"""
    risk_level: Literal["low", "medium", "high", "undetermined"] = "low"
    matched_rules: list[str] = Field(default_factory=list)
    reasoning_summary: str = ""
    immediate_review_required: bool = False
    degraded: bool = False


# ── Agent 结果 ──────────────────────────────────────────────────────────────

class AgentResultEnvelope(BaseModel):
    """标准化 Agent 返回信封"""
    agent_name: Literal["inquiry", "diagnosis", "treatment", "knowledge", "humanistic"]
    status: Literal["success", "skipped", "insufficient", "error"] = "success"
    score: float | None = Field(default=None, ge=0, le=100)
    analysis: str = ""
    skip_reason: str | None = None
    human_review_needed: bool = False
    review_reason: str | None = None
    citations: list[dict] = Field(default_factory=list)
    trace: dict = Field(default_factory=dict)


# ── 维度结果 ────────────────────────────────────────────────────────────────

class DimensionResult(BaseModel):
    """单个评估维度的结构化结果"""
    dimension: str
    status: Literal["scored", "not_applicable", "not_submitted", "insufficient", "error"] = "scored"
    score: float | None = Field(default=None, ge=0, le=100)
    analysis: str = ""


# ── 进度事件 ────────────────────────────────────────────────────────────────

class ProgressEvent(BaseModel):
    """WebSocket 进度事件"""
    progress: int = Field(ge=0, le=100)
    message: str = ""
    node_name: str = ""


# ── 节点错误 ────────────────────────────────────────────────────────────────

class NodeError(BaseModel):
    """节点执行错误记录"""
    node_name: str
    error_type: str = ""
    error_message: str = ""
    attempt: int = 1


# ── 主状态 ──────────────────────────────────────────────────────────────────

class EvaluationState(TypedDict, total=False):
    """LangGraph 评估流程统一状态

    使用 Annotated + reducer 的字段支持并行分支合并。
    所有字段均为可选（total=False），节点按需写入。
    """
    # 运行标识
    run_id: str
    consultation_id: int
    graph_version: str
    scoring_policy_version: str

    # 输入
    context: EvaluationContext
    consultation_type: Literal["initial", "follow_up", "emergency", "communication"]
    submission_flags: SubmissionFlags
    route_plan: RoutePlan

    # Safety
    safety_result: SafetyResult | None

    # Agent 结果（使用 reducer 支持并行分支合并）
    agent_results: Annotated[list[AgentResultEnvelope], add]
    # 错误记录（使用 reducer）
    node_errors: Annotated[list[NodeError], add]
    # 进度事件（使用 reducer）
    progress_events: Annotated[list[ProgressEvent], add]

    # 维度结果（聚合节点写入）
    dimension_results: dict[str, DimensionResult]

    # 评分
    total_score: float | None
    overall_summary: str | None
    improvement_suggestions: list[str]

    # 最终状态
    evaluation_status: Literal["running", "completed", "needs_review", "failed"]
    human_review_needed: bool
    review_reason: str | None
