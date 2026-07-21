"""LangGraph 评估流程统一状态与数据契约"""

from __future__ import annotations

from operator import add
from typing import Annotated, Any, Literal, TypedDict

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
    # Knowledge Agent 检索到的指南证据（由图流程注入，供 Diagnosis/Treatment Agent 使用）
    knowledge_citations: list[dict] = Field(default_factory=list)


# ── 提交标志 ────────────────────────────────────────────────────────────────

class SubmissionFlags(BaseModel):
    """医生提交状态标志"""
    has_diagnosis: bool = False
    has_treatment: bool = False


# ── 路由计划（保留向后兼容） ────────────────────────────────────────────────

class RoutePlan(BaseModel):
    """动态路由计划（旧版，保留向后兼容）"""
    consultation_type: Literal["initial", "follow_up", "emergency", "communication"] = "initial"
    selected_agents: list[str] = Field(default_factory=list)
    skipped_agents: list[str] = Field(default_factory=list)
    skip_reasons: dict[str, str] = Field(default_factory=dict)


# ── 评估计划（Plan-Execute 模式） ────────────────────────────────────────────

class PlanStep(BaseModel):
    """评估计划中的单个步骤"""
    step_id: str
    step_type: Literal["agent_evaluation", "aggregation", "scoring", "suggestion"] = "agent_evaluation"
    agent_name: str = ""
    description: str = ""
    depends_on: list[str] = Field(default_factory=list)
    params: dict[str, Any] = Field(default_factory=dict)
    status: Literal["pending", "running", "completed", "skipped", "error"] = "pending"


class EvaluationPlan(BaseModel):
    """评估计划 — 升级版的 RoutePlan，包含完整的评估步骤信息

    Plan-Execute 模式的核心数据结构：
    - plan_evaluation 节点生成此计划
    - validate_plan 节点校验其完整性
    - execute 阶段基于 steps 执行（Send fan-out 按 steps 分发）
    """
    consultation_type: Literal["initial", "follow_up", "emergency", "communication"] = "initial"
    steps: list[PlanStep] = Field(default_factory=list)
    skipped_agents: list[str] = Field(default_factory=list)
    skip_reasons: dict[str, str] = Field(default_factory=dict)
    plan_version: str = "v1"

    @property
    def agent_steps(self) -> list[PlanStep]:
        """获取所有需要执行的 agent 评估步骤"""
        return [s for s in self.steps if s.step_type == "agent_evaluation" and s.status == "pending"]

    @property
    def selected_agents(self) -> list[str]:
        """兼容旧 RoutePlan 接口：返回选中的 agent 名称列表"""
        return [s.agent_name for s in self.agent_steps]


class ExecutionResult(BaseModel):
    """单个计划步骤的执行结果"""
    step_id: str
    agent_name: str
    status: Literal["success", "skipped", "insufficient", "error"] = "success"
    score: float | None = Field(default=None, ge=0, le=100)
    analysis: str = ""
    execution_order: int = 0
    envelope: AgentResultEnvelope | None = None


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


# ── 反思结果 ────────────────────────────────────────────────────────────────

class ReflectionIssue(BaseModel):
    """反思发现的问题"""
    issue_type: Literal["score_contradiction", "insufficient_evidence", "score_anomaly", "missing_dimension"] = "score_anomaly"
    severity: Literal["low", "medium", "high"] = "low"
    description: str = ""
    affected_dimensions: list[str] = Field(default_factory=list)
    recommendation: str = ""


class ReflectionResult(BaseModel):
    """反思智能体评估结果"""
    overall_quality: Literal["good", "acceptable", "needs_attention", "problematic"] = "acceptable"
    confidence: float = Field(default=0.5, ge=0, le=1)
    issues_found: list[ReflectionIssue] = Field(default_factory=list)
    consistency_score: float = Field(default=0.5, ge=0, le=1)
    evidence_adequacy_score: float = Field(default=0.5, ge=0, le=1)
    summary: str = ""
    needs_review: bool = False
    review_reasons: list[str] = Field(default_factory=list)
    react_steps_count: int = 0
    dimension_count: int = 0
    disabled: bool = False


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

    # 路由与计划
    route_plan: RoutePlan                          # 旧版，保留兼容
    evaluation_plan: EvaluationPlan                # 新版评估计划（Plan-Execute 核心）

    # 计划校验
    plan_valid: bool
    plan_validation_errors: list[str]

    # Safety
    safety_result: SafetyResult | None

    # Agent 结果（使用 reducer 支持并行分支合并）
    agent_results: Annotated[list[AgentResultEnvelope], add]
    # 错误记录（使用 reducer）
    node_errors: Annotated[list[NodeError], add]
    # 进度事件（使用 reducer）
    progress_events: Annotated[list[ProgressEvent], add]

    # 执行结果（基于计划步骤，使用 reducer 累积）
    execution_results: Annotated[list[ExecutionResult], add]

    # 维度结果（聚合节点写入）
    dimension_results: dict[str, DimensionResult]

    # 评分
    total_score: float | None
    overall_summary: str | None
    improvement_suggestions: list[str]

    # 反思结果
    reflection_result: ReflectionResult | None

    # Knowledge Agent 检索到的指南证据（供 Diagnosis/Treatment Agent 使用）
    knowledge_citations: list[dict]

    # 最终状态
    evaluation_status: Literal["running", "completed", "needs_review", "pending_review", "review_completed", "failed"]
    human_review_needed: bool
    review_reason: str | None

    # 人工复核相关字段
    review_feedback: str | None
    review_completed_by: str | None
    review_completed_at: str | None
    evaluation_id: str | None
