# -*- coding: utf-8 -*-
"""确定性评分计算器 — 纯代码，不调用 LLM"""

from pydantic import BaseModel
from app.services.scoring.policies import ScoringPolicy


class DimensionResult(BaseModel):
    """单个维度的评分结果"""
    dimension: str
    status: str  # "scored", "not_applicable", "not_submitted", "insufficient", "error"
    score: float | None = None  # 0-100，仅 status="scored" 时有值
    analysis: str = ""


class ScoreCalculation(BaseModel):
    """评分计算结果"""
    total_score: int | None = None
    status: str  # "completed", "incomplete"
    reason: str | None = None  # incomplete 时的原因
    weights_used: dict[str, float] = {}
    dimension_scores: dict[str, float | None] = {}


class ScoreCalculator:
    """确定性评分计算器

    关键规则：
    1. 只有 status="scored" 的维度参与加权
    2. 禁止因本次缺失某维度而临时重分配权重
    3. 缺失必需维度时 total_score=None
    4. 所有维度都 None 时 total_score=None
    """

    def __init__(self, policy: ScoringPolicy):
        self.policy = policy

    def calculate(self, dimensions: dict[str, DimensionResult]) -> ScoreCalculation:
        """计算加权总分

        Args:
            dimensions: 维度名 -> DimensionResult 映射

        Returns:
            ScoreCalculation 包含 total_score 和详细状态
        """
        dim_scores = {
            name: dimensions[name].score
            for name in self.policy.weights
            if name in dimensions
        }

        # 1. 检查必需维度
        for required in self.policy.required_dimensions:
            dim = dimensions.get(required)
            if dim is None or dim.status != "scored":
                return ScoreCalculation(
                    total_score=None,
                    status="incomplete",
                    reason=f"required_dimension_{dim.status if dim else 'missing'}:{required}",
                    weights_used=self.policy.weights,
                    dimension_scores=dim_scores,
                )

        # 2. 收集有效维度（status="scored" 且有 score）
        valid = {}
        for name, weight in self.policy.weights.items():
            dim = dimensions.get(name)
            if dim and dim.status == "scored" and dim.score is not None:
                valid[name] = dim.score

        # 3. 全部 None -> total=None
        if not valid:
            return ScoreCalculation(
                total_score=None,
                status="incomplete",
                reason="all_dimensions_unscored",
                weights_used=self.policy.weights,
                dimension_scores=dim_scores,
            )

        # 4. 使用固定权重计算（不做临时权重重分配）
        # 只有当所有维度都有效时，才使用标准加权求和
        # 如果有维度缺失，total_score=None（因为权重分配不完整）
        if len(valid) == len(self.policy.weights):
            # 所有维度都有分数，正常加权
            total = sum(valid[k] * w for k, w in self.policy.weights.items())
            total_score = round(total)
        else:
            # 部分维度缺失 -> 本阶段策略：total_score=None
            # 后续可引入批准的"部分评分"策略版本
            total_score = None

        return ScoreCalculation(
            total_score=total_score,
            status="completed" if total_score is not None else "incomplete",
            reason=None if total_score is not None else "partial_dimensions",
            weights_used=self.policy.weights,
            dimension_scores=dim_scores,
        )
