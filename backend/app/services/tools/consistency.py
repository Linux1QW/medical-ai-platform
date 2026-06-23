# -*- coding: utf-8 -*-
"""一致性检查工具集 — 供 Reflection Agent 通过 Function Calling 调用

包含 4 个工具：
1. CheckScoreConsistency — 检查评分维度间的一致性
2. CheckEvidenceSufficiency — 检查评估结果的证据充分性
3. DetectScoreContradictions — 检测评分矛盾（如诊断高分但知识核对低分）
4. SummarizeEvaluation — 汇总评估结果摘要
"""

import logging
from typing import List, Optional

from pydantic import BaseModel, Field

from app.services.tools.base import BaseTool, ToolContext
from app.services.tools.registry import ToolRegistry

logger = logging.getLogger(__name__)


# ── Args Schemas ──────────────────────────────────────────────────────────────


class CheckScoreConsistencyArgs(BaseModel):
    dimension_scores: List[dict] = Field(
        description="各维度评分结果列表，每项含 dimension, score, status, analysis",
    )
    threshold: float = Field(
        default=0.3,
        description="一致性偏差阈值（0-1），维度间分数差异超过此比例时标记为不一致",
    )


class CheckEvidenceSufficiencyArgs(BaseModel):
    dimension_scores: List[dict] = Field(
        description="各维度评分结果列表",
    )
    min_score_threshold: float = Field(
        default=60.0,
        description="最低可接受分数阈值，低于此分数的维度标记为证据不足",
    )


class DetectScoreContradictionsArgs(BaseModel):
    dimension_scores: List[dict] = Field(
        description="各维度评分结果列表",
    )
    contradiction_rules: List[dict] = Field(
        default_factory=list,
        description="矛盾检测规则，每项含 dim_a, dim_b, condition, message",
    )


class SummarizeEvaluationArgs(BaseModel):
    dimension_scores: List[dict] = Field(
        description="各维度评分结果列表",
    )
    total_score: Optional[float] = Field(
        default=None,
        description="总分（如有）",
    )
    include_recommendations: bool = Field(
        default=True,
        description="是否包含改进建议",
    )


# ── Tool 1: CheckScoreConsistency ────────────────────────────────────────────


class CheckScoreConsistency(BaseTool):
    name = "check_score_consistency"
    description = "检查各评估维度间分数的一致性，检测异常偏差"
    args_schema = CheckScoreConsistencyArgs
    timeout_seconds = 10
    critical = False

    async def execute(
        self, args: CheckScoreConsistencyArgs, context: ToolContext
    ) -> dict:
        """检查维度间评分一致性"""
        scores = []
        for item in args.dimension_scores:
            score = item.get("score")
            if score is not None:
                scores.append({
                    "dimension": item.get("dimension", "unknown"),
                    "score": float(score),
                    "status": item.get("status", "unknown"),
                })

        if len(scores) < 2:
            return {
                "consistent": True,
                "inconsistencies": [],
                "summary": "评分维度不足，无法进行一致性检查",
            }

        # 计算分数统计
        score_values = [s["score"] for s in scores]
        mean_score = sum(score_values) / len(score_values)
        max_score = max(score_values)
        min_score = min(score_values)
        score_range = max_score - min_score

        # 检测不一致：维度间差异超过阈值
        inconsistencies = []
        threshold_pct = args.threshold

        for i, s1 in enumerate(scores):
            for s2 in scores[i + 1:]:
                diff = abs(s1["score"] - s2["score"])
                # 使用相对差异（相对于满分100）
                relative_diff = diff / 100.0
                if relative_diff > threshold_pct:
                    inconsistencies.append({
                        "dim_a": s1["dimension"],
                        "score_a": s1["score"],
                        "dim_b": s2["dimension"],
                        "score_b": s2["score"],
                        "difference": round(diff, 1),
                        "relative_difference": round(relative_diff, 3),
                        "severity": "high" if relative_diff > 0.5 else "medium",
                    })

        is_consistent = len(inconsistencies) == 0

        return {
            "consistent": is_consistent,
            "inconsistencies": inconsistencies,
            "statistics": {
                "mean_score": round(mean_score, 1),
                "max_score": round(max_score, 1),
                "min_score": round(min_score, 1),
                "score_range": round(score_range, 1),
                "dimension_count": len(scores),
            },
            "summary": (
                f"共检查 {len(scores)} 个维度，"
                f"发现 {len(inconsistencies)} 处不一致"
            ),
        }


# ── Tool 2: CheckEvidenceSufficiency ─────────────────────────────────────────


class CheckEvidenceSufficiency(BaseTool):
    name = "check_evidence_sufficiency"
    description = "检查各评估维度的证据充分性，识别证据不足的维度"
    args_schema = CheckEvidenceSufficiencyArgs
    timeout_seconds = 10
    critical = False

    async def execute(
        self, args: CheckEvidenceSufficiencyArgs, context: ToolContext
    ) -> dict:
        """检查证据充分性"""
        insufficient_dims = []
        error_dims = []
        sufficient_dims = []

        for item in args.dimension_scores:
            dim = item.get("dimension", "unknown")
            score = item.get("score")
            status = item.get("status", "unknown")
            analysis = item.get("analysis", "")

            if status in ("error", "insufficient"):
                error_dims.append({
                    "dimension": dim,
                    "status": status,
                    "analysis": analysis[:200],
                })
            elif score is not None and score < args.min_score_threshold:
                insufficient_dims.append({
                    "dimension": dim,
                    "score": score,
                    "threshold": args.min_score_threshold,
                    "analysis": analysis[:200],
                })
            elif score is not None:
                sufficient_dims.append({
                    "dimension": dim,
                    "score": score,
                })

        total_dims = len(args.dimension_scores)
        sufficiency_ratio = len(sufficient_dims) / total_dims if total_dims > 0 else 0

        return {
            "overall_sufficient": len(error_dims) == 0 and len(insufficient_dims) == 0,
            "sufficient_dimensions": sufficient_dims,
            "insufficient_dimensions": insufficient_dims,
            "error_dimensions": error_dims,
            "sufficiency_ratio": round(sufficiency_ratio, 2),
            "summary": (
                f"共 {total_dims} 个维度：{len(sufficient_dims)} 个证据充分，"
                f"{len(insufficient_dims)} 个分数偏低，{len(error_dims)} 个存在错误"
            ),
        }


# ── Tool 3: DetectScoreContradictions ────────────────────────────────────────


class DetectScoreContradictions(BaseTool):
    name = "detect_score_contradictions"
    description = "检测评分维度间的逻辑矛盾，如诊断高分但知识核对低分"
    args_schema = DetectScoreContradictionsArgs
    timeout_seconds = 10
    critical = False

    async def execute(
        self, args: DetectScoreContradictionsArgs, context: ToolContext
    ) -> dict:
        """检测评分矛盾"""
        # 构建维度分数映射
        score_map = {}
        for item in args.dimension_scores:
            dim = item.get("dimension", "unknown")
            score = item.get("score")
            if score is not None:
                score_map[dim] = {
                    "score": float(score),
                    "status": item.get("status", "unknown"),
                    "analysis": item.get("analysis", "")[:200],
                }

        contradictions = []

        # 内置矛盾检测规则
        builtin_rules = [
            {
                "dim_a": "diagnosis",
                "dim_b": "knowledge",
                "condition": "high_low",
                "threshold": 30,
                "message": "诊断评分高但知识核对评分低，可能存在诊断正确但缺乏循证依据",
            },
            {
                "dim_a": "treatment",
                "dim_b": "knowledge",
                "condition": "high_low",
                "threshold": 30,
                "message": "治疗方案评分高但知识核对评分低，治疗方案可能缺乏指南支持",
            },
            {
                "dim_a": "inquiry",
                "dim_b": "diagnosis",
                "condition": "low_high",
                "threshold": 30,
                "message": "病史采集评分低但诊断评分高，诊断依据可能不充分",
            },
        ]

        # 合并自定义规则
        all_rules = builtin_rules + args.contradiction_rules

        for rule in all_rules:
            dim_a = rule.get("dim_a", "")
            dim_b = rule.get("dim_b", "")
            condition = rule.get("condition", "")
            threshold = rule.get("threshold", 30)
            message = rule.get("message", "")

            if dim_a not in score_map or dim_b not in score_map:
                continue

            score_a = score_map[dim_a]["score"]
            score_b = score_map[dim_b]["score"]
            diff = score_a - score_b

            is_contradiction = False
            if condition == "high_low" and diff > threshold:
                is_contradiction = True
            elif condition == "low_high" and diff < -threshold:
                is_contradiction = True
            elif condition == "any_diff" and abs(diff) > threshold:
                is_contradiction = True

            if is_contradiction:
                contradictions.append({
                    "dim_a": dim_a,
                    "score_a": score_a,
                    "dim_b": dim_b,
                    "score_b": score_b,
                    "difference": round(diff, 1),
                    "rule": condition,
                    "message": message,
                    "severity": "high" if abs(diff) > 50 else "medium",
                })

        return {
            "has_contradictions": len(contradictions) > 0,
            "contradictions": contradictions,
            "checked_rules": len(all_rules),
            "summary": (
                f"检测了 {len(all_rules)} 条矛盾规则，"
                f"发现 {len(contradictions)} 处矛盾"
            ),
        }


# ── Tool 4: SummarizeEvaluation ──────────────────────────────────────────────


class SummarizeEvaluation(BaseTool):
    name = "summarize_evaluation"
    description = "汇总评估结果，生成结构化摘要"
    args_schema = SummarizeEvaluationArgs
    timeout_seconds = 10
    critical = False

    async def execute(
        self, args: SummarizeEvaluationArgs, context: ToolContext
    ) -> dict:
        """汇总评估结果"""
        dimensions = []
        total_weighted = 0.0
        has_score = False

        # 维度权重（与评估系统一致）
        weights = {
            "inquiry": 0.25,
            "knowledge": 0.25,
            "humanistic": 0.20,
            "diagnosis": 0.15,
            "treatment": 0.15,
        }

        for item in args.dimension_scores:
            dim = item.get("dimension", "unknown")
            score = item.get("score")
            status = item.get("status", "unknown")
            weight = weights.get(dim, 0.1)

            dim_info = {
                "dimension": dim,
                "score": score,
                "status": status,
                "weight": weight,
            }
            dimensions.append(dim_info)

            if score is not None:
                total_weighted += score * weight
                has_score = True

        computed_total = round(total_weighted, 1) if has_score else None

        # 生成改进建议
        recommendations = []
        if args.include_recommendations:
            for dim_info in dimensions:
                if dim_info["score"] is not None and dim_info["score"] < 70:
                    recommendations.append(
                        f"{dim_info['dimension']}维度得分偏低({dim_info['score']}分)，建议重点改进"
                    )
                elif dim_info["status"] in ("error", "insufficient"):
                    recommendations.append(
                        f"{dim_info['dimension']}维度评估未完成({dim_info['status']})，需要补充信息"
                    )

        return {
            "dimensions": dimensions,
            "computed_total": computed_total,
            "provided_total": args.total_score,
            "total_match": (
                abs(computed_total - args.total_score) < 1.0
                if computed_total is not None and args.total_score is not None
                else None
            ),
            "recommendations": recommendations,
            "summary": (
                f"评估涵盖 {len(dimensions)} 个维度，"
                f"{'总分: ' + str(args.total_score) if args.total_score else '未计算总分'}"
            ),
        }


# ── 注册函数 ─────────────────────────────────────────────────────────────────


def register_consistency_tools(registry: ToolRegistry) -> None:
    """注册所有一致性检查工具"""
    tools = [
        CheckScoreConsistency(),
        CheckEvidenceSufficiency(),
        DetectScoreContradictions(),
        SummarizeEvaluation(),
    ]
    for tool in tools:
        try:
            registry.register(tool)
        except ValueError:
            pass  # 已注册则跳过
