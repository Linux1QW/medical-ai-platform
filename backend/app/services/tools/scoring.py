# -*- coding: utf-8 -*-
"""评分相关只读工具 — 辅助 LLM 生成更好的摘要和建议

重要原则：
- total_score 仍由 ScoreCalculator 确定性计算，LLM 不允许自主计算总分
- 以下工具只用于辅助 LLM 生成更好的摘要和建议
- 所有工具均为只读，不修改任何评分状态
"""

import logging
from typing import Optional
from pydantic import BaseModel, Field

from .base import BaseTool, ToolContext
from app.utils.json_parser import extract_json_from_text
from .registry import ToolRegistry
from ..scoring.policies import ScoringPolicy, get_default_policy
from ..qwen_client import call_qwen_chat

logger = logging.getLogger(__name__)


# ── Tool 1: GetEvaluationCriteria ─────────────────────────────────────────────

class GetEvaluationCriteriaArgs(BaseModel):
    """查询评估维度的参数"""
    dimension: Optional[str] = Field(
        default=None,
        description="指定维度名（如 inquiry, knowledge, humanistic, diagnosis, treatment），为空则返回全部"
    )


class GetEvaluationCriteria(BaseTool):
    """查询各评估维度的评分标准和权重配置"""
    name: str = "get_evaluation_criteria"
    description: str = "查询各评估维度的评分标准和权重配置"
    args_schema: type[BaseModel] = GetEvaluationCriteriaArgs
    timeout_seconds: int = 10
    critical: bool = False

    def __init__(self, policy: ScoringPolicy | None = None):
        """初始化工具
        
        Args:
            policy: 评分策略，为 None 时使用默认策略
        """
        self.policy = policy or get_default_policy()

    async def execute(self, args: BaseModel, context: ToolContext) -> dict:
        """执行查询评估标准
        
        Returns:
            {
                "dimensions": {
                    "inquiry": {"weight": 0.25, "description": "问诊技巧评估"},
                    ...
                },
                "total_weight": 1.0,
                "policy_version": "v1"
            }
        """
        dimension_filter = getattr(args, "dimension", None)
        
        # 构建维度信息
        dimensions_info = {}
        for dim_name, weight in self.policy.weights.items():
            # 如果指定了维度过滤，只返回指定维度
            if dimension_filter and dim_name != dimension_filter:
                continue
            
            description = self.policy.dimension_names.get(dim_name, dim_name)
            dimensions_info[dim_name] = {
                "weight": weight,
                "description": description,
            }
        
        # 计算总权重
        total_weight = sum(dim["weight"] for dim in dimensions_info.values())
        
        return {
            "dimensions": dimensions_info,
            "total_weight": round(total_weight, 4),
            "policy_version": self.policy.version,
        }


# ── Tool 2: GenerateImprovementPlan ───────────────────────────────────────────

class GenerateImprovementPlanArgs(BaseModel):
    """生成改进建议的参数"""
    dimension_scores: dict[str, Optional[float]] = Field(
        ...,
        description="各维度得分，如 {'inquiry': 72.0, 'knowledge': 78.0, ...}"
    )
    focus_areas: Optional[list[str]] = Field(
        default=None,
        description="重点关注的维度列表，如 ['inquiry', 'treatment']"
    )


class GenerateImprovementPlan(BaseTool):
    """基于各维度评分结果生成个性化改进建议"""
    name: str = "generate_improvement_plan"
    description: str = "基于各维度评分结果生成个性化改进建议"
    args_schema: type[BaseModel] = GenerateImprovementPlanArgs
    timeout_seconds: int = 30
    critical: bool = False

    def __init__(self, policy: ScoringPolicy | None = None):
        """初始化工具
        
        Args:
            policy: 评分策略，为 None 时使用默认策略
        """
        self.policy = policy or get_default_policy()

    async def execute(self, args: BaseModel, context: ToolContext) -> dict:
        """执行生成改进建议
        
        Returns:
            {
                "improvement_suggestions": [
                    {"dimension": "inquiry", "priority": "high", "suggestion": "..."},
                    ...
                ],
                "overall_recommendation": "..."
            }
        """
        dimension_scores = getattr(args, "dimension_scores", {})
        focus_areas = getattr(args, "focus_areas", None)
        
        # 构建维度得分信息
        scored_dims = []
        unscored_dims = []
        for dim_name, score in dimension_scores.items():
            description = self.policy.dimension_names.get(dim_name, dim_name)
            if score is not None:
                scored_dims.append((dim_name, description, score))
            else:
                unscored_dims.append((dim_name, description))
        
        # 按分数排序（低分优先）
        scored_dims.sort(key=lambda x: x[2])
        
        # 构建 prompt
        prompt_parts = [
            "你是一名临床问诊评估专家，请根据以下各维度评分结果，生成个性化的改进建议。",
            "",
            "【评分结果】"
        ]
        
        for dim_name, description, score in scored_dims:
            priority = "高" if score < 70 else "中" if score < 80 else "低"
            prompt_parts.append(f"- {description}({dim_name}): {score}分，改进优先级: {priority}")
        
        if unscored_dims:
            prompt_parts.append("")
            prompt_parts.append("【未评分维度】")
            for dim_name, description in unscored_dims:
                prompt_parts.append(f"- {description}({dim_name}): 未评分")
        
        if focus_areas:
            prompt_parts.append("")
            prompt_parts.append("【重点关注领域】")
            for area in focus_areas:
                desc = self.policy.dimension_names.get(area, area)
                prompt_parts.append(f"- {desc}")
        
        prompt_parts.extend([
            "",
            "请生成改进建议，要求：",
            "1. 针对每个得分低于80分的维度给出具体改进建议",
            "2. 优先级划分：低于70分为high，70-79分为medium，80分以上为low",
            "3. 建议应具体、可操作，避免泛泛而谈",
            "4. 最后给出一条总体建议",
            "",
            "请以JSON格式返回：",
            '{',
            '  "improvement_suggestions": [',
            '    {"dimension": "维度名", "priority": "high/medium/low", "suggestion": "具体建议"},',
            '    ...',
            '  ],',
            '  "overall_recommendation": "总体建议"',
            '}',
        ])
        
        prompt = "\n".join(prompt_parts)
        
        try:
            messages = [
                {"role": "system", "content": "你是临床问诊评估专家，负责生成专业的改进建议。"},
                {"role": "user", "content": prompt},
            ]
            
            result = await call_qwen_chat(messages, temperature=0.3, max_tokens=1500)
            
            # 解析 JSON
            data = _extract_json(result)
            
            # 验证和规范化输出
            suggestions = []
            for item in data.get("improvement_suggestions", []):
                dim = item.get("dimension", "")
                priority = item.get("priority", "medium")
                suggestion = item.get("suggestion", "")
                
                # 规范化优先级
                if priority not in ("high", "medium", "low"):
                    priority = "medium"
                
                if dim and suggestion:
                    suggestions.append({
                        "dimension": dim,
                        "priority": priority,
                        "suggestion": suggestion,
                    })
            
            overall = data.get("overall_recommendation", "")
            
            return {
                "improvement_suggestions": suggestions,
                "overall_recommendation": overall,
            }
            
        except Exception as e:
            logger.error(f"GenerateImprovementPlan LLM 调用失败: {e}")
            # 降级：基于规则生成简单建议
            return self._fallback_suggestions(scored_dims, focus_areas)
    
    def _fallback_suggestions(
        self, 
        scored_dims: list[tuple[str, str, float]], 
        focus_areas: list[str] | None
    ) -> dict:
        """降级建议生成（LLM 失败时使用）"""
        suggestions = []
        
        for dim_name, description, score in scored_dims:
            if score < 70:
                priority = "high"
                suggestion = f"{description}得分较低({score}分)，建议重点加强该维度的训练和学习。"
            elif score < 80:
                priority = "medium"
                suggestion = f"{description}得分中等({score}分)，建议进一步优化该维度的表现。"
            else:
                priority = "low"
                suggestion = f"{description}表现良好({score}分)，建议继续保持。"
            
            suggestions.append({
                "dimension": dim_name,
                "priority": priority,
                "suggestion": suggestion,
            })
        
        # 如果有重点关注领域，优先提及
        overall_parts = []
        if focus_areas:
            for area in focus_areas:
                desc = self.policy.dimension_names.get(area, area)
                overall_parts.append(f"建议重点关注{desc}领域")
        
        if not overall_parts:
            overall_parts.append("建议综合各维度表现，制定系统性的改进计划。")
        
        return {
            "improvement_suggestions": suggestions,
            "overall_recommendation": "。".join(overall_parts) + "。",
        }


# ── JSON 提取辅助函数 ─────────────────────────────────────────────────────────

def _extract_json(text: str) -> dict:
    """从 LLM 返回文本中提取 JSON"""
    return extract_json_from_text(text)


# ── 注册函数 ──────────────────────────────────────────────────────────────────

def register_scoring_tools(registry: ToolRegistry, policy: ScoringPolicy | None = None) -> None:
    """注册评分相关工具到工具注册表
    
    Args:
        registry: 工具注册表
        policy: 评分策略，为 None 时使用默认策略
    """
    registry.register(GetEvaluationCriteria(policy))
    registry.register(GenerateImprovementPlan(policy))
