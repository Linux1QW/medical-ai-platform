# -*- coding: utf-8 -*-
"""评分摘要生成器 — LLM 摘要 + 确定性降级"""

import logging

from app.core.config import settings
from app.services.qwen_client import call_qwen_chat, call_qwen_with_tools
from app.services.prompts import get_prompt
from app.services.scoring.calculator import DimensionResult
from app.services.scoring.policies import ScoringPolicy
from app.utils.json_parser import extract_json_from_text

logger = logging.getLogger(__name__)

# ── LLM Prompts（复用 scoring_agent.py 原始 prompt，保持一致性）──

SYSTEM_PROMPT = get_prompt("scoring.summary_system")

FEWSHOT_USER = """【维度1 - 问诊技巧评估】
评分：72分
分析：问诊过程在病史采集方面表现中等偏上。医生大致遵循了从症状了解到检查解读的流程，详细询问了大便次数、性状、排气情况、腹痛部位和性质、体重变化，并追问了进食后症状加重的特点和生活方式。但未严格按照标准问诊顺序，既往史、个人史和家族史缺失。

【维度2 - 医学知识评估】
评分：78分
分析：医学知识应用表现良好。医生正确识别了肠道功能紊乱的可能性，结合肠镜结果排除了器质性病变。对息肉切除后的随访安排合理。但未详细评估肠易激综合征的Rome IV诊断标准，未考虑食物不耐受的可能。

【维度3 - 人文关怀评估】
评分：73分
分析：沟通整体表现尚可。医生态度平和亲切，未出现不耐烦或打断患者的情况。使用了通俗语言，避免了专业术语。但共情能力中等，用药告知不充分。

【维度4 - 诊断能力评估】
评分：75分
分析：诊断思维基本合理，能结合患者症状和检查结果进行综合分析，但鉴别诊断范围不够广，未充分考虑功能性胃肠病的亚型分类。

【维度5 - 治疗方案评估】
评分：70分
分析：治疗方案大体合理，药物选择基本恰当，但缺少非药物治疗建议，如饮食调整、心理疏导和生活方式改善等具体措施。

请生成综合评估摘要。"""

FEWSHOT_ASSISTANT = """{"summary": "该医生在本次问诊中的综合表现良好，加权总分73分。各维度按权重计算：问诊技巧72分x25%=18.0，医学知识78分x25%=19.5，人文关怀73分x20%=14.6，诊断能力75分x15%=11.3，治疗方案70分x15%=10.5。整体表现概述：本次问诊医生较为系统地了解了患者症状特征和检查结果，做出了合理的临床判断，各维度表现较为均衡，医学知识是亮点。主要优点：症状采集较详细，涵盖大便频次、性状、排气、体重变化等关键指标；诊断推理清晰，能结合检查结果综合分析；态度友善，语言通俗。主要不足：病史采集缺少既往史、个人史、家族史；鉴别诊断不够全面；治疗方案缺少非药物干预建议；用药告知不充分。各维度表现排名（从高到低）：医学知识（78分）、诊断能力（75分）、人文关怀（73分）、问诊技巧（72分）、治疗方案（70分）。建议重点加强问诊系统性训练、完善鉴别诊断和丰富治疗方案。"}"""


class SummaryGenerator:
    """评分摘要生成器"""

    def __init__(self, policy: ScoringPolicy):
        self.policy = policy

    async def generate(
        self,
        dimensions: dict[str, DimensionResult],
        total_score: int | None,
    ) -> str:
        """生成综合评估摘要

        优先调用 LLM，失败时降级为确定性模板摘要。
        """
        try:
            return await self._llm_summary(dimensions, total_score)
        except Exception as e:
            logger.error(f"LLM 摘要生成失败: {e}，使用降级摘要")
            return self._fallback_summary(dimensions, total_score)

    async def generate_with_tools(
        self,
        dimensions: dict[str, DimensionResult],
        total_score: int | None,
        tool_executor=None,
        context=None,
    ) -> str:
        """使用 Tool Use 增强摘要生成（可选）。

        如果 tool_executor 为 None 或 settings.ENABLE_TOOL_USE 为 False，
        则回退到普通 generate() 方法。

        增强点：
        - LLM 可调用 get_evaluation_criteria 了解评分标准
        - LLM 可调用 generate_improvement_plan 生成更精准的建议
        - 但 LLM 不能修改任何维度分数或总分

        关键约束：
        - total_score 和 dimension_results 中的值不可被 LLM 修改
        - 如果 total_score 为 None，摘要中必须说明"本次因证据不足/人工复核未生成总分"
        """
        # 如果未启用 Tool Use 或未提供执行器，回退到普通方法
        if tool_executor is None or not settings.ENABLE_TOOL_USE:
            return await self.generate(dimensions, total_score)

        try:
            return await self._llm_summary_with_tools(
                dimensions, total_score, tool_executor, context
            )
        except Exception as e:
            logger.error(f"Tool Use 摘要生成失败: {e}，回退到普通方法")
            return await self.generate(dimensions, total_score)

    async def _llm_summary_with_tools(
        self, dimensions, total_score, tool_executor, context
    ) -> str:
        """使用 Tool Use 调用 LLM 生成摘要"""
        from app.services.tools.scoring import (
            GenerateImprovementPlan,
            GetEvaluationCriteria,
        )

        # 构建维度信息（与普通方法相同）
        blocks = []
        dimension_scores = {}
        for i, (name, label) in enumerate(
            [(k, v) for k, v in self.policy.dimension_names.items()], 1
        ):
            dim = dimensions.get(name)
            if dim and dim.status == "scored" and dim.score is not None:
                blocks.append(f"【维度{i} - {label}】\n评分：{dim.score}分\n分析：{dim.analysis}")
                dimension_scores[name] = dim.score
            else:
                status_desc = {
                    "not_applicable": "不适用",
                    "not_submitted": "未提交",
                    "insufficient": "证据不足",
                    "error": "执行异常",
                }.get(dim.status if dim else "missing", "未评估")
                blocks.append(f"【维度{i} - {label}】\n该维度{status_desc}，不计入加权总分。")
                dimension_scores[name] = None

        # 构建 system prompt，强调分数不可修改
        system_prompt = SYSTEM_PROMPT + """

重要约束：
- 你可以通过工具获取评分标准和生成改进建议
- 但你不能修改任何维度分数或总分
- 总分是由系统确定性计算的，你必须使用系统提供的总分
"""
        if total_score is None:
            system_prompt += """
- 本次评估未生成总分（可能因证据不足或人工复核），请在摘要中明确说明"本次因证据不足/人工复核未生成总分"
"""

        user_content = "\n\n".join(blocks) + "\n\n请生成综合评估摘要。"
        if total_score is not None:
            user_content += f"\n\n系统计算的加权总分为：{total_score}分（此分数不可修改）。"
        else:
            user_content += "\n\n本次评估未生成总分，请在摘要中说明。"

        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": FEWSHOT_USER},
            {"role": "assistant", "content": FEWSHOT_ASSISTANT},
            {"role": "user", "content": user_content},
        ]

        # 获取工具 schema
        tools = [
            GetEvaluationCriteria(self.policy).openai_schema(),
            GenerateImprovementPlan(self.policy).openai_schema(),
        ]

        # 调用 LLM with tools
        result = await call_qwen_with_tools(
            messages=messages,
            tools=tools,
            tool_executor=tool_executor,
            temperature=0.3,
            max_tokens=2000,
        )

        # 提取摘要
        content = result.content
        if result.degraded or not content:
            raise ValueError(f"Tool Use 调用失败或返回为空: {result.error}")

        data = _extract_json(content)
        summary = data.get("summary", "")
        if not summary:
            raise ValueError("LLM summary 为空")

        # 关键：确保 total_score=None 时摘要中包含说明
        if total_score is None:
            if "未生成总分" not in summary and "证据不足" not in summary and "人工复核" not in summary:
                summary = "本次因证据不足/人工复核未生成总分。" + summary

        return summary

    async def _llm_summary(self, dimensions, total_score) -> str:
        """调用 LLM 生成摘要"""
        blocks = []
        for i, (name, label) in enumerate(
            [(k, v) for k, v in self.policy.dimension_names.items()], 1
        ):
            dim = dimensions.get(name)
            if dim and dim.status == "scored" and dim.score is not None:
                blocks.append(f"【维度{i} - {label}】\n评分：{dim.score}分\n分析：{dim.analysis}")
            else:
                status_desc = {
                    "not_applicable": "不适用",
                    "not_submitted": "未提交",
                    "insufficient": "证据不足",
                    "error": "执行异常",
                }.get(dim.status if dim else "missing", "未评估")
                blocks.append(f"【维度{i} - {label}】\n该维度{status_desc}，不计入加权总分。")

        user_content = "\n\n".join(blocks) + "\n\n请生成综合评估摘要。"

        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": FEWSHOT_USER},
            {"role": "assistant", "content": FEWSHOT_ASSISTANT},
            {"role": "user", "content": user_content},
        ]

        result = await call_qwen_chat(messages, temperature=0.3)
        data = _extract_json(result)
        summary = data.get("summary", "")
        if not summary:
            raise ValueError("LLM summary 为空")
        return summary

    def _fallback_summary(self, dimensions, total_score) -> str:
        """确定性降级摘要 — 覆盖五个维度，正确表达各种状态"""
        valid_dims = []
        skipped_dims = []

        for name in self.policy.weights:
            label = self.policy.dimension_names.get(name, name)
            dim = dimensions.get(name)
            if dim and dim.status == "scored" and dim.score is not None:
                valid_dims.append((label, dim.score))
            else:
                status_desc = {
                    "not_applicable": "不适用",
                    "not_submitted": "未提交",
                    "insufficient": "证据不足",
                    "error": "执行异常",
                }.get(dim.status if dim else "missing", "未评估")
                skipped_dims.append(f"{label}（{status_desc}）")

        if total_score is None:
            if not valid_dims:
                return "所有维度均未评估，无法生成综合摘要。"
            ranking = "、".join(
                f"{name}（{score}分）" for name, score in sorted(valid_dims, key=lambda x: x[1], reverse=True)
            )
            skipped_note = f"以下维度未参与评分：{'、'.join(skipped_dims)}。" if skipped_dims else ""
            return (
                f"本次评估部分维度完成。已评分维度排名：{ranking}。"
                f"{skipped_note}由于存在未评分维度，综合总分暂未生成，待所有维度完成后确认。"
            )

        # total_score 有值时
        level = "优秀" if total_score >= 90 else "良好" if total_score >= 80 else "一般" if total_score >= 60 else "不及格"
        ranking = "、".join(
            f"{name}（{score}分）" for name, score in sorted(valid_dims, key=lambda x: x[1], reverse=True)
        )

        weight_details = ""
        for name in self.policy.weights:
            dim = dimensions.get(name)
            if dim and dim.status == "scored" and dim.score is not None:
                w = self.policy.weights[name]
                weight_details += f"{self.policy.dimension_names.get(name, name)}{dim.score}分x{int(w*100)}%={dim.score * w:.1f}，"
        weight_details = weight_details.rstrip("，")

        skipped_note = f"以下维度未参与评分：{'、'.join(skipped_dims)}。" if skipped_dims else ""

        return (
            f"该医生在本次问诊中的综合表现为{level}，加权总分{total_score}分。"
            f"各维度按权重计算：{weight_details}。"
            f"各维度表现排名（从高到低）：{ranking}。"
            f"{skipped_note}"
        )


def _extract_json(text: str) -> dict:
    """从 LLM 返回文本中提取 JSON"""
    return extract_json_from_text(text)
