# -*- coding: utf-8 -*-
"""反思智能体 — 基于 ReAct 架构的评估结果自我反思与验证

核心功能：
1. 接收所有维度的评估结果
2. 使用 ReAct 推理链检查评分一致性、证据充分性、逻辑矛盾
3. 输出反思报告（不替代原始评分，仅作为辅助参考）

设计原则：
- 反思结果用于辅助，不替代现有确定性评分
- 保持评分的确定性，ScoreCalculator 的结果不受影响
- 反思发现的问题标记为 review flags，供人工参考
"""

import json
import re
import time
import uuid
import logging
from typing import Optional

from app.utils.json_parser import extract_json_from_text

from app.core.config import settings
from app.services.qwen_client import call_qwen_chat
from app.services.tools.base import ToolContext
from app.services.tools.registry import ToolRegistry
from app.services.tools.executor import ToolExecutor
from app.services.tools.budget import ToolBudget
from app.services.tools.consistency import register_consistency_tools

logger = logging.getLogger(__name__)


# ── System Prompt ─────────────────────────────────────────────────────────────

REFLECTION_SYSTEM_PROMPT = """你是一名医学评估质量审核专家，负责使用 ReAct 框架对临床问诊评估结果进行反思和验证。

你的任务是：
1. 检查各维度评分之间是否存在逻辑矛盾
2. 评估每个维度的证据是否充分
3. 发现可能被遗漏的问题或风险
4. 给出整体评估质量的评价

推理必须遵循 ReAct 格式：

Thought: [分析当前状态，说明为什么要检查某个方面]
Action: [工具名称]
Action Input: [工具参数的 JSON 对象]

收到 Observation 后继续推理。当完成所有检查后，输出最终反思报告：

Thought: [总结所有检查结果]
Final Answer: [严格 JSON 格式的反思报告]

可用工具：
1. check_score_consistency — 检查各维度评分的一致性
   Action Input: {"dimension_scores": [...], "threshold": 0.3}
2. check_evidence_sufficiency — 检查证据充分性
   Action Input: {"dimension_scores": [...], "min_score_threshold": 60.0}
3. detect_score_contradictions — 检测评分矛盾
   Action Input: {"dimension_scores": [...], "contradiction_rules": []}
4. summarize_evaluation — 汇总评估结果
   Action Input: {"dimension_scores": [...], "total_score": 75.0, "include_recommendations": true}

反思规则：
- 必须使用工具进行客观检查，不能仅凭主观判断
- 每个发现必须有工具检查结果作为依据
- 反思结果不改变原始评分，仅标记需要关注的问题
- 如果评估质量良好，应明确说明无需额外关注

最终反思报告 JSON 格式：
{
  "overall_quality": "good | acceptable | needs_attention | problematic",
  "confidence": 0.0-1.0,
  "issues_found": [
    {
      "type": "score_contradiction | insufficient_evidence | score_anomaly | missing_dimension",
      "severity": "low | medium | high",
      "description": "问题描述",
      "affected_dimensions": ["维度名"],
      "recommendation": "建议措施"
    }
  ],
  "consistency_score": 0.0-1.0,
  "evidence_adequacy_score": 0.0-1.0,
  "summary": "反思总结文本",
  "reasoning_steps": ["步骤1摘要", "步骤2摘要"]
}"""


# ── ReAct 步骤解析（复用 Knowledge Agent 的模式）─────────────────────────────


def _parse_react_step(text: str) -> dict:
    """解析 ReAct 推理步骤"""
    result = {
        "thought": "",
        "action": "",
        "action_input": {},
        "final_answer": None,
        "is_final": False,
    }

    thought_match = re.search(r"Thought:\s*(.+?)(?=\n(?:Action|Final Answer)|\Z)", text, re.DOTALL)
    if thought_match:
        result["thought"] = thought_match.group(1).strip()

    final_match = re.search(r"Final Answer:\s*(.+)", text, re.DOTALL)
    if final_match:
        result["is_final"] = True
        result["final_answer"] = final_match.group(1).strip()
        return result

    action_match = re.search(r"Action:\s*(\w+)", text)
    if action_match:
        result["action"] = action_match.group(1).strip()

    input_match = re.search(r"Action Input:\s*(\{.+?\})", text, re.DOTALL)
    if input_match:
        try:
            result["action_input"] = json.loads(input_match.group(1))
        except json.JSONDecodeError:
            raw = input_match.group(1)
            raw = re.sub(r",\s*}", "}", raw)
            raw = re.sub(r",\s*]", "]", raw)
            try:
                result["action_input"] = json.loads(raw)
            except json.JSONDecodeError:
                result["action_input"] = {}

    return result


def _extract_json(text: str) -> dict:
    """从 LLM 返回文本中提取 JSON"""
    return extract_json_from_text(text)


# ── 辅助类：ToolExecutor 桥接器 ─────────────────────────────────────────────


class _ToolExecutorBridge:
    """桥接工具执行器"""

    def __init__(self, executor: ToolExecutor, context: ToolContext, budget: ToolBudget):
        self.executor = executor
        self.context = context
        self.budget = budget

    async def execute(self, tool_name: str, arguments_json: str) -> dict:
        return await self.executor.execute(
            tool_name, arguments_json,
            context=self.context,
            budget=self.budget,
        )


# ── 主函数 ────────────────────────────────────────────────────────────────────


async def run_reflection(
    dimension_results: dict,
    total_score: Optional[float] = None,
) -> dict:
    """执行反思评估

    使用 ReAct 推理链检查评估结果的一致性、证据充分性和逻辑矛盾。

    Args:
        dimension_results: 各维度评估结果 {dim_name: DimensionResult}
        total_score: 总分（由 ScoreCalculator 计算）

    Returns:
        dict 包含反思报告
    """
    try:
        # ── Step 1: 构建维度评分列表 ──
        dimension_scores = []
        for dim_name, dim_result in dimension_results.items():
            score = getattr(dim_result, "score", None)
            status = getattr(dim_result, "status", "unknown")
            analysis = getattr(dim_result, "analysis", "")
            dimension_scores.append({
                "dimension": dim_name,
                "score": score,
                "status": status,
                "analysis": analysis[:300],
            })

        if not dimension_scores:
            return _build_no_data_result()

        logger.info(
            f"[Reflection] 开始反思评估：{len(dimension_scores)} 个维度, "
            f"总分={total_score}"
        )

        # ── Step 2: 构造工具环境 ──
        context = ToolContext(
            run_id=str(uuid.uuid4()),
            agent_name="reflection_agent",
            budgets={
                "check_score_consistency": 2,
                "check_evidence_sufficiency": 2,
                "detect_score_contradictions": 2,
                "summarize_evaluation": 1,
            },
            allowed_citation_ids=set(),
            evidence_cache={},
        )

        registry = ToolRegistry()
        register_consistency_tools(registry)
        executor = ToolExecutor(registry, max_result_chars=3000)
        budget = ToolBudget(context.budgets)
        bridge = _ToolExecutorBridge(executor, context, budget)

        # ── Step 3: 构建初始输入 ──
        dim_summary = json.dumps(dimension_scores, ensure_ascii=False, indent=2)
        user_content = (
            f"【评估结果数据】\n{dim_summary}\n\n"
            f"【总分】{total_score if total_score is not None else '未计算'}\n\n"
            "请使用 ReAct 框架逐步检查此评估结果的质量，"
            "包括评分一致性、证据充分性和逻辑矛盾。"
        )

        # ── Step 4: ReAct 推理循环 ──
        max_steps = settings.REACT_MAX_STEPS
        messages = [
            {"role": "system", "content": REFLECTION_SYSTEM_PROMPT},
            {"role": "user", "content": user_content},
        ]

        react_trace = []
        final_parsed = None
        step_count = 0

        for step_idx in range(max_steps):
            step_count += 1
            step_start = time.monotonic()

            response = await call_qwen_chat(messages, temperature=0.2)
            elapsed_ms = round((time.monotonic() - step_start) * 1000, 1)

            parsed_step = _parse_react_step(response)
            react_trace.append({
                "step": step_idx + 1,
                "thought": parsed_step["thought"][:200],
                "action": parsed_step["action"],
                "elapsed_ms": elapsed_ms,
            })

            if parsed_step["is_final"]:
                try:
                    final_parsed = _extract_json(parsed_step["final_answer"])
                except ValueError:
                    try:
                        final_parsed = _extract_json(response)
                    except ValueError:
                        final_parsed = None
                break

            if parsed_step["action"] and parsed_step["action_input"]:
                tool_name = parsed_step["action"]
                tool_args = parsed_step["action_input"]

                # 自动注入 dimension_scores（如果工具需要但未提供）
                if "dimension_scores" not in tool_args:
                    tool_args["dimension_scores"] = dimension_scores

                try:
                    tool_result = await bridge.execute(
                        tool_name,
                        json.dumps(tool_args, ensure_ascii=False),
                    )
                    observation = json.dumps(tool_result, ensure_ascii=False, default=str)
                    if len(observation) > 3000:
                        observation = observation[:3000] + "...(结果已截断)"
                except Exception as e:
                    observation = f"工具执行失败: {type(e).__name__}: {str(e)[:200]}"
                    logger.warning(f"[Reflection] 工具 {tool_name} 执行失败: {e}")

                react_trace[-1]["observation_summary"] = observation[:200]

                messages.append({"role": "assistant", "content": response})
                messages.append({
                    "role": "user",
                    "content": f"Observation: {observation}\n\n请继续推理。如果检查完成，输出 Final Answer。",
                })
            else:
                messages.append({"role": "assistant", "content": response})
                messages.append({
                    "role": "user",
                    "content": "请继续你的推理。使用工具检查评估结果，或输出 Final Answer。",
                })

        # ── Step 5: 处理结果 ──
        if final_parsed is None:
            logger.warning(f"[Reflection] 达到最大步数 {max_steps}，尝试强制获取最终答案")
            messages.append({
                "role": "user",
                "content": "推理步骤已达上限。请立即基于已收集的检查结果输出 Final Answer（JSON 格式）。",
            })
            forced_response = await call_qwen_chat(messages, temperature=0.1)
            try:
                final_parsed = _extract_json(forced_response)
            except ValueError:
                return _build_fallback_result(dimension_scores, react_trace)

        # ── Step 6: 校验和构建返回结构 ──
        overall_quality = final_parsed.get("overall_quality", "acceptable")
        valid_qualities = {"good", "acceptable", "needs_attention", "problematic"}
        if overall_quality not in valid_qualities:
            overall_quality = "acceptable"

        confidence = float(final_parsed.get("confidence", 0.5))
        confidence = max(0.0, min(1.0, confidence))

        issues_found = final_parsed.get("issues_found", [])
        consistency_score = float(final_parsed.get("consistency_score", 0.5))
        consistency_score = max(0.0, min(1.0, consistency_score))
        evidence_adequacy_score = float(final_parsed.get("evidence_adequacy_score", 0.5))
        evidence_adequacy_score = max(0.0, min(1.0, evidence_adequacy_score))
        summary = final_parsed.get("summary", "")

        # 确定是否需要人工复核
        needs_review = overall_quality in ("needs_attention", "problematic")
        review_reasons = []
        for issue in issues_found:
            if issue.get("severity") in ("medium", "high"):
                review_reasons.append(issue.get("description", ""))

        result = {
            "overall_quality": overall_quality,
            "confidence": confidence,
            "issues_found": issues_found,
            "consistency_score": consistency_score,
            "evidence_adequacy_score": evidence_adequacy_score,
            "summary": summary,
            "needs_review": needs_review,
            "review_reasons": review_reasons,
            "react_trace": react_trace,
            "react_steps_count": step_count,
            "dimension_count": len(dimension_scores),
        }

        logger.info(
            f"[Reflection] 反思评估完成：quality={overall_quality}, "
            f"confidence={confidence:.2f}, issues={len(issues_found)}, "
            f"steps={step_count}"
        )
        return result

    except Exception as e:
        logger.error(f"[Reflection] 反思评估异常: {e}", exc_info=True)
        return _build_error_result(str(e))


# ── 辅助函数 ─────────────────────────────────────────────────────────────────


def _build_no_data_result() -> dict:
    """无数据时的默认返回"""
    return {
        "overall_quality": "acceptable",
        "confidence": 0.3,
        "issues_found": [],
        "consistency_score": 0.5,
        "evidence_adequacy_score": 0.3,
        "summary": "无评估数据可供反思",
        "needs_review": False,
        "review_reasons": [],
        "react_trace": [],
        "react_steps_count": 0,
        "dimension_count": 0,
    }


def _build_fallback_result(dimension_scores: list, react_trace: list) -> dict:
    """ReAct 推理失败时的降级返回"""
    # 基于简单规则进行基础检查
    issues = []
    low_score_dims = [d for d in dimension_scores if d.get("score") is not None and d["score"] < 60]
    error_dims = [d for d in dimension_scores if d.get("status") in ("error", "insufficient")]

    if low_score_dims:
        issues.append({
            "type": "insufficient_evidence",
            "severity": "medium",
            "description": f"发现 {len(low_score_dims)} 个维度分数偏低",
            "affected_dimensions": [d["dimension"] for d in low_score_dims],
            "recommendation": "建议关注低分维度的具体原因",
        })

    if error_dims:
        issues.append({
            "type": "missing_dimension",
            "severity": "high",
            "description": f"发现 {len(error_dims)} 个维度评估未完成",
            "affected_dimensions": [d["dimension"] for d in error_dims],
            "recommendation": "需要补充信息重新评估",
        })

    quality = "good" if not issues else ("needs_attention" if len(issues) > 1 else "acceptable")

    return {
        "overall_quality": quality,
        "confidence": 0.4,
        "issues_found": issues,
        "consistency_score": 0.5,
        "evidence_adequacy_score": 0.4,
        "summary": "反思推理未能完成，已使用基础规则进行检查",
        "needs_review": quality == "needs_attention",
        "review_reasons": [i["description"] for i in issues if i.get("severity") in ("medium", "high")],
        "react_trace": react_trace,
        "react_steps_count": len(react_trace),
        "dimension_count": len(dimension_scores),
    }


def _build_error_result(error_msg: str) -> dict:
    """错误时的返回结构"""
    return {
        "overall_quality": "acceptable",
        "confidence": 0.3,
        "issues_found": [],
        "consistency_score": 0.5,
        "evidence_adequacy_score": 0.5,
        "summary": f"反思评估过程遇到技术问题: {error_msg[:100]}",
        "needs_review": False,
        "review_reasons": [],
        "react_trace": [],
        "react_steps_count": 0,
        "dimension_count": 0,
    }
