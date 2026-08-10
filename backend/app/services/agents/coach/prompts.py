"""Prompt templates for coach agent LLM calls."""
from __future__ import annotations

INTENT_CLASSIFICATION_PROMPT = """你是一个临床问诊意图分类器。
根据以下对话内容，判断医生下一条消息最可能的意图。

可用的意图标签：
{intent_labels}

对话上下文：
{context}

医生最新消息：{message}

请只返回一个意图标签，不要解释。"""

FOLLOWUP_GENERATION_PROMPT = """你是一个问诊教练。根据当前问诊阶段和已收集的信息，建议下一步应该询问的内容。

当前阶段：{stage}
已收集的槽位信息：{slots}
尚未覆盖的阶段：{unvisited_stages}

请给出1-3个建议，每个包含意图标签和简要理由。"""
