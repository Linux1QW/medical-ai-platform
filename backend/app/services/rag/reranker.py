# -*- coding: utf-8 -*-
"""Cross-Encoder 重排序模块 — 使用 LLM 对检索结果进行精细化相关性评分

通过将 query 和 document 同时输入 LLM，获得更精准的相关性判断，
弥补向量检索"粗召回"的不足。
"""

import asyncio
import json
import logging
import re
from typing import Dict, List

from app.services.qwen_client import call_qwen_chat

logger = logging.getLogger(__name__)

# ── 重排序 System Prompt ──
RERANK_SYSTEM_PROMPT = """你是一名医学文献相关性评估专家。你的任务是判断给定的医学证据文本与查询的相关程度。

请对每条证据与查询的相关性打分，评分范围 0-10：
- 9-10：高度相关，证据直接回答或支持查询内容
- 7-8：较相关，证据涉及查询的核心主题
- 5-6：部分相关，证据涉及相关领域但不直接相关
- 3-4：弱相关，仅有少量关联
- 0-2：不相关

输出要求：
- 严格输出 JSON 数组，每个元素为整数评分
- 数组长度必须与证据条数一致
- 示例：[9, 7, 3, 8, 2]
- 不要输出任何解释性文字"""

# 最大并发重排序的文档数量（避免 prompt 过长）
MAX_RERANK_DOCS = 10

# 相关性阈值：低于此分数的文档将被过滤
RELEVANCE_THRESHOLD = 4


async def rerank_documents(
    query: str,
    documents: List[Dict],
    top_k: int = 5,
    threshold: float = RELEVANCE_THRESHOLD,
) -> List[Dict]:
    """使用 LLM Cross-Encoder 对检索结果进行重排序

    Args:
        query: 查询文本
        documents: 候选文档列表，每个文档需包含 "text" 字段
        top_k: 返回的最大文档数
        threshold: 最低相关性分数阈值（0-10），低于此分的文档将被过滤

    Returns:
        重排序并过滤后的文档列表，每个文档增加 "rerank_score" 字段
    """
    if not documents or not query or not query.strip():
        return documents[:top_k] if documents else []

    # 限制参与重排序的文档数量
    candidates = documents[:MAX_RERANK_DOCS]

    # 构建评分请求
    evidence_parts = []
    for i, doc in enumerate(candidates, 1):
        text = doc.get("text", "")
        # 截取前300字符避免 prompt 过长
        truncated = text[:300] + "..." if len(text) > 300 else text
        evidence_parts.append(f"证据{i}：{truncated}")

    evidence_text = "\n\n".join(evidence_parts)

    user_content = (
        f"【查询】\n{query}\n\n"
        f"【待评分的证据（共{len(candidates)}条）】\n{evidence_text}\n\n"
        f"请对以上{len(candidates)}条证据与查询的相关性分别打分（0-10整数），输出JSON数组。"
    )

    messages = [
        {"role": "system", "content": RERANK_SYSTEM_PROMPT},
        {"role": "user", "content": user_content},
    ]

    try:
        response = await call_qwen_chat(
            messages=messages,
            temperature=0.1,  # 低温度确保评分稳定
            max_tokens=200,
        )

        # 解析评分数组
        scores = _parse_scores(response, len(candidates))

        if scores is None:
            logger.warning("重排序评分解析失败，保持原始排序")
            return documents[:top_k]

        # 为每个文档附加重排序分数
        scored_docs = []
        for doc, score in zip(candidates, scores):
            doc_copy = dict(doc)
            doc_copy["rerank_score"] = score
            scored_docs.append(doc_copy)

        # 过滤低相关性文档
        filtered_docs = [d for d in scored_docs if d["rerank_score"] >= threshold]

        # 按重排序分数降序排列
        filtered_docs.sort(key=lambda x: x["rerank_score"], reverse=True)

        logger.info(
            f"Cross-Encoder 重排序完成：{len(candidates)} 条候选 → "
            f"过滤后 {len(filtered_docs)} 条（阈值={threshold}），返回 top_{top_k}"
        )

        return filtered_docs[:top_k]

    except Exception as e:
        logger.warning(f"Cross-Encoder 重排序失败，保持原始排序: {e}")
        return documents[:top_k]


def _parse_scores(response: str, expected_count: int) -> List[int]:
    """从 LLM 响应中解析评分数组

    Args:
        response: LLM 原始响应文本
        expected_count: 期望的评分数量

    Returns:
        评分列表，解析失败返回 None
    """
    if not response or not response.strip():
        return None

    # 1. 直接尝试解析
    try:
        scores = json.loads(response.strip())
        if isinstance(scores, list) and len(scores) == expected_count:
            return [max(0, min(10, int(s))) for s in scores]
    except (json.JSONDecodeError, ValueError, TypeError):
        pass

    # 2. 尝试提取 JSON 数组
    match = re.search(r'\[[\d\s,]+\]', response)
    if match:
        try:
            scores = json.loads(match.group())
            if isinstance(scores, list) and len(scores) == expected_count:
                return [max(0, min(10, int(s))) for s in scores]
        except (json.JSONDecodeError, ValueError, TypeError):
            pass

    # 3. 尝试提取所有数字
    numbers = re.findall(r'\b(\d+)\b', response)
    if len(numbers) >= expected_count:
        scores = [max(0, min(10, int(n))) for n in numbers[:expected_count]]
        return scores

    return None
