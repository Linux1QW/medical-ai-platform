# -*- coding: utf-8 -*-
"""MQE（Multi-Query Expansion）— LLM 多查询扩展"""

import json
import logging
import re
from typing import List

from app.services.qwen_client import call_qwen_chat

logger = logging.getLogger(__name__)


async def expand_queries(original_query: str, n: int = 3) -> List[str]:
    """使用 LLM 将原始医学查询扩展为 n 条语义等价但措辞不同的查询变体

    Args:
        original_query: 原始医学查询文本
        n: 需要扩展的查询数量

    Returns:
        扩展后的查询列表（不包含原始查询）
    """
    system_prompt = f"""你是一个医学查询扩展专家。请将用户的医学查询扩展为 {n} 条语义等价但措辞不同的查询变体。

扩展方向包括：
1. 同义词替换（如"治疗"→"疗法"、"药物"→"药品"）
2. 中英文术语互换（如"非小细胞肺癌"→"NSCLC"、"靶向治疗"→"targeted therapy"）
3. 缩写展开（如"NSCLC"→"非小细胞肺癌"）
4. 上下位概念（如"肺癌"→"肺腺癌/肺鳞癌"、"EGFR突变"→"基因突变"）

要求：
- 扩展后的查询必须与原始查询语义等价，不能改变原意
- 每条扩展查询应该是完整的、可独立用于检索的短语
- 输出必须是严格的 JSON 数组格式，例如：["扩展查询1", "扩展查询2", "扩展查询3"]
- 不要包含任何解释性文字，只输出 JSON 数组"""

    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": f"请扩展以下医学查询：{original_query}"}
    ]

    try:
        response = await call_qwen_chat(
            messages=messages,
            temperature=0.7,
            max_tokens=2000
        )

        # 尝试从响应中提取 JSON 数组
        # 先尝试直接解析整个响应
        try:
            expanded = json.loads(response.strip())
            if isinstance(expanded, list):
                # 过滤掉非字符串项和空字符串
                expanded = [str(q).strip() for q in expanded if isinstance(q, (str,)) and str(q).strip()]
                logger.info(f"查询扩展成功：原始查询 '{original_query}' 扩展为 {len(expanded)} 条变体")
                return expanded
        except json.JSONDecodeError:
            pass

        # 尝试从响应中提取 JSON 数组（使用正则表达式）
        json_match = re.search(r'\[.*?\]', response, re.DOTALL)
        if json_match:
            try:
                expanded = json.loads(json_match.group())
                if isinstance(expanded, list):
                    expanded = [str(q).strip() for q in expanded if isinstance(q, (str,)) and str(q).strip()]
                    logger.info(f"查询扩展成功：原始查询 '{original_query}' 扩展为 {len(expanded)} 条变体")
                    return expanded
            except json.JSONDecodeError:
                pass

        logger.warning(f"查询扩展解析失败：无法从 LLM 响应中解析 JSON 数组，响应内容：{response[:200]}")
        return []

    except Exception as e:
        logger.warning(f"查询扩展失败：{e}")
        return []
