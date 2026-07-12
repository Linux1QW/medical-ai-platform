# -*- coding: utf-8 -*-
"""统一 JSON 提取工具 — 三层解析策略"""

import json
import re
import logging
from typing import Any, Union

logger = logging.getLogger(__name__)


def extract_json_from_text(
    text: str,
    *,
    default: Any = None,
    raise_on_failure: bool = True,
) -> Union[dict, list]:
    """从 LLM 输出文本中提取 JSON 对象或数组。

    三层解析策略：
    1. 直接 json.loads
    2. 去除 markdown 代码块后重试
    3. 贪婪正则提取最外层 {...}

    Args:
        text: 待解析的文本
        default: 解析失败时的默认返回值（raise_on_failure=False 时生效）
        raise_on_failure: 解析失败时是否抛出 ValueError

    Returns:
        解析后的 dict 或 list
    """
    if not text or not text.strip():
        if raise_on_failure:
            raise ValueError("LLM 返回内容为空")
        return default if default is not None else {}

    # Layer 1: 直接解析
    try:
        return json.loads(text.strip())
    except (json.JSONDecodeError, ValueError):
        pass

    # Layer 2: 去除 markdown 代码块后重试
    cleaned = re.sub(r"```(?:json)?\s*", "", text).strip().rstrip("`")
    try:
        return json.loads(cleaned)
    except (json.JSONDecodeError, ValueError):
        pass

    # Layer 3: 贪婪正则提取最外层 JSON 对象
    try:
        match = re.search(r"(\{.*\})", cleaned, re.DOTALL)
        if match:
            return json.loads(match.group(1))
    except (json.JSONDecodeError, AttributeError):
        pass

    # 所有层都失败
    if raise_on_failure:
        raise ValueError(f"无法解析 JSON: {text[:200]}...")

    logger.warning("JSON extraction failed, returning default value")
    return default if default is not None else {}
