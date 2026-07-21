# -*- coding: utf-8 -*-
"""Safety 工具边界说明

Safety 首期不引入 LLM 自主 Tool Use。
红旗规则扫描由代码确定性执行，调用顺序强制为：
  run_safety_check() → 确定性红旗规则扫描 → 高风险直接 review

本文件保留为未来扩展接口，当前不注册任何 LLM 可调用的工具。

设计原则：
1. Safety 检查完全由确定性代码执行，不依赖 LLM 判断
2. 红旗规则命中时直接标记，LLM 不得降级
3. 未来如需扩展 Safety 工具，必须经过严格审核
"""

from ..agents.safety_agent import run_safety_check

# 显式导出，方便未来扩展
__all__ = ["run_safety_check"]


def register_safety_tools(registry) -> None:
    """注册 Safety 相关工具到工具注册表

    注意：当前 Safety 不引入 LLM 自主 Tool Use。
    此函数保留为空实现，未来如需扩展，需经过严格审核。

    Args:
        registry: 工具注册表（ToolRegistry 实例）
    """
    # 显式不注册任何工具
    # Safety 首期设计为纯代码确定性执行，不暴露给 LLM 自主调用
    pass
