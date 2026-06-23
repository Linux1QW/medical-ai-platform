from .base import BaseTool, ToolContext
from .registry import ToolRegistry
from .executor import ToolExecutor
from .budget import ToolBudget
from .scoring import register_scoring_tools
from .medical_retrieval import register_medical_retrieval_tools
from .citation import register_citation_tools


def register_all_tools(registry: ToolRegistry) -> None:
    """注册所有工具到 registry（医学检索 + 引用校验 + 评分）"""
    register_medical_retrieval_tools(registry)
    register_citation_tools(registry)
    register_scoring_tools(registry)


__all__ = [
    "BaseTool",
    "ToolContext",
    "ToolRegistry",
    "ToolExecutor",
    "ToolBudget",
    "register_all_tools",
    "register_scoring_tools",
]


def register_all_tools(registry: ToolRegistry) -> None:
    """注册所有可用工具"""
    from .medical_retrieval import register_medical_retrieval_tools
    from .citation import register_citation_tools
    
    register_medical_retrieval_tools(registry)
    register_citation_tools(registry)
    register_scoring_tools(registry)
