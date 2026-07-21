from .base import BaseTool, ToolContext
from .budget import ToolBudget
from .citation import register_citation_tools
from .consistency import register_consistency_tools
from .executor import ToolExecutor
from .medical_retrieval import register_medical_retrieval_tools
from .registry import ToolRegistry
from .robust_tool_executor import (
    CircuitBreaker,
    CircuitState,
    ResultValidator,
    RetryPolicy,
    RobustToolExecutor,
    ToolCallStats,
)
from .scoring import register_scoring_tools
from .tool_budget_manager import (
    BudgetAlertLevel,
    BudgetConfig,
    SessionBudget,
    ToolBudgetManager,
    ToolCostConfig,
)
from .tool_health_checker import (
    DegradedResultBuilder,
    HealthCheckConfig,
    ToolHealthChecker,
    ToolHealthStatus,
)


def register_all_tools(registry: ToolRegistry) -> None:
    """注册所有可用工具（幂等操作）

    注意：评分工具已移至 evaluation_service 中按需注册，不再包含在此函数中。
    """
    register_medical_retrieval_tools(registry)
    register_citation_tools(registry)
    register_consistency_tools(registry)


__all__ = [
    # 基础组件
    "BaseTool",
    "ToolContext",
    "ToolRegistry",
    "ToolExecutor",
    "ToolBudget",
    # 健壮执行器
    "RobustToolExecutor",
    "CircuitBreaker",
    "CircuitState",
    "RetryPolicy",
    "ToolCallStats",
    "ResultValidator",
    # 预算管理
    "ToolBudgetManager",
    "BudgetConfig",
    "BudgetAlertLevel",
    "ToolCostConfig",
    "SessionBudget",
    # 健康检查
    "ToolHealthChecker",
    "ToolHealthStatus",
    "HealthCheckConfig",
    "DegradedResultBuilder",
    # 注册函数
    "register_all_tools",
    "register_scoring_tools",
    "register_consistency_tools",
]
