from .base import BaseTool, ToolContext
from .registry import ToolRegistry
from .executor import ToolExecutor
from .budget import ToolBudget
from .robust_tool_executor import (
    RobustToolExecutor,
    CircuitBreaker,
    CircuitState,
    RetryPolicy,
    ToolCallStats,
    ResultValidator,
)
from .tool_budget_manager import (
    ToolBudgetManager,
    BudgetConfig,
    BudgetAlertLevel,
    ToolCostConfig,
    SessionBudget,
)
from .tool_health_checker import (
    ToolHealthChecker,
    ToolHealthStatus,
    HealthCheckConfig,
    DegradedResultBuilder,
)
from .scoring import register_scoring_tools
from .medical_retrieval import register_medical_retrieval_tools
from .citation import register_citation_tools


def register_all_tools(registry: ToolRegistry) -> None:
    """注册所有可用工具（幂等操作）
    
    注意：评分工具已移至 evaluation_service 中按需注册，不再包含在此函数中。
    """
    register_medical_retrieval_tools(registry)
    register_citation_tools(registry)


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
]
