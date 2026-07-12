from abc import ABC, abstractmethod
from pydantic import BaseModel, ConfigDict
from typing import Any


class ToolContext(BaseModel):
    """工具执行上下文，由调用方构造"""
    run_id: str | None = None
    agent_name: str = ""
    budgets: dict[str, int] = {}  # {"search_medical_kb": 3, "expand_query": 2, ...}
    allowed_citation_ids: set[str] = set()  # 合法 citation_id 白名单
    evidence_cache: dict[str, Any] = {}  # 缓存已检索的 evidence
    extras: dict[str, Any] = {}  # 扩展字段（如 rag_trace）

    model_config = ConfigDict(arbitrary_types_allowed=True)


class BaseTool(ABC):
    """所有工具的基类"""
    name: str = ""
    description: str = ""
    args_schema: type[BaseModel] = None  # Pydantic 模型，用于参数校验
    result_schema: type[BaseModel] | None = None  # 可选的返回结构定义
    timeout_seconds: int = 30
    critical: bool = False  # 关键工具失败时标记 degraded

    @abstractmethod
    async def execute(self, args: BaseModel, context: ToolContext) -> dict:
        """执行工具逻辑，返回结果 dict"""
        ...

    def openai_schema(self) -> dict:
        """生成 OpenAI Function Calling 格式的 tool schema"""
        schema = self.args_schema.model_json_schema()
        # 提取 properties 和 required
        properties = schema.get("properties", {})
        required = schema.get("required", [])
        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                "parameters": {
                    "type": "object",
                    "properties": properties,
                    "required": required,
                },
            },
        }
