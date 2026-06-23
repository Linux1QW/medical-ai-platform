from .base import BaseTool


class ToolRegistry:
    """单例工具注册表"""
    _instance = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._tools = {}
        return cls._instance

    def register(self, tool: BaseTool) -> None:
        """注册工具，重复注册同名工具时跳过（幂等操作）"""
        if tool.name in self._tools:
            return
        self._tools[tool.name] = tool

    def get(self, name: str) -> BaseTool | None:
        """根据名称获取工具，未注册返回 None"""
        return self._tools.get(name)

    def list_tools(self) -> list[str]:
        """返回所有已注册工具名"""
        return list(self._tools.keys())

    def get_openai_schemas(self, tool_names: list[str] | None = None) -> list[dict]:
        """获取 OpenAI tool schema 列表，tool_names 为 None 时返回全部"""
        names = tool_names or self.list_tools()
        schemas = []
        for name in names:
            tool = self._tools.get(name)
            if tool:
                schemas.append(tool.openai_schema())
        return schemas

    def reset(self) -> None:
        """清空注册表（测试用）"""
        self._tools.clear()
