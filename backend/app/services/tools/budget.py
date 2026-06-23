class ToolBudget:
    """管理工具调用预算"""

    def __init__(self, limits: dict[str, int]):
        self._limits = dict(limits)  # {"search_medical_kb": 3, ...}
        self._used: dict[str, int] = {k: 0 for k in limits}

    def check(self, tool_name: str) -> bool:
        """检查是否还有预算，未设限的工具始终返回 True"""
        if tool_name not in self._limits:
            return True  # 未设限 = 无限制
        return self._used.get(tool_name, 0) < self._limits[tool_name]

    def consume(self, tool_name: str) -> None:
        """消耗一次预算"""
        if tool_name in self._used:
            self._used[tool_name] += 1
        else:
            self._used[tool_name] = 1

    def remaining(self, tool_name: str) -> int:
        """返回剩余预算，未设限返回 -1"""
        if tool_name not in self._limits:
            return -1
        return max(0, self._limits[tool_name] - self._used.get(tool_name, 0))

    def summary(self) -> dict[str, dict]:
        """返回预算使用摘要"""
        result = {}
        for name in self._limits:
            result[name] = {
                "limit": self._limits[name],
                "used": self._used.get(name, 0),
                "remaining": max(0, self._limits[name] - self._used.get(name, 0)),
            }
        return result
