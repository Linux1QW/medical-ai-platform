"""模型路由器 — 根据场景选择模型，支持降级"""

import logging
from typing import Optional

from app.core.config import settings

logger = logging.getLogger(__name__)


class ModelRouter:
    """根据场景选择模型，支持自动降级"""

    # 场景 → 模型名映射
    ROUTES: dict[str, str] = {
        "critical": settings.MODEL_CRITICAL,      # 评估智能体（核心）
        "standard": settings.MODEL_STANDARD,      # 人文关怀评估
        "lightweight": settings.MODEL_LIGHTWEIGHT, # 输入清洗/格式化
    }

    # 降级链：主模型 → 备用模型
    FALLBACK_CHAIN: dict[str, str] = {
        settings.MODEL_CRITICAL: settings.MODEL_STANDARD,
        settings.MODEL_STANDARD: settings.MODEL_LIGHTWEIGHT,
        settings.MODEL_LIGHTWEIGHT: settings.MODEL_CRITICAL,  # turbo 降级到 max（兜底）
    }

    def get_model(self, scenario: str) -> str:
        """根据场景返回模型名

        Args:
            scenario: 场景标识 ("critical" | "standard" | "lightweight")

        Returns:
            模型名称字符串
        """
        model = self.ROUTES.get(scenario)
        if model:
            return model
        logger.warning(f"未知场景 '{scenario}'，使用默认 critical 模型")
        return self.ROUTES["critical"]

    def get_fallback_model(self, model: str) -> Optional[str]:
        """返回指定模型的降级备用模型

        Args:
            model: 当前使用的模型名

        Returns:
            备用模型名，若无可用备用则返回 None
        """
        return self.FALLBACK_CHAIN.get(model)

    def degrade(self, model: str) -> str:
        """执行一次降级，返回降级后的模型

        如果已降级到最低级别，则循环回最高级别。
        """
        fallback = self.get_fallback_model(model)
        if fallback:
            logger.warning(f"模型降级: {model} → {fallback}")
            return fallback
        return model


# 全局单例
model_router = ModelRouter()
