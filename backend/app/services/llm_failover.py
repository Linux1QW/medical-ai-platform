"""LLM Provider 故障切换管理器 — 跨 Provider 熔断与自动切换"""

import logging
import threading
from typing import Optional

from app.core.config import settings

logger = logging.getLogger(__name__)


class LLMFailoverManager:
    """LLM Provider 故障切换（熔断器模式）

    支持配置多个 LLM Provider（api_key / base_url / model），
    当主 Provider 连续失败达到阈值时自动切换到备用 Provider。
    """

    def __init__(self):
        self._providers: list[dict] = settings.get_llm_providers()
        self._current_index: int = 0
        self._failure_counts: dict[int, int] = {i: 0 for i in range(len(self._providers))}
        self._circuit_breaker_threshold: int = settings.LLM_CIRCUIT_BREAKER_THRESHOLD
        self._lock = threading.RLock()  # 可重入锁，防止 switch_to_next 内部调用 get_current_provider 死锁

    @property
    def provider_count(self) -> int:
        return len(self._providers)

    def get_current_provider(self) -> dict:
        """获取当前 Provider 配置

        Returns:
            dict: {"name": ..., "api_key": ..., "base_url": ..., "model": ...}
        """
        with self._lock:
            provider = self._providers[self._current_index]
            return {
                "name": provider.get("name", f"provider-{self._current_index}"),
                "api_key": provider["api_key"],
                "base_url": provider.get("base_url", settings.QWEN_API_BASE_URL),
                "model": provider.get("model", settings.QWEN_MODEL),
            }

    def report_failure(self) -> None:
        """报告一次失败，累加当前 Provider 的失败计数"""
        with self._lock:
            idx = self._current_index
            self._failure_counts[idx] = self._failure_counts.get(idx, 0) + 1
            count = self._failure_counts[idx]
            logger.warning(
                f"Provider '{self._providers[idx].get('name', idx)}' 失败计数: {count}"
            )

    def report_success(self) -> None:
        """报告一次成功，重置当前 Provider 的失败计数"""
        with self._lock:
            idx = self._current_index
            if self._failure_counts.get(idx, 0) > 0:
                logger.info(
                    f"Provider '{self._providers[idx].get('name', idx)}' 恢复正常，重置失败计数"
                )
            self._failure_counts[idx] = 0

    def should_switch(self) -> bool:
        """是否应该切换到备用 Provider"""
        with self._lock:
            idx = self._current_index
            return self._failure_counts.get(idx, 0) >= self._circuit_breaker_threshold

    def switch_to_next(self) -> dict:
        """切换到下一个可用 Provider

        Returns:
            新 Provider 的配置 dict
        """
        with self._lock:
            old_idx = self._current_index
            old_name = self._providers[old_idx].get("name", old_idx)

            # 尝试切换到下一个未熔断的 Provider
            tried = 0
            while tried < len(self._providers):
                self._current_index = (self._current_index + 1) % len(self._providers)
                if self._failure_counts.get(self._current_index, 0) < self._circuit_breaker_threshold:
                    new_name = self._providers[self._current_index].get("name", self._current_index)
                    logger.warning(
                        f"Provider 切换: {old_name} → {new_name} "
                        f"(连续失败 {self._failure_counts[old_idx]} 次)"
                    )
                    # 重置旧 Provider 的失败计数，允许后续恢复
                    self._failure_counts[old_idx] = 0
                    return self.get_current_provider()
                tried += 1

            # 所有 Provider 都已熔断，重置并回到第一个
            logger.error("所有 Provider 均已熔断，重置失败计数")
            self._current_index = 0
            self._failure_counts = {i: 0 for i in range(len(self._providers))}
            return self.get_current_provider()

    def get_status(self) -> dict:
        """获取当前 failover 状态（供监控接口使用）"""
        with self._lock:
            return {
                "current_index": self._current_index,
                "current_provider": self._providers[self._current_index].get(
                    "name", self._current_index
                ),
                "failure_counts": dict(self._failure_counts),
                "threshold": self._circuit_breaker_threshold,
                "total_providers": len(self._providers),
            }


# 全局单例
failover_manager = LLMFailoverManager()
