"""告警管理器 — 支持钉钉/企微 Webhook 通知

功能：
1. LLM 错误率超阈值告警
2. 工具预算告警（WARNING / CRITICAL / EXHAUSTED）
3. 预算成本告警
"""
import logging
from typing import Optional

import httpx

from app.core.config import settings

logger = logging.getLogger(__name__)


class AlertManager:
    """告警管理器"""

    _instance: Optional["AlertManager"] = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._http_client = httpx.AsyncClient(timeout=10.0)
        return cls._instance

    def __init__(self):
        # 防止重复初始化
        if not hasattr(self, "_initialized"):
            self._initialized = True
            self._consecutive_llm_errors: int = 0
            self._llm_error_alert_sent: bool = False

    async def send_alert(
        self,
        level: str,
        title: str,
        message: str,
        metadata: dict | None = None,
    ) -> bool:
        """发送告警通知（钉钉/企微 Webhook）

        Args:
            level: 告警级别 info / warning / critical
            title: 告警标题
            message: 告警详情
            metadata: 附加元数据

        Returns:
            True 表示发送成功，False 表示失败或未配置
        """
        webhook_url = settings.ALERT_WEBHOOK_URL
        if not webhook_url:
            logger.debug(f"[AlertManager] Webhook 未配置，跳过告警: {title}")
            return False

        webhook_type = settings.ALERT_WEBHOOK_TYPE
        payload = self._build_payload(webhook_type, level, title, message, metadata)

        try:
            resp = await self._http_client.post(webhook_url, json=payload)
            if resp.status_code == 200:
                logger.info(f"[AlertManager] 告警发送成功: [{level}] {title}")
                return True
            else:
                logger.warning(
                    f"[AlertManager] 告警发送失败: status={resp.status_code}, body={resp.text[:200]}"
                )
                return False
        except Exception as e:
            logger.error(f"[AlertManager] 告警发送异常: {e}")
            return False

    def _build_payload(
        self,
        webhook_type: str,
        level: str,
        title: str,
        message: str,
        metadata: dict | None,
    ) -> dict:
        """根据 Webhook 类型构建请求体"""
        meta_str = ""
        if metadata:
            meta_parts = [f"- {k}: {v}" for k, v in metadata.items()]
            meta_str = "\n" + "\n".join(meta_parts)

        full_text = f"**[{level.upper()}]** {title}\n\n{message}{meta_str}"

        if webhook_type == "wecom":
            # 企业微信 Webhook 格式
            return {
                "msgtype": "markdown",
                "markdown": {"content": full_text},
            }
        else:
            # 钉钉 Webhook 格式（默认）
            return {
                "msgtype": "markdown",
                "markdown": {
                    "title": f"[{level.upper()}] {title}",
                    "text": full_text,
                },
            }

    # ── LLM 错误率告警 ──────────────────────────────────────────────────────

    async def record_llm_error(self) -> None:
        """记录一次 LLM 错误，连续 3 次或错误率超阈值时触发告警"""
        self._consecutive_llm_errors += 1

        if self._consecutive_llm_errors >= 3 and not self._llm_error_alert_sent:
            self._llm_error_alert_sent = True
            await self.send_alert(
                level="critical",
                title="LLM 连续调用失败",
                message=f"LLM 已连续失败 {self._consecutive_llm_errors} 次，请检查 API 服务状态。",
                metadata={"consecutive_errors": self._consecutive_llm_errors},
            )

    def record_llm_success(self) -> None:
        """LLM 调用成功，重置连续错误计数"""
        self._consecutive_llm_errors = 0
        self._llm_error_alert_sent = False

    async def check_llm_error_rate(self, total_calls: int, total_failures: int) -> bool:
        """检查 LLM 错误率是否超阈值

        Returns:
            True 表示错误率超阈值（已发送告警）
        """
        if total_calls < 10:
            # 样本量不足，不触发告警
            return False

        error_rate = total_failures / total_calls
        if error_rate >= settings.LLM_ERROR_RATE_THRESHOLD:
            await self.send_alert(
                level="warning",
                title="LLM 错误率超阈值",
                message=(
                    f"当前 LLM 错误率为 {error_rate:.1%}，"
                    f"超过阈值 {settings.LLM_ERROR_RATE_THRESHOLD:.0%}。"
                ),
                metadata={
                    "total_calls": total_calls,
                    "total_failures": total_failures,
                    "error_rate": f"{error_rate:.1%}",
                },
            )
            return True
        return False

    # ── 预算告警 ──────────────────────────────────────────────────────────────

    async def check_budget_alert(
        self,
        alert_level: str,
        session_id: str,
        tool_name: str,
        total_calls: int,
        total_cost: float,
    ) -> None:
        """工具预算告警

        Args:
            alert_level: warning / critical / exhausted
            session_id: 会话 ID
            tool_name: 触发告警的工具名
            total_calls: 当前调用次数
            total_cost: 当前成本
        """
        level_map = {
            "warning": "warning",
            "critical": "critical",
            "exhausted": "critical",
        }
        level = level_map.get(alert_level, "warning")

        title_map = {
            "warning": "工具预算预警（70%）",
            "critical": "工具预算严重预警（90%）",
            "exhausted": "工具预算已耗尽",
        }
        title = title_map.get(alert_level, "工具预算告警")

        await self.send_alert(
            level=level,
            title=title,
            message=f"会话 {session_id} 工具 {tool_name} 触发预算告警。",
            metadata={
                "session_id": session_id,
                "tool_name": tool_name,
                "total_calls": total_calls,
                "total_cost": f"{total_cost:.4f}",
            },
        )


# 全局单例
alert_manager = AlertManager()
