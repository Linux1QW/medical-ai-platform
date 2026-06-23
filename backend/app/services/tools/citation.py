# -*- coding: utf-8 -*-
"""引用校验工具 — 校验 LLM 输出中使用的引用 ID 合法性"""

import logging
from typing import List

from pydantic import BaseModel, Field

from app.services.tools.base import BaseTool, ToolContext
from app.services.tools.registry import ToolRegistry

logger = logging.getLogger(__name__)


# ── Args Schema ───────────────────────────────────────────────────────────────


class VerifyCitationArgs(BaseModel):
    used_citation_ids: List[str] = Field(
        description="LLM 声称使用的引用 ID 列表",
    )


# ── Tool: VerifyCitation ─────────────────────────────────────────────────────


class VerifyCitation(BaseTool):
    name = "verify_citation"
    description = "校验 LLM 输出中使用的引用 ID 是否在合法白名单中"
    args_schema = VerifyCitationArgs
    timeout_seconds = 10
    critical = False

    async def execute(
        self, args: VerifyCitationArgs, context: ToolContext
    ) -> dict:
        """检查每个 citation_id 是否在 context.allowed_citation_ids 白名单中"""
        allowed = context.allowed_citation_ids
        used = args.used_citation_ids

        invalid_ids = [cid for cid in used if cid not in allowed]
        is_valid = len(invalid_ids) == 0

        # 判断是否遗漏了必须引用的证据（白名单非空但 LLM 未使用任何引用）
        missing_required = bool(allowed) and len(used) == 0

        result = {
            "valid": is_valid,
            "invalid_citation_ids": invalid_ids,
            "missing_required_citations": missing_required,
            "verified_count": len(used) - len(invalid_ids),
        }

        if not is_valid:
            logger.warning(
                f"verify_citation 发现非法引用: {invalid_ids} "
                f"(白名单大小={len(allowed)})"
            )

        return result


# ── 注册函数 ─────────────────────────────────────────────────────────────────


def register_citation_tools(registry: ToolRegistry) -> None:
    """注册引用校验工具"""
    try:
        registry.register(VerifyCitation())
    except ValueError:
        pass  # 已注册则跳过
