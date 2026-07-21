# -*- coding: utf-8 -*-
"""Prompt 外置化与版本管理模块。"""

from app.services.prompts.manager import (
    PromptManager,
    get_prompt,
    get_prompt_manager,
    render_prompt,
)

__all__ = [
    "PromptManager",
    "get_prompt",
    "get_prompt_manager",
    "render_prompt",
]
