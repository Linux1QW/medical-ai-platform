"""输入验证工具函数"""
import html
import re


def strip_html_tags(text: str) -> str:
    """移除 HTML 标签，防止 XSS 注入"""
    if not text:
        return text
    # 移除所有 HTML 标签
    cleaned = re.sub(r"<[^>]+>", "", text)
    # 反转义 HTML 实体
    cleaned = html.unescape(cleaned)
    return cleaned


def sanitize_text(text: str) -> str:
    """清理文本输入：移除 HTML 标签、首尾空白"""
    if not text:
        return text
    cleaned = strip_html_tags(text)
    return cleaned.strip()


def contains_html(text: str) -> bool:
    """检测文本是否包含 HTML 标签"""
    if not text:
        return False
    return bool(re.search(r"<[^>]+>", text))
