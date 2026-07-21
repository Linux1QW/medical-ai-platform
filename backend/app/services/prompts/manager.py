# -*- coding: utf-8 -*-
"""Prompt 外置化与版本管理

设计目标（零新增依赖）：
- **外置**：所有 Prompt 文本移出 Python 源码，存放于 ``backend/app/prompts/<key>/<version>.txt``（原始多行文本，无需转义，Git 友好、可被非开发人员编辑）。
- **版本管理**：``manifest.json`` 声明每个 Prompt 的可用版本与当前活跃版本；可通过环境变量
  ``PROMPT_ACTIVE_VERSIONS``（JSON）在不改文件的情况下切换活跃版本，便于灰度 / A-B 对比。
- **渲染**：``render`` 用 ``str.format_map`` 做变量替换，缺失占位符原样保留（不抛错）。
- **健壮**：加载失败 / 文件缺失时可通过 ``default`` 降级；带内存缓存与线程锁。

目录结构::

    backend/app/prompts/
        manifest.json                 # {"diagnosis.system": {"active": "v1", "description": "..."}, ...}
        diagnosis.system/
            v1.txt
            v2.txt                    # 新版本仅需新增文件并在 manifest 中登记
"""

import json
import logging
import threading
from pathlib import Path
from typing import Dict, List, Optional

logger = logging.getLogger(__name__)

# backend/app/prompts —— 与 backend/app/services 平级
PROMPTS_DIR = Path(__file__).resolve().parents[2] / "prompts"
MANIFEST_FILE = PROMPTS_DIR / "manifest.json"


class _SafeDict(dict):
    """str.format_map 用：缺失键原样保留为 ``{key}``，避免 KeyError。"""

    def __missing__(self, key):  # noqa: D401
        return "{" + key + "}"


class PromptManager:
    """Prompt 加载器（单例）—— 负责外置文本的加载、版本选择、渲染与缓存。"""

    def __init__(self, prompts_dir: Optional[Path] = None):
        self._dir = Path(prompts_dir) if prompts_dir else PROMPTS_DIR
        self._manifest: Dict[str, dict] = {}
        self._cache: Dict[str, str] = {}          # f"{key}@{version}" -> content
        self._overrides: Dict[str, str] = {}       # key -> version（环境变量覆盖）
        self._lock = threading.RLock()
        self._loaded = False

    # ── 加载 ──────────────────────────────────────────────────────────────
    def _ensure_loaded(self) -> None:
        if self._loaded:
            return
        with self._lock:
            if self._loaded:
                return
            self._load_manifest()
            self._load_overrides()
            self._loaded = True

    def _load_manifest(self) -> None:
        try:
            raw = (self._dir / "manifest.json").read_text(encoding="utf-8")
            data = json.loads(raw)
            if isinstance(data, dict):
                self._manifest = data
            else:
                logger.warning("Prompt manifest 格式异常（非对象），已忽略")
        except FileNotFoundError:
            logger.warning(f"Prompt manifest 不存在: {self._dir / 'manifest.json'}")
        except (json.JSONDecodeError, OSError) as e:
            logger.warning(f"Prompt manifest 加载失败: {e}")

    def _load_overrides(self) -> None:
        """从 settings.PROMPT_ACTIVE_VERSIONS 读取版本覆盖（JSON 映射 key->version）。"""
        try:
            from app.core.config import settings

            raw = getattr(settings, "PROMPT_ACTIVE_VERSIONS", "{}") or "{}"
            data = json.loads(raw)
            if isinstance(data, dict):
                self._overrides = {str(k): str(v) for k, v in data.items()}
        except Exception as e:  # 覆盖是可选能力，任何异常都不应影响主流程
            logger.debug(f"Prompt 版本覆盖解析失败，忽略: {e}")

    # ── 版本解析 ──────────────────────────────────────────────────────────
    def get_active_version(self, key: str) -> Optional[str]:
        """返回某 Prompt 当前生效的版本（环境覆盖 > manifest.active）。"""
        self._ensure_loaded()
        if key in self._overrides:
            return self._overrides[key]
        entry = self._manifest.get(key)
        if isinstance(entry, dict):
            return entry.get("active")
        return None

    def list_prompts(self) -> List[str]:
        """列出 manifest 中登记的所有 Prompt key。"""
        self._ensure_loaded()
        return sorted(self._manifest.keys())

    def list_versions(self, key: str) -> List[str]:
        """列出某 Prompt 目录下所有可用版本（按文件名 stem）。"""
        self._ensure_loaded()
        d = self._dir / key
        if not d.is_dir():
            return []
        return sorted(p.stem for p in d.glob("*.txt"))

    # ── 读取 ──────────────────────────────────────────────────────────────
    def get(
        self,
        key: str,
        version: Optional[str] = None,
        *,
        default: Optional[str] = None,
    ) -> str:
        """获取 Prompt 文本。

        Args:
            key: Prompt 标识（对应 ``app/prompts/<key>/`` 目录）。
            version: 指定版本；None 时取活跃版本（环境覆盖 > manifest.active）。
            default: 文件缺失 / 加载失败时的降级文本；未提供则抛 ``KeyError``。

        Returns:
            Prompt 原始文本。
        """
        self._ensure_loaded()
        ver = version or self.get_active_version(key)
        if not ver:
            if default is not None:
                return default
            raise KeyError(
                f"Prompt '{key}' 未在 manifest 中登记且未指定版本"
            )

        cache_key = f"{key}@{ver}"
        cached = self._cache.get(cache_key)
        if cached is not None:
            return cached

        path = self._dir / key / f"{ver}.txt"
        try:
            content = path.read_text(encoding="utf-8")
        except (FileNotFoundError, OSError) as e:
            if default is not None:
                logger.warning(f"Prompt '{key}@{ver}' 读取失败，使用降级文本: {e}")
                return default
            raise KeyError(f"Prompt 文件不存在: {path}") from e

        with self._lock:
            self._cache[cache_key] = content
        return content

    def render(
        self,
        key: str,
        version: Optional[str] = None,
        *,
        default: Optional[str] = None,
        **variables,
    ) -> str:
        """获取并渲染 Prompt（``{var}`` 占位符替换；缺失占位符原样保留）。"""
        template = self.get(key, version, default=default)
        if not variables:
            return template
        try:
            return template.format_map(_SafeDict(variables))
        except (ValueError, IndexError) as e:
            # 例如模板中存在未转义的 '{' 花括号；渲染失败时退回原文
            logger.warning(f"Prompt '{key}' 渲染失败，返回原始模板: {e}")
            return template

    def reload(self) -> None:
        """清空缓存并重新加载 manifest / 覆盖配置（用于热更新）。"""
        with self._lock:
            self._cache.clear()
            self._manifest.clear()
            self._overrides.clear()
            self._loaded = False
        self._ensure_loaded()


# ── 全局单例与便捷函数 ────────────────────────────────────────────────────────
_manager: Optional[PromptManager] = None
_manager_lock = threading.Lock()


def get_prompt_manager() -> PromptManager:
    """获取全局 PromptManager 单例（懒初始化）。"""
    global _manager
    if _manager is None:
        with _manager_lock:
            if _manager is None:
                _manager = PromptManager()
    return _manager


def get_prompt(key: str, version: Optional[str] = None, *, default: Optional[str] = None) -> str:
    """便捷函数：获取 Prompt 文本。"""
    return get_prompt_manager().get(key, version, default=default)


def render_prompt(
    key: str,
    version: Optional[str] = None,
    *,
    default: Optional[str] = None,
    **variables,
) -> str:
    """便捷函数：获取并渲染 Prompt。"""
    return get_prompt_manager().render(key, version, default=default, **variables)
