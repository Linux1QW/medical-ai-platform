# -*- coding: utf-8 -*-
"""PromptManager 单元测试

覆盖：
- 加载 manifest 与按活跃版本读取文本
- 指定版本读取 / 版本切换（环境覆盖）
- render 变量渲染与缺失占位符原样保留
- 文件缺失 / 未登记时的降级（default）与异常（KeyError）
- 缓存命中与 reload 热更新
- 与仓库内真实 app/prompts 目录的联通性（13 个 key 均可加载）
"""

import json

import pytest

from app.services.prompts.manager import PromptManager, get_prompt, get_prompt_manager

# ── 测试用临时 prompts 目录 ─────────────────────────────────────────────────────


@pytest.fixture
def prompts_dir(tmp_path):
    """构造一个最小可用的外置 prompts 目录。"""
    manifest = {
        "demo.system": {"active": "v1", "description": "demo"},
        "demo.greeting": {"active": "v2", "description": "greeting"},
        "orphan.key": {"active": "v9", "description": "文件缺失用于测试降级"},
    }
    (tmp_path / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False), encoding="utf-8"
    )

    d1 = tmp_path / "demo.system"
    d1.mkdir()
    (d1 / "v1.txt").write_text("你是助手 v1", encoding="utf-8")
    (d1 / "v2.txt").write_text("你是助手 v2", encoding="utf-8")

    d2 = tmp_path / "demo.greeting"
    d2.mkdir()
    (d2 / "v1.txt").write_text("你好 {name}", encoding="utf-8")
    (d2 / "v2.txt").write_text("您好，{name}！欢迎 {place}", encoding="utf-8")

    return tmp_path


@pytest.fixture
def manager(prompts_dir):
    return PromptManager(prompts_dir=prompts_dir)


# ── 加载 / 活跃版本 ─────────────────────────────────────────────────────────────


def test_list_prompts(manager):
    assert manager.list_prompts() == ["demo.greeting", "demo.system", "orphan.key"]


def test_list_versions(manager):
    assert manager.list_versions("demo.system") == ["v1", "v2"]
    assert manager.list_versions("not.exist") == []


def test_get_active_version(manager):
    assert manager.get_active_version("demo.system") == "v1"
    assert manager.get_active_version("demo.greeting") == "v2"
    assert manager.get_active_version("not.registered") is None


def test_get_uses_active_version(manager):
    assert manager.get("demo.system") == "你是助手 v1"
    # greeting 活跃版本为 v2
    assert manager.get("demo.greeting") == "您好，{name}！欢迎 {place}"


def test_get_explicit_version(manager):
    assert manager.get("demo.system", "v2") == "你是助手 v2"


# ── 版本切换（环境覆盖） ────────────────────────────────────────────────────────


def test_version_override(prompts_dir, monkeypatch):
    from app.core import config

    monkeypatch.setattr(
        config.settings, "PROMPT_ACTIVE_VERSIONS", json.dumps({"demo.system": "v2"})
    )
    m = PromptManager(prompts_dir=prompts_dir)
    assert m.get_active_version("demo.system") == "v2"
    assert m.get("demo.system") == "你是助手 v2"
    # 未覆盖的 key 仍走 manifest
    assert m.get_active_version("demo.greeting") == "v2"


def test_override_invalid_json_ignored(prompts_dir, monkeypatch):
    from app.core import config

    monkeypatch.setattr(config.settings, "PROMPT_ACTIVE_VERSIONS", "{not-json")
    m = PromptManager(prompts_dir=prompts_dir)
    # 覆盖解析失败不影响主流程，回退 manifest.active
    assert m.get_active_version("demo.system") == "v1"


# ── 渲染 ────────────────────────────────────────────────────────────────────────


def test_render_with_variables(manager):
    assert manager.render("demo.greeting", "v1", name="张三") == "你好 张三"


def test_render_missing_placeholder_kept(manager):
    # 只提供 name，place 占位符原样保留
    out = manager.render("demo.greeting", "v2", name="李四")
    assert out == "您好，李四！欢迎 {place}"


def test_render_without_variables_returns_template(manager):
    assert manager.render("demo.greeting", "v1") == "你好 {name}"


# ── 缺失 / 降级 ─────────────────────────────────────────────────────────────────


def test_missing_file_with_default(manager):
    # orphan.key 活跃版本 v9 的文件不存在 → 返回 default
    assert manager.get("orphan.key", default="兜底") == "兜底"


def test_missing_file_raises_without_default(manager):
    with pytest.raises(KeyError):
        manager.get("orphan.key")


def test_unregistered_key_raises_without_default(manager):
    with pytest.raises(KeyError):
        manager.get("totally.unknown")


def test_unregistered_key_with_default(manager):
    assert manager.get("totally.unknown", default="X") == "X"


# ── 缓存 / 热更新 ───────────────────────────────────────────────────────────────


def test_cache_hit_ignores_file_change(manager, prompts_dir):
    assert manager.get("demo.system", "v1") == "你是助手 v1"
    (prompts_dir / "demo.system" / "v1.txt").write_text("改动后", encoding="utf-8")
    # 已缓存，读到旧值
    assert manager.get("demo.system", "v1") == "你是助手 v1"


def test_reload_picks_up_changes(manager, prompts_dir):
    assert manager.get("demo.system", "v1") == "你是助手 v1"
    (prompts_dir / "demo.system" / "v1.txt").write_text("改动后", encoding="utf-8")
    manager.reload()
    assert manager.get("demo.system", "v1") == "改动后"


# ── 真实仓库 prompts 目录联通性 ─────────────────────────────────────────────────


def test_real_repository_prompts_all_loadable():
    """确保迁移后仓库内登记的所有 prompt 均可实际读取到非空文本。"""
    m = get_prompt_manager()
    keys = m.list_prompts()
    assert len(keys) >= 13
    for key in keys:
        text = m.get(key)
        assert isinstance(text, str) and text.strip(), f"{key} 文本为空"


def test_get_prompt_convenience_matches_manager():
    m = get_prompt_manager()
    assert get_prompt("diagnosis.system") == m.get("diagnosis.system")
