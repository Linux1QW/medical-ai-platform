# -*- coding: utf-8 -*-
"""install_git_hooks 安装器单元测试（用临时假仓库，不触碰真实 .git）。"""
import importlib.util
from pathlib import Path

_MOD_PATH = Path(__file__).resolve().parents[2] / "scripts" / "install_git_hooks.py"
_spec = importlib.util.spec_from_file_location("install_git_hooks", _MOD_PATH)
igh = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(igh)


def test_find_git_dir_plain_repo(tmp_path):
    (tmp_path / ".git").mkdir()
    sub = tmp_path / "backend" / "scripts"
    sub.mkdir(parents=True)
    assert igh.find_git_dir(sub) == tmp_path / ".git"


def test_find_git_dir_worktree_file(tmp_path):
    real = tmp_path / "realgit"
    real.mkdir()
    (tmp_path / ".git").write_text(f"gitdir: {real}", encoding="utf-8")
    assert igh.find_git_dir(tmp_path) == real


def test_install_copies_hook_and_sets_marker(tmp_path):
    git_dir = tmp_path / ".git"
    git_dir.mkdir()
    ok, _ = igh.install(git_dir)
    assert ok
    dst = git_dir / "hooks" / igh.HOOK_NAME
    assert dst.exists()
    assert igh.MARKER in dst.read_text(encoding="utf-8")
    assert igh.status(git_dir) == "已安装(托管)"


def test_uninstall_removes_managed_hook(tmp_path):
    git_dir = tmp_path / ".git"
    git_dir.mkdir()
    igh.install(git_dir)
    ok, _ = igh.uninstall(git_dir)
    assert ok
    assert not (git_dir / "hooks" / igh.HOOK_NAME).exists()


def test_install_refuses_to_clobber_foreign_hook(tmp_path):
    git_dir = tmp_path / ".git"
    (git_dir / "hooks").mkdir(parents=True)
    foreign = git_dir / "hooks" / igh.HOOK_NAME
    foreign.write_text("#!/bin/sh\necho 用户自定义钩子\n", encoding="utf-8")
    ok, _ = igh.install(git_dir)  # 无 force -> 拒绝
    assert not ok
    assert "用户自定义钩子" in foreign.read_text(encoding="utf-8")  # 原内容保留
    # --force 才覆盖
    ok2, _ = igh.install(git_dir, force=True)
    assert ok2
    assert igh.MARKER in foreign.read_text(encoding="utf-8")


def test_uninstall_refuses_foreign_hook(tmp_path):
    git_dir = tmp_path / ".git"
    (git_dir / "hooks").mkdir(parents=True)
    foreign = git_dir / "hooks" / igh.HOOK_NAME
    foreign.write_text("#!/bin/sh\necho keep me\n", encoding="utf-8")
    ok, _ = igh.uninstall(git_dir)
    assert not ok
    assert foreign.exists()
