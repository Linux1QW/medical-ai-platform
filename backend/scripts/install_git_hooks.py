# -*- coding: utf-8 -*-
"""安装/卸载患者评测回归 pre-push 护栏到本仓库 .git/hooks。

用法（在 backend 目录下）：
    .\\venv\\Scripts\\python.exe scripts\\install_git_hooks.py             # 安装
    .\\venv\\Scripts\\python.exe scripts\\install_git_hooks.py --check     # 查看状态
    .\\venv\\Scripts\\python.exe scripts\\install_git_hooks.py --uninstall # 卸载
    .\\venv\\Scripts\\python.exe scripts\\install_git_hooks.py --force     # 覆盖非托管的已有钩子

设计：源钩子 scripts/hooks/pre-push 随仓库版本化，安装即拷贝到 .git/hooks/pre-push
并置可执行位。仅当目标带托管标记（MARKER）时才允许覆盖/卸载，避免误删用户自定义
钩子（--force 例外）。退出码：0=成功；1=拒绝操作；2=环境不满足。
"""
import argparse
import shutil
import stat
import sys
from pathlib import Path

MARKER = "QODER-MANAGED-HOOK: patient-eval-regression"
HOOK_NAME = "pre-push"
SRC_HOOK = Path(__file__).parent / "hooks" / HOOK_NAME


def find_git_dir(start: Path) -> Path | None:
    """从 start 向上找 .git；兼容普通仓库（.git 目录）与 worktree（.git 文件）。"""
    for d in [start, *start.parents]:
        g = d / ".git"
        if g.is_dir():
            return g
        if g.is_file():  # worktree: .git 是指向真实 gitdir 的文本文件
            txt = g.read_text(encoding="utf-8").strip()
            if txt.startswith("gitdir:"):
                return Path(txt.split(":", 1)[1].strip())
    return None


def _is_managed(path: Path) -> bool:
    return path.exists() and MARKER in path.read_text(encoding="utf-8", errors="ignore")


def install(git_dir: Path, force: bool = False) -> tuple[bool, str]:
    hooks_dir = git_dir / "hooks"
    hooks_dir.mkdir(parents=True, exist_ok=True)
    dst = hooks_dir / HOOK_NAME
    if dst.exists() and not _is_managed(dst) and not force:
        return False, f"已存在非托管的 {HOOK_NAME} 钩子，未覆盖（加 --force 可强制覆盖）"
    shutil.copyfile(SRC_HOOK, dst)
    dst.chmod(dst.stat().st_mode | stat.S_IEXEC | stat.S_IXGRP | stat.S_IXOTH)
    return True, f"已安装 {HOOK_NAME} -> {dst}"


def uninstall(git_dir: Path) -> tuple[bool, str]:
    dst = git_dir / "hooks" / HOOK_NAME
    if not dst.exists():
        return True, "未安装，无需卸载"
    if not _is_managed(dst):
        return False, f"{HOOK_NAME} 非本工具托管，拒绝删除"
    dst.unlink()
    return True, f"已卸载 {HOOK_NAME}"


def status(git_dir: Path) -> str:
    dst = git_dir / "hooks" / HOOK_NAME
    if not dst.exists():
        return "未安装"
    return "已安装(托管)" if _is_managed(dst) else "存在但非本工具托管"


def main() -> int:
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass
    ap = argparse.ArgumentParser(description="安装患者评测回归 pre-push 护栏")
    ap.add_argument("--uninstall", action="store_true", help="卸载钩子")
    ap.add_argument("--check", action="store_true", help="仅查看安装状态")
    ap.add_argument("--force", action="store_true", help="覆盖已存在的非托管钩子")
    args = ap.parse_args()

    git_dir = find_git_dir(Path(__file__).resolve())
    if git_dir is None:
        print("未找到 .git，请在 git 仓库内运行")
        return 2
    if not SRC_HOOK.exists():
        print(f"源钩子缺失: {SRC_HOOK}")
        return 2

    if args.check:
        print(f"{HOOK_NAME}: {status(git_dir)}")
        return 0
    if args.uninstall:
        ok, msg = uninstall(git_dir)
    else:
        ok, msg = install(git_dir, force=args.force)
    print(msg)
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
