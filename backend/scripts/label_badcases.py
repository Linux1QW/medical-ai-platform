# -*- coding: utf-8 -*-
"""Badcase 归因标注 CLI：为回放 badcase JSONL 填充 attribution 标签，产出去标识失败模式清单。

用法（在 backend 目录下）：
    .\\venv\\Scripts\\python.exe scripts\\label_badcases.py                       # 取最新 badcase_*.jsonl
    .\\venv\\Scripts\\python.exe scripts\\label_badcases.py --badcase <path>
    .\\venv\\Scripts\\python.exe scripts\\label_badcases.py --no-inplace          # 只出清单，不回写

产物：
  1) 原地回写带 attribution 的 badcase JSONL（含对话，位于 reports/ 已 gitignore）
  2) 去标识失败模式清单 JSON（无对话文本，可外发/入档）
纯离线，无 LLM/网络调用。退出码：0=正常；2=未找到 badcase 文件。
"""
import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))
from evaluation.badcase_attribution import (  # noqa: E402
    DIMENSION_MODE,
    label_badcases,
    summarize_badcases,
)

BAD_DIR = Path(__file__).parent.parent / "evaluation" / "reports" / "patient_ab"


def _latest_badcase() -> Path | None:
    if not BAD_DIR.exists():
        return None
    files = sorted(BAD_DIR.glob("badcase_20*.jsonl"))  # 时间戳前缀，排除 summary
    return files[-1] if files else None


def _read_jsonl(path: Path) -> list[dict]:
    return [json.loads(ln) for ln in path.read_text(encoding="utf-8").splitlines() if ln.strip()]


def main() -> int:
    try:
        sys.stdout.reconfigure(encoding="utf-8")  # Windows 控制台中文输出
    except Exception:
        pass
    ap = argparse.ArgumentParser(description="Badcase 失败模式归因标注")
    ap.add_argument("--badcase", default="", help="badcase JSONL 路径；缺省取最新")
    ap.add_argument("--summary-out", default="", help="去标识清单输出路径；缺省与 badcase 同目录")
    ap.add_argument("--no-inplace", action="store_true", help="不回写原 badcase 文件")
    args = ap.parse_args()

    bad_path = Path(args.badcase) if args.badcase else _latest_badcase()
    if bad_path is None or not bad_path.exists():
        print("未找到 badcase JSONL，请先运行 ab_patient_replay.py --judge")
        return 2

    records = _read_jsonl(bad_path)
    labeled = label_badcases(records)
    if not args.no_inplace:
        with open(bad_path, "w", encoding="utf-8") as f:
            for r in labeled:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")

    summary = summarize_badcases(records)
    if args.summary_out:
        sum_path = Path(args.summary_out)
    else:
        sum_path = bad_path.with_name(bad_path.stem.replace("badcase_", "badcase_summary_") + ".json")
    sum_path.parent.mkdir(parents=True, exist_ok=True)
    sum_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"badcase 文件: {bad_path.name}（{summary['total']} 条，阈值 dim≤{summary['threshold']}）")
    print("\n按失败模式（主因）:")
    for mode, n in summary["by_mode"].items():
        label = DIMENSION_MODE.get(mode, mode)
        print(f"  {n:>2}  {mode:<22} {label}")
    print("\n按臂:")
    for arm, md in summary["by_arm"].items():
        parts = ", ".join(f"{k}:{v}" for k, v in sorted(md.items(), key=lambda kv: -kv[1]))
        print(f"  {arm:<14} {parts}")
    print(f"\n去标识清单已写入: {sum_path.name}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
