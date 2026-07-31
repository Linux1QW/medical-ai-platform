# -*- coding: utf-8 -*-
"""构建患者模拟评测集：从 dataset 分层抽样，固化为版本化 JSONL。

用法（在 backend 目录下）：
    .\\venv\\Scripts\\python.exe scripts\\build_eval_set.py --n 18 --seed 42
    .\\venv\\Scripts\\python.exe scripts\\build_eval_set.py --n 18 --out evaluation/patient_cases/patient_sim_v1.jsonl

幂等：同 (dataset, n, seed) 重跑产出完全一致。
"""
import argparse
import json
import sys
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from evaluation.patient_eval_set import scan_dataset, stratified_sample  # noqa: E402

DATASET_DIR = Path(__file__).parent.parent.parent / "dataset"
DEFAULT_OUT = Path(__file__).parent.parent / "evaluation" / "patient_cases" / "patient_sim_v1.jsonl"


def main():
    sys.stdout.reconfigure(encoding="utf-8")
    parser = argparse.ArgumentParser(description="构建患者模拟评测集（分层抽样）")
    parser.add_argument("--n", type=int, default=18, help="抽样目标例数")
    parser.add_argument("--seed", type=int, default=42, help="确定性抽样种子")
    parser.add_argument("--version", default="v1", help="评测集版本号")
    parser.add_argument("--out", default=str(DEFAULT_OUT), help="输出 JSONL 路径")
    args = parser.parse_args()

    records = scan_dataset(DATASET_DIR)
    print(f"扫描 dataset：{len(records)} 例有效（含门诊对话）")
    selected = stratified_sample(records, n=args.n, seed=args.seed)

    # 人格/诊断覆盖统计
    from collections import Counter
    pc = Counter(r["personality"] for r in selected)
    dc = Counter(r["diagnosis"] for r in selected)
    print(f"抽样 {len(selected)} 例 | 人格覆盖 {dict(pc)} | 诊断种类 {len(dc)}")

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    meta = {
        "_meta": {
            "version": args.version,
            "seed": args.seed,
            "n": len(selected),
            "generated_at": datetime.now().isoformat(timespec="seconds"),
            "source": "dataset",
        }
    }
    with open(out_path, "w", encoding="utf-8") as f:
        f.write(json.dumps(meta, ensure_ascii=False) + "\n")
        for r in selected:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"评测集已写入: {out_path}")


if __name__ == "__main__":
    main()
