# -*- coding: utf-8 -*-
"""问诊覆盖报告 — 披露账本汇总为客观统计，回馈评估侧

把"医生问出了多少/漏问了什么"从 LLM 主观判断变为系统客观数据，
注入病史采集评估 Agent 的 patient_info 提升评分准确性。
"""
from .memory import MemoryState


def build_coverage_report(memory: MemoryState) -> dict:
    """账本 -> 覆盖统计（纯计算，无 LLM）"""
    total = len(memory.facts)
    disclosed = memory.facts_by_status("disclosed")
    stage_path: list[str] = []
    for s in memory.stage_history:
        if not stage_path or stage_path[-1] != s:
            stage_path.append(s)
    return {
        "total_facts": total,
        "disclosed_count": len(disclosed),
        "disclosure_rate": round(len(disclosed) / total, 4) if total else 0.0,
        "undisclosed_facts": [f.content for f in memory.facts_by_status("undisclosed")],
        "stage_path": stage_path,
        "final_trust": memory.trust,
        "final_emotion": memory.emotion,
    }


def format_coverage_text(report: dict) -> str:
    """覆盖报告 -> 注入评估 prompt 的中文文本块"""
    undisclosed = report["undisclosed_facts"]
    lines = [
        f"事实披露率：{report['disclosure_rate'] * 100:.1f}%（{report['disclosed_count']}/{report['total_facts']}）",
        f"问诊阶段路径：{' -> '.join(report['stage_path']) or '（无）'}",
        f"结束时患者信任度：{report['final_trust']}，情绪：{report['final_emotion']}",
        "医生未问出的档案事实：" + ("、".join(undisclosed) if undisclosed else "（无，全部问出）"),
    ]
    return "\n".join(lines)
