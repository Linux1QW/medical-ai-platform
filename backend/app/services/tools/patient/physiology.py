# -*- coding: utf-8 -*-
"""生理指标计算器 — 确定性生命体征生成

同一会话内同一指标多次测量结果稳定（seed = consultation_id:vital:abnormal），
避免 LLM 自由发挥导致"上次 38.5 这次 36.2"的不一致。纯本地计算，零 LLM 成本。
"""
import random
from typing import Any

from pydantic import BaseModel, Field

from app.services.tools.base import BaseTool, ToolContext

# 正常/异常取值范围：(下限, 上限, 小数位数, 单位)；血压项为嵌套元组，故元素类型异构
_BASELINES: dict[str, tuple[Any, ...]] = {
    "body_temperature": (36.2, 37.0, 1, "℃"),
    "heart_rate": (62, 95, 0, "次/分"),
    "respiratory_rate": (14, 19, 0, "次/分"),
    "blood_pressure": ((105, 130), (65, 85), 0, "mmHg"),  # (收缩压范围, 舒张压范围)
}
_ABNORMAL: dict[str, tuple[Any, ...]] = {
    "body_temperature": (37.8, 39.5, 1, "℃"),
    "heart_rate": (102, 130, 0, "次/分"),
    "respiratory_rate": (22, 30, 0, "次/分"),
    "blood_pressure": ((145, 175), (92, 110), 0, "mmHg"),
}


class PhysiologyCalculatorArgs(BaseModel):
    vital: str = Field(description="指标名: body_temperature/heart_rate/respiratory_rate/blood_pressure")
    consultation_id: int = Field(default=0, description="会话 ID（确定性种子，缺省时从上下文注入）")
    abnormal: bool = Field(default=False, description="是否按异常（病情相关）范围生成")


class PhysiologyCalculator(BaseTool):
    name = "physiology_calculator"
    description = "按会话确定性生成生命体征数值（体温/心率/呼吸/血压）"
    args_schema = PhysiologyCalculatorArgs
    timeout_seconds = 5
    critical = False

    async def execute(self, args: PhysiologyCalculatorArgs, context: ToolContext) -> dict:
        table = _ABNORMAL if args.abnormal else _BASELINES
        if args.vital not in table:
            return {"error": f"未知指标: {args.vital}", "vital": args.vital}
        # LLM 不可靠填 ID：缺省时由调用方通过 ToolContext.extras 注入确定性种子
        seed = args.consultation_id or context.extras.get("consultation_id", 0)
        rng = random.Random(f"{seed}:{args.vital}:{args.abnormal}")
        spec = table[args.vital]
        if args.vital == "blood_pressure":
            (sys_lo, sys_hi), (dia_lo, dia_hi), _, unit = spec
            value = f"{rng.randint(sys_lo, sys_hi)}/{rng.randint(dia_lo, dia_hi)}"
        else:
            lo, hi, digits, unit = spec
            value = str(round(rng.uniform(lo, hi), digits) if digits else rng.randint(int(lo), int(hi)))
        return {"vital": args.vital, "value": value, "unit": unit}
