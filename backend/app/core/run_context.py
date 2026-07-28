# -*- coding: utf-8 -*-
"""评估运行上下文 — contextvars 透传 run_id / agent_name（Harness）

用于 run 级成本归因：evaluation_service 在图执行前 set run_id，
run_agent 节点在各自的 asyncio 任务上下文中 set agent_name（Send 并行
分支各自持有任务级上下文副本，互不污染），token_tracker 记账时读取，
使 qwen_client 无需改动任何调用签名即可获得 run/agent 两个归因维度。
"""

from contextvars import ContextVar

# 当前评估 run（EvaluationRun.id）；None = 不在评估链路内（如问诊聊天）
current_run_id: ContextVar[str | None] = ContextVar("current_run_id", default=None)

# 当前执行中的 agent 名（graph.run_agent 节点内设置）；None = 非 agent 调用
current_agent_name: ContextVar[str | None] = ContextVar(
    "current_agent_name", default=None
)
