# -*- coding: utf-8 -*-
"""知识核对 Agent 包

由原 knowledge_agent.py（上帝文件）拆分而来：
- facts.py    — 结构化病例事实提取
- queries.py  — 三类检索查询构建
- scoring.py  — 评分映射与分析文本生成
- pipeline.py — RAG 管线模式（run_knowledge_check）
- tool_use.py — Tool Use 模式（run_knowledge_check_with_tools）
- react.py    — ReAct 模式（run_knowledge_check_react）

原 import 路径 app.services.agents.knowledge_agent 通过兼容层继续可用。
"""

from app.services.agents.knowledge.facts import extract_clinical_facts
from app.services.agents.knowledge.pipeline import (
    CONSISTENCY_SYSTEM_PROMPT,
    _llm_consistency_check,
    run_knowledge_check,
)
from app.services.agents.knowledge.queries import (
    _build_case_query,
    _build_diagnosis_query,
    _build_treatment_query,
    _patient_demographic,
    build_queries,
)
from app.services.agents.knowledge.react import (
    REACT_SYSTEM_PROMPT,
    _parse_react_step,
    run_knowledge_check_react,
)
from app.services.agents.knowledge.scoring import (
    _extract_json,
    _generate_analysis,
    _map_consistency_to_score,
    _map_consistency_to_score_v2,
)
from app.services.agents.knowledge.tool_use import (
    TOOL_USE_SYSTEM_PROMPT,
    _build_citation_failed_result,
    _build_error_result,
    _build_rag_trace,
    _extract_consultation_data,
    _format_tool_trace,
    _ToolExecutorBridge,
    run_knowledge_check_with_tools,
)

__all__ = [
    # facts
    "extract_clinical_facts",
    # queries
    "build_queries",
    "_patient_demographic",
    "_build_case_query",
    "_build_diagnosis_query",
    "_build_treatment_query",
    # scoring
    "_extract_json",
    "_map_consistency_to_score",
    "_map_consistency_to_score_v2",
    "_generate_analysis",
    # pipeline
    "CONSISTENCY_SYSTEM_PROMPT",
    "run_knowledge_check",
    "_llm_consistency_check",
    # tool_use
    "TOOL_USE_SYSTEM_PROMPT",
    "_extract_consultation_data",
    "_ToolExecutorBridge",
    "run_knowledge_check_with_tools",
    "_build_rag_trace",
    "_format_tool_trace",
    "_build_error_result",
    "_build_citation_failed_result",
    # react
    "REACT_SYSTEM_PROMPT",
    "_parse_react_step",
    "run_knowledge_check_react",
]
