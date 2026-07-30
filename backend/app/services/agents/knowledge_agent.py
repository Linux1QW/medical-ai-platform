# -*- coding: utf-8 -*-
"""知识核对 Agent（兼容层）

原 1400+ 行实现已拆分至 app/services/agents/knowledge/ 包：
- facts.py    — 结构化病例事实提取
- queries.py  — 三类检索查询构建
- scoring.py  — 评分映射与分析文本生成
- pipeline.py — RAG 管线模式（run_knowledge_check）
- tool_use.py — Tool Use 模式（run_knowledge_check_with_tools）
- react.py    — ReAct 模式（run_knowledge_check_react）

本模块仅保留 re-export，确保既有 import 路径
`app.services.agents.knowledge_agent` 不受影响。
新代码请直接 import `app.services.agents.knowledge`。
"""

from app.services.agents.knowledge import (
    CONSISTENCY_SYSTEM_PROMPT,
    REACT_SYSTEM_PROMPT,
    TOOL_USE_SYSTEM_PROMPT,
    _build_case_query,
    _build_citation_failed_result,
    _build_diagnosis_query,
    _build_error_result,
    _build_rag_trace,
    _build_treatment_query,
    _extract_consultation_data,
    _extract_json,
    _format_tool_trace,
    _generate_analysis,
    _llm_consistency_check,
    _map_consistency_to_score,
    _map_consistency_to_score_v2,
    _parse_react_step,
    _patient_demographic,
    build_queries,
    extract_clinical_facts,
    run_knowledge_check,
    run_knowledge_check_react,
    run_knowledge_check_with_tools,
)

__all__ = [
    "CONSISTENCY_SYSTEM_PROMPT",
    "REACT_SYSTEM_PROMPT",
    "TOOL_USE_SYSTEM_PROMPT",
    "_build_case_query",
    "_build_citation_failed_result",
    "_build_diagnosis_query",
    "_build_error_result",
    "_build_rag_trace",
    "_build_treatment_query",
    "_extract_consultation_data",
    "_extract_json",
    "_format_tool_trace",
    "_generate_analysis",
    "_llm_consistency_check",
    "_map_consistency_to_score",
    "_map_consistency_to_score_v2",
    "_parse_react_step",
    "_patient_demographic",
    "build_queries",
    "extract_clinical_facts",
    "run_knowledge_check",
    "run_knowledge_check_react",
    "run_knowledge_check_with_tools",
]
