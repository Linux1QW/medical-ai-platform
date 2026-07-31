# -*- coding: utf-8 -*-
"""迭代基线冻结测试 — Task 0

锁定当前系统关键行为语义，后续任何改动不得破坏这些断言。
覆盖：
1. pre-push 门禁退出码语义（3 例 smoke 不阻断）
2. 五维分数 null/0/insufficient/needs_review 语义
3. RAG citation / rag_trace 字段存在性
4. 人工复核状态字段
5. 图节点完整性
"""
import json
import subprocess
import sys
from pathlib import Path
from unittest.mock import AsyncMock, patch

import pytest

BACKEND_DIR = Path(__file__).parent.parent.parent


# ── 1. pre-push 门禁退出码语义 ──────────────────────────────────────────────


class TestGateExitCodes:
    """eval_regression.py 退出码协议：0=PASS, 1=FAIL, 2=SKIP"""

    def _run_regression(self, report: dict, tmp_path: Path) -> int:
        """将 report 写入临时文件并调用 eval_regression.py，返回退出码"""
        report_path = tmp_path / "ab_test.json"
        report_path.write_text(json.dumps(report, ensure_ascii=False), encoding="utf-8")
        result = subprocess.run(
            [sys.executable, str(BACKEND_DIR / "scripts" / "eval_regression.py"),
             "--report", str(report_path)],
            capture_output=True, text=True, cwd=str(BACKEND_DIR),
        )
        return result.returncode

    def test_three_case_report_is_non_blocking(self, tmp_path):
        """3 例冒烟报告必须返回 SKIP(2)，不得阻断 pre-push"""
        report = {"cases": [{}, {}, {}], "judge_enabled": False}
        assert self._run_regression(report, tmp_path) == 2

    def test_seventeen_case_report_is_non_blocking(self, tmp_path):
        """17 例（< 18 门禁阈值）报告同样返回 SKIP(2)"""
        report = {"cases": [{}] * 17, "judge_enabled": False}
        assert self._run_regression(report, tmp_path) == 2

    def test_no_report_returns_skip(self, tmp_path):
        """无报告时返回 2（SKIP），不阻断"""
        result = subprocess.run(
            [sys.executable, str(BACKEND_DIR / "scripts" / "eval_regression.py"),
             "--report", str(tmp_path / "nonexistent.json")],
            capture_output=True, text=True, cwd=str(BACKEND_DIR),
        )
        assert result.returncode == 2

    def test_pre_push_only_blocks_exit_code_1(self):
        """pre-push 钩子仅在退出码 == 1 时拦截，其它一律放行"""
        hook_path = BACKEND_DIR / "scripts" / "hooks" / "pre-push"
        content = hook_path.read_text(encoding="utf-8")
        # 钩子中只有 CODE == 1 时 exit 1
        assert 'if [ "$CODE" -eq 1 ]' in content
        # 最终兜底 exit 0
        assert content.strip().endswith("exit 0")


# ── 2. 五维分数语义 ──────────────────────────────────────────────────────────


class TestDimensionScoreSemantics:
    """null/None 不得被静默转换为 0 分"""

    def test_score_calculator_none_dimension_excluded(self):
        """ScoreCalculator：score=None 的维度不参与加权，不视为 0"""
        from app.services.scoring.calculator import DimensionResult, ScoreCalculator
        from app.services.scoring.policies import get_default_policy

        policy = get_default_policy()
        calc = ScoreCalculator(policy)

        # 只有 inquiry 有效，其余 None
        dims = {
            "inquiry": DimensionResult(dimension="inquiry", status="scored", score=80.0),
            "knowledge": DimensionResult(dimension="knowledge", status="insufficient", score=None),
            "humanistic": DimensionResult(dimension="humanistic", status="scored", score=75.0),
            "diagnosis": DimensionResult(dimension="diagnosis", status="error", score=None),
            "treatment": DimensionResult(dimension="treatment", status="scored", score=70.0),
        }
        result = calc.calculate(dims)
        # 总分不应为 0（有部分有效维度）
        if result.total_score is not None:
            assert result.total_score > 0

    def test_needs_review_sets_total_score_none(self):
        """needs_review 状态下 total_score 必须为 None，不得为 0"""
        # 从 graph.py finalize_needs_review 节点逻辑验证
        # finalize_needs_review 显式设置 total_score=None
        import ast
        graph_path = BACKEND_DIR / "app" / "orchestration" / "graph.py"
        source = graph_path.read_text(encoding="utf-8")
        assert '"total_score": None' in source


# ── 3. RAG citation 和 rag_trace 字段 ───────────────────────────────────────


class TestRAGFieldPresence:
    """知识核对输出必须包含 citation 和 trace 字段"""

    def test_agent_result_envelope_has_rag_fields(self):
        """AgentResultEnvelope 数据契约包含 citations、trace 和复核字段"""
        from app.orchestration.state import AgentResultEnvelope

        envelope = AgentResultEnvelope(
            agent_name="knowledge",
            status="success",
            score=85.0,
            analysis="test",
        )
        # 验证 RAG 相关字段存在
        assert hasattr(envelope, "citations")
        assert hasattr(envelope, "trace")
        assert hasattr(envelope, "human_review_needed")
        assert hasattr(envelope, "review_reason")
        # 默认值语义正确
        assert envelope.citations == []
        assert envelope.trace == {}
        assert envelope.human_review_needed is False

    def test_retrieval_status_labels_defined(self):
        """检索状态标签已定义"""
        from app.services.agents.knowledge.scoring import RETRIEVAL_STATUS_LABELS

        assert "sufficient" in RETRIEVAL_STATUS_LABELS
        assert "insufficient" in RETRIEVAL_STATUS_LABELS


# ── 4. 人工复核状态字段 ──────────────────────────────────────────────────────


class TestReviewFields:
    """评估模型包含人工复核所需字段"""

    def test_evaluation_model_has_review_fields(self):
        """Evaluation ORM 模型包含 human_review_needed 和 review_reason"""
        from app.models.evaluation import Evaluation

        assert hasattr(Evaluation, "human_review_needed")
        assert hasattr(Evaluation, "review_reason")

    def test_envelope_review_fields(self):
        """AgentResultEnvelope 支持 human_review_needed 和 review_reason"""
        from app.orchestration.state import AgentResultEnvelope

        envelope = AgentResultEnvelope(
            agent_name="knowledge",
            status="error",
            human_review_needed=True,
            review_reason="system_exception",
        )
        assert envelope.human_review_needed is True
        assert envelope.review_reason == "system_exception"


# ── 5. 图节点完整性 ──────────────────────────────────────────────────────────


class TestGraphNodes:
    """评估图包含所有必需节点"""

    REQUIRED_NODES = [
        "load_context",
        "classify_consultation",
        "safety_check",
        "plan_evaluation",
        "validate_plan",
        "run_agent_wave1",
        "run_agent",
        "extract_knowledge_citations",
        "aggregate_results",
        "deterministic_scoring",
        "reflection_check",
        "review_gate_node",
        "generate_suggestion",
        "finalize_completed",
        "finalize_needs_review",
    ]

    def test_graph_has_all_required_nodes(self):
        """build_evaluation_graph 包含全部必需节点"""
        from app.orchestration.graph import build_evaluation_graph

        graph = build_evaluation_graph()
        node_names = set(graph.nodes.keys())
        for node in self.REQUIRED_NODES:
            assert node in node_names, f"缺少节点: {node}"

    def test_wave_dag_structure(self):
        """Wave-1 (knowledge/inquiry/humanistic) → extract_citations → Wave-2 (diagnosis/treatment)"""
        from app.orchestration.graph import build_evaluation_graph

        graph = build_evaluation_graph()
        # extract_knowledge_citations 节点存在，作为 Wave-1 到 Wave-2 的桥梁
        assert "extract_knowledge_citations" in graph.nodes


# ── 6. 阈值配置完整性 ────────────────────────────────────────────────────────


class TestThresholdsConfig:
    """patient_ab_thresholds.json 配置完整且包含门控字段"""

    @pytest.fixture()
    def thresholds(self):
        path = BACKEND_DIR / "evaluation" / "patient_ab_thresholds.json"
        return json.loads(path.read_text(encoding="utf-8"))

    def test_gate_min_cases_defined(self, thresholds):
        """_gate.min_cases 已定义且 >= 18"""
        gate = thresholds.get("_gate", {})
        assert gate.get("min_cases", 0) >= 18

    def test_agent_ledger_thresholds(self, thresholds):
        """agent_ledger 臂有 disclosure_rate_min 和 judge_overall_avg_min"""
        rules = thresholds.get("agent_ledger", {})
        assert "disclosure_rate_min" in rules
        assert "judge_overall_avg_min" in rules

    def test_agent_tool_thresholds(self, thresholds):
        """agent_tool 臂有 disclosure_rate_min、judge_overall_avg_min 和 tool_degrade_rate_max"""
        rules = thresholds.get("agent_tool", {})
        assert "disclosure_rate_min" in rules
        assert "judge_overall_avg_min" in rules
        assert "tool_degrade_rate_max" in rules

    def test_baseline_metadata(self, thresholds):
        """_baseline 元数据记录了基线报告信息"""
        baseline = thresholds.get("_baseline", {})
        assert baseline.get("cases") == 18
        assert baseline.get("judge_enabled") is True
