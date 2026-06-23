"""测试 LangGraph 编排相关的 ORM 模型"""

import pytest


def test_evaluation_run_model():
    """测试 EvaluationRun 可以实例化"""
    from app.models.evaluation_run import EvaluationRun

    run = EvaluationRun(
        id="test-run-uuid",
        consultation_id=1,
        checkpoint_thread_id="thread-123",
        status="running",
        graph_version="evaluation-graph-v1",
        scoring_policy_version="v1",
        attempt=1,
    )
    assert run.id == "test-run-uuid"
    assert run.consultation_id == 1
    assert run.checkpoint_thread_id == "thread-123"
    assert run.status == "running"
    assert run.graph_version == "evaluation-graph-v1"
    assert run.scoring_policy_version == "v1"
    assert run.attempt == 1

    # 验证表名
    assert EvaluationRun.__tablename__ == "evaluation_runs"


def test_evaluation_node_result_model():
    """测试 EvaluationNodeResult 可以实例化"""
    from app.models.evaluation_node_result import EvaluationNodeResult

    node = EvaluationNodeResult(
        run_id="test-run-uuid",
        node_name="safety_check",
        status="success",
        attempt=1,
    )
    assert node.run_id == "test-run-uuid"
    assert node.node_name == "safety_check"
    assert node.status == "success"
    assert node.attempt == 1

    # 验证表名
    assert EvaluationNodeResult.__tablename__ == "evaluation_node_results"


def test_evaluation_has_new_fields():
    """测试 Evaluation 新字段存在"""
    from app.models.evaluation import Evaluation

    eval_obj = Evaluation(
        consultation_id=1,
        run_id="test-run-uuid",
        safety_data={"passed": True},
        applicable_dimensions=["inquiry", "diagnosis"],
        scoring_policy_version="v1",
        graph_version="evaluation-graph-v1",
    )
    assert eval_obj.run_id == "test-run-uuid"
    assert eval_obj.safety_data == {"passed": True}
    assert eval_obj.applicable_dimensions == ["inquiry", "diagnosis"]
    assert eval_obj.scoring_policy_version == "v1"
    assert eval_obj.graph_version == "evaluation-graph-v1"


def test_consultation_has_consultation_type():
    """测试 Consultation 新字段 consultation_type 存在"""
    from app.models.consultation import Consultation

    consultation = Consultation(
        doctor_id=1,
        patient_id=1,
        consultation_type="follow_up",
    )
    assert consultation.consultation_type == "follow_up"

    # 验证字段在模型中存在（通过检查列属性）
    assert hasattr(Consultation, "consultation_type")
