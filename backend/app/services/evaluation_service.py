import asyncio
import json
import logging
from typing import List, Optional

from fastapi import HTTPException, status
from sqlalchemy import func, or_, select
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.config import settings
from app.core.failure_reasons import EvaluationDeadlineExceeded
from app.core.websocket import manager
from app.models.consultation import Consultation, ConsultationMessage
from app.models.evaluation import Evaluation
from app.models.patient import VirtualPatient
from app.models.user import User
from app.services.agents.diagnosis_agent import run_diagnosis_evaluation
from app.services.agents.humanistic_agent import run_humanistic_evaluation
from app.services.agents.inquiry_agent import run_inquiry_analysis
from app.services.agents.knowledge.scoring import (
    EVIDENCE_STANCE_LABELS,
    RETRIEVAL_STATUS_LABELS,
)
from app.services.agents.knowledge_agent import run_knowledge_check
from app.services.agents.scoring_agent import run_scoring
from app.services.agents.suggestion_agent import run_suggestion
from app.services.agents.treatment_agent import run_treatment_evaluation
from app.utils.json_parser import extract_json_from_text


class EvaluationValidationError(Exception):
    """评估结果解析异常"""
    def __init__(self, message: str, raw_response: str):
        self.message = message
        self.raw_response = raw_response
        super().__init__(self.message)


def _extract_json(text: str) -> dict:
    """从 LLM 返回的文本中提取 JSON，解析失败时抛出 ValidationError"""
    try:
        data = extract_json_from_text(text)
    except ValueError as e:
        raise EvaluationValidationError("评估格式异常，无法解析 JSON", text) from e
    if not isinstance(data, dict):
        raise EvaluationValidationError("评估格式异常，JSON 顶层不是对象", text)
    return data


async def run_evaluation(db: AsyncSession, consultation_id: int, resume: bool = False) -> Evaluation:
    """运行评估 — 根据配置选择 LangGraph 图或旧编排

    Args:
        resume: Celery 重试时为 True，尝试从上次失败 run 的 checkpoint 断点续跑
    """
    if settings.LANGGRAPH_ENABLED:
        return await _run_evaluation_graph(db, consultation_id, resume=resume)
    else:
        return await _run_evaluation_legacy(db, consultation_id)


async def _run_evaluation_graph(
    db: AsyncSession, consultation_id: int, resume: bool = False
) -> Evaluation:
    """LangGraph 图执行路径"""
    import uuid
    from datetime import datetime

    from sqlalchemy import select

    from app.core.config import settings
    from app.core.websocket import manager
    from app.models.consultation import Consultation, ConsultationMessage
    from app.models.evaluation_run import EvaluationRun
    from app.models.patient import VirtualPatient
    from app.orchestration.graph import get_graph
    from app.orchestration.progress import send_progress_events
    from app.orchestration.routes import build_submission_flags, get_consultation_type
    from app.orchestration.state import EvaluationContext, EvaluationState

    # 1. 加载数据
    await manager.send_progress(consultation_id, 0, "正在初始化...")

    consultation_result = await db.execute(
        select(Consultation).where(Consultation.id == consultation_id)
    )
    consultation = consultation_result.scalar_one()

    patient_result = await db.execute(
        select(VirtualPatient).where(VirtualPatient.id == consultation.patient_id)
    )
    patient = patient_result.scalar_one()

    msgs = await db.execute(
        select(ConsultationMessage)
        .where(ConsultationMessage.consultation_id == consultation_id)
        .order_by(ConsultationMessage.sequence)
    )
    messages = msgs.scalars().all()

    # 长对话超预算时早期消息摘要化（带 Redis 增量缓存，失败降级全量拼接）
    from app.services.context_budget import build_eval_conversation_text

    conversation_text = await build_eval_conversation_text(
        messages, patient_profile=patient.chief_complaint or ""
    )

    # 2. 构建初始 State（断点续跑时复用失败 run 的标识，保持 checkpoint thread 一致）
    resume_run = None
    if resume and settings.EVAL_RESUME_FROM_CHECKPOINT:
        resume_run = await _find_resumable_run(db, consultation_id)

    run_id = resume_run.id if resume_run is not None else str(uuid.uuid4())
    consultation_type = get_consultation_type(consultation)
    submission_flags = build_submission_flags(consultation)

    context = EvaluationContext(
        conversation_text=conversation_text,
        patient_age=patient.age,
        patient_gender=patient.gender,
        chief_complaint=patient.chief_complaint,
        medical_history=patient.medical_history,
        symptoms=_parse_symptoms(patient.symptoms),
        doctor_diagnosis=consultation.diagnosis if consultation.diagnosis and consultation.diagnosis.strip() else None,
        treatment_plan=consultation.treatment_plan if consultation.treatment_plan and consultation.treatment_plan.strip() else None,
    )

    initial_state = EvaluationState(
        run_id=run_id,
        consultation_id=consultation_id,
        graph_version=settings.LANGGRAPH_GRAPH_VERSION,
        context=context,
        consultation_type=consultation_type,
        submission_flags=submission_flags,
    )

    # 3. 创建/复用 EvaluationRun（断点续跑时翻回 running 并递增 attempt）
    if resume_run is not None:
        eval_run = resume_run
        eval_run.status = "running"
        eval_run.attempt += 1
        eval_run.error_type = None
        eval_run.error_message = None
        eval_run.finished_at = None
        eval_run.updated_at = datetime.utcnow()
    else:
        eval_run = EvaluationRun(
            id=run_id,
            consultation_id=consultation_id,
            graph_version=settings.LANGGRAPH_GRAPH_VERSION,
            scoring_policy_version="v1",
            checkpoint_thread_id=f"evaluation:{run_id}",
            status="running",
            selected_agents=None,
            started_at=datetime.utcnow(),
            created_at=datetime.utcnow(),
            updated_at=datetime.utcnow(),
        )
        db.add(eval_run)
    await db.flush()

    # 4. 执行图
    try:
        graph = await get_graph()
        if graph is None:
            raise RuntimeError("LangGraph 图未初始化，请检查 LANGGRAPH_ENABLED 配置和 Redis Checkpointer 状态")

        config = {
            "configurable": {
                "thread_id": eval_run.checkpoint_thread_id,
            }
        }

        # 断点续跑：存在未完成的 checkpoint 时以 None 输入从断点恢复，否则全新执行
        graph_input = initial_state
        if resume_run is not None:
            graph_input = await _resolve_graph_input(graph, config, initial_state)

        final_state = await _invoke_graph_with_deadline(graph, graph_input, config)

        # 5. 发送进度事件
        progress_events = final_state.get("progress_events", [])
        await send_progress_events(consultation_id, progress_events)

        # 6. 从 final_state 构建 Evaluation
        evaluation = _build_evaluation_from_state(final_state, consultation_id)

        # 7. 更新 Run 状态
        eval_run.status = final_state.get("evaluation_status", "completed")
        eval_run.selected_agents = (
            [r.agent_name for r in final_state.get("agent_results", [])]
            if final_state.get("agent_results") else None
        )

        # 保存评估计划（Plan-Execute 模式）
        eval_plan = final_state.get("evaluation_plan")
        if eval_plan is not None:
            eval_run.evaluation_plan = eval_plan.model_dump() if hasattr(eval_plan, "model_dump") else eval_plan

        # 保存执行结果
        exec_results = final_state.get("execution_results")
        if exec_results:
            eval_run.execution_results = [
                r.model_dump() if hasattr(r, "model_dump") else r
                for r in exec_results
            ]

        eval_run.finished_at = datetime.utcnow()

        # 7.5 持久化 run 级 trace 树：每个 agent 节点一行（含工具调用 trace）+ 节点错误
        node_rows = _build_node_results(run_id, final_state)
        if node_rows:
            db.add_all(node_rows)

        # 8. 保存 Evaluation
        db.add(evaluation)
        consultation.status = "evaluated"
        await db.commit()
        await db.refresh(evaluation)

        # 更新 run 的 evaluation_id
        eval_run.evaluation_id = evaluation.id
        await db.commit()

        await manager.send_progress(consultation_id, 100, "评估完成")
        return evaluation

    except Exception as e:
        from app.core.failure_reasons import classify_failure

        logging.error(f"LangGraph 评估流程异常: {e}")
        eval_run.status = "failed"
        # error_type 存标准化原因码（供监控聚合），原始类名保留在 error_message
        eval_run.error_type = classify_failure(e)
        eval_run.error_message = f"{type(e).__name__}: {e}"[:500]
        eval_run.finished_at = datetime.utcnow()
        await db.commit()
        raise


async def _find_resumable_run(db: AsyncSession, consultation_id: int):
    """查找最近一次 failed 的 EvaluationRun 用于断点续跑（无则返回 None）"""
    from sqlalchemy import select

    from app.models.evaluation_run import EvaluationRun

    result = await db.execute(
        select(EvaluationRun)
        .where(
            EvaluationRun.consultation_id == consultation_id,
            EvaluationRun.status == "failed",
        )
        .order_by(EvaluationRun.started_at.desc())
        .limit(1)
    )
    return result.scalars().first()


async def _resolve_graph_input(graph, config, initial_state):
    """断点续跑时解析图输入

    thread 存在未完成的 checkpoint（values 非空且有待执行节点）时返回 None，
    LangGraph 以空输入 + 同 thread_id 调用即从最后一个成功 superstep 恢复；
    无 checkpoint 或读取失败时降级为全新执行（返回 initial_state）。
    """
    try:
        snapshot = await graph.aget_state(config)
    except Exception as e:
        logging.warning(f"读取 checkpoint 状态失败，降级为全新执行: {e}")
        return initial_state
    if snapshot is not None and getattr(snapshot, "values", None) and getattr(snapshot, "next", None):
        logging.info(f"从 checkpoint 断点续跑: 待执行节点={snapshot.next}")
        return None
    return initial_state


async def _invoke_graph_with_deadline(graph, initial_state, config):
    """执行图并施加评估级总时长预算（EVALUATION_RUN_TIMEOUT_SECONDS，0 = 不限制）

    超时抛 EvaluationDeadlineExceeded（TimeoutError 子类）：
    走既有失败路径落库 error_type="timeout"，并命中 Celery 可重试判定；
    在 Celery 软超时（300s）强杀前优雅降级，避免 run 状态丢失。
    """
    timeout = settings.EVALUATION_RUN_TIMEOUT_SECONDS
    if timeout <= 0:
        return await graph.ainvoke(initial_state, config=config)
    try:
        return await asyncio.wait_for(
            graph.ainvoke(initial_state, config=config), timeout=timeout
        )
    except asyncio.TimeoutError:
        raise EvaluationDeadlineExceeded(
            f"评估超出总时长预算 {timeout}s"
        ) from None


def _build_node_results(run_id: str, state: dict) -> list:
    """从 final state 构建 EvaluationNodeResult 审计行（run 级 trace 树）

    以 run_id 为根：
    - 每个 agent 执行一行（优先用 execution_results，含耗时/时间戳，
      result_summary.trace 携带工具调用明细，不再仅 knowledge 一路落库）
    - 未被 agent 行覆盖的 node_errors 各补一行 error 记录
    """
    from datetime import datetime

    from app.models.evaluation_node_result import EvaluationNodeResult

    def _parse_ts(value):
        try:
            return datetime.fromisoformat(value) if value else None
        except (TypeError, ValueError):
            return None

    node_errors = list(state.get("node_errors") or [])
    error_by_node = {err.node_name: err for err in node_errors}

    rows: list[EvaluationNodeResult] = []
    covered_agents: set[str] = set()

    for result in state.get("execution_results") or []:
        envelope = result.envelope
        matched_error = error_by_node.get(f"run_agent:{result.agent_name}")
        rows.append(
            EvaluationNodeResult(
                run_id=run_id,
                node_name=f"agent:{result.agent_name}",
                status=result.status,
                duration_ms=result.duration_ms,
                result_summary={
                    "step_id": result.step_id,
                    "score": result.score,
                    "skip_reason": envelope.skip_reason if envelope else None,
                    "review_reason": envelope.review_reason if envelope else None,
                    "trace": envelope.trace if envelope else {},
                },
                error_type=matched_error.error_type if matched_error else None,
                started_at=_parse_ts(result.started_at),
                finished_at=_parse_ts(result.finished_at),
            )
        )
        covered_agents.add(result.agent_name)

    # 旧 dispatch_and_run 路径只产生 agent_results，兼容补齐
    for envelope in state.get("agent_results") or []:
        if envelope.agent_name in covered_agents:
            continue
        rows.append(
            EvaluationNodeResult(
                run_id=run_id,
                node_name=f"agent:{envelope.agent_name}",
                status=envelope.status,
                result_summary={
                    "score": envelope.score,
                    "skip_reason": envelope.skip_reason,
                    "review_reason": envelope.review_reason,
                    "trace": envelope.trace,
                },
            )
        )
        covered_agents.add(envelope.agent_name)

    # 非 agent 节点的错误（或 agent 行未覆盖的）单独落审计行
    for err in node_errors:
        agent_name = err.node_name.split(":", 1)[1] if ":" in err.node_name else None
        if agent_name and agent_name in covered_agents:
            continue
        rows.append(
            EvaluationNodeResult(
                run_id=run_id,
                node_name=err.node_name,
                attempt=err.attempt,
                status="error",
                result_summary={"error_message": err.error_message},
                error_type=err.error_type,
            )
        )

    return rows


def _build_evaluation_from_state(state: dict, consultation_id: int) -> Evaluation:
    """从 LangGraph final state 构建 Evaluation ORM 对象"""
    from app.models.evaluation import Evaluation

    dimensions = state.get("dimension_results", {})

    def get_dim(name):
        dim = dimensions.get(name)
        if dim and dim.status == "scored":
            return dim.score, dim.analysis
        return None, (dim.analysis if dim else "")

    inquiry_score, inquiry_analysis = get_dim("inquiry")
    knowledge_score, knowledge_analysis = get_dim("knowledge")
    humanistic_score, humanistic_analysis = get_dim("humanistic")
    diagnosis_score, diagnosis_analysis = get_dim("diagnosis")
    treatment_score, treatment_analysis = get_dim("treatment")

    # 从 agent_results 提取 RAG 审计字段
    citation_data = None
    retrieval_status = "not_run"
    evidence_stance = "undetermined"
    rag_trace_data = None

    for result in state.get("agent_results", []):
        if result.agent_name == "knowledge":
            citation_data = result.citations or None
            rag_trace_data = result.trace or None
            if result.status == "insufficient":
                retrieval_status = "insufficient"
                evidence_stance = "refusal"

    # 改进建议
    suggestions = state.get("improvement_suggestions", [])
    suggestions_text = "\n".join(suggestions) if suggestions else ""

    eval_status = state.get("evaluation_status", "completed")

    return Evaluation(
        consultation_id=consultation_id,
        inquiry_score=inquiry_score or 0,
        inquiry_analysis=inquiry_analysis,
        knowledge_score=knowledge_score,
        knowledge_analysis=knowledge_analysis,
        humanistic_score=humanistic_score or 0,
        humanistic_analysis=humanistic_analysis,
        diagnosis_score=diagnosis_score or 0,
        diagnosis_analysis=diagnosis_analysis,
        treatment_score=treatment_score or 0,
        treatment_analysis=treatment_analysis,
        total_score=state.get("total_score"),
        overall_summary=state.get("overall_summary", ""),
        improvement_suggestions=suggestions_text,
        citation_data=citation_data,
        retrieval_status=retrieval_status,
        evidence_stance=evidence_stance,
        human_review_needed=state.get("human_review_needed", False),
        review_reason=state.get("review_reason"),
        rag_trace_data=rag_trace_data,
        evaluation_status=eval_status,
        # LangGraph 审计字段
        run_id=state.get("run_id"),
        safety_data=(
            safety_result.model_dump()
            if (safety_result := state.get("safety_result")) is not None
            else None
        ),
        applicable_dimensions=list(dimensions.keys()) if dimensions else None,
        scoring_policy_version=state.get("scoring_policy_version"),
        graph_version=state.get("graph_version"),
    )


def _parse_symptoms(symptoms_str) -> list[str]:
    """解析患者症状字段"""
    if not symptoms_str:
        return []
    try:
        data = json.loads(symptoms_str)
        if isinstance(data, list):
            return data
        return [symptoms_str]
    except (json.JSONDecodeError, TypeError):
        return [symptoms_str] if symptoms_str else []


async def _run_evaluation_legacy(db: AsyncSession, consultation_id: int) -> Evaluation:
    """旧编排路径 — 保留原有代码作为回退"""
    await manager.send_progress(consultation_id, 0, "正在初始化...")
    consultation_result = await db.execute(
        select(Consultation).where(Consultation.id == consultation_id)
    )
    consultation = consultation_result.scalar_one()

    patient_result = await db.execute(
        select(VirtualPatient).where(VirtualPatient.id == consultation.patient_id)
    )
    patient = patient_result.scalar_one()

    msgs = await db.execute(
        select(ConsultationMessage)
        .where(ConsultationMessage.consultation_id == consultation_id)
        .order_by(ConsultationMessage.sequence)
    )
    messages = msgs.scalars().all()

    conversation_text = "\n".join(
        f"{'医生' if m.role == 'doctor' else '患者'}: {m.content}" for m in messages
    )
    patient_info = (
        f"姓名: {patient.name}, 年龄: {patient.age}, 性别: {patient.gender}\n"
        f"人格类型: {patient.personality_type}\n"
        f"主诉: {patient.chief_complaint}\n"
        f"病史: {patient.medical_history}\n"
        f"症状: {patient.symptoms}\n"
        f"预期诊断: {patient.expected_diagnosis}"
    )
    doctor_diagnosis = consultation.diagnosis or "（医生未提交诊断结果）"
    treatment_plan = consultation.treatment_plan or "（医生未提交治疗方案）"

    # 用于跟踪并行任务进度的计数器
    completed_agents = {"count": 0}
    agent_names = ["病史采集", "医学知识", "沟通交流", "诊断结果", "治疗方案"]
    failed_agents: list[str] = []

    async def run_with_progress(coro, agent_index: int):
        """包装异步任务，完成后更新进度"""
        result = await coro
        completed_agents["count"] += 1
        progress = 10 + completed_agents["count"] * 10  # 10% -> 20% -> 30% -> 40% -> 50% -> 60%
        completed_name = agent_names[agent_index]
        remaining = 5 - completed_agents["count"]
        if remaining > 0:
            msg = f"{completed_name}评估完成，还有 {remaining} 项评估进行中..."
        else:
            msg = "五维评估全部完成，正在汇总..."
        await manager.send_progress(consultation_id, progress, msg)
        return result

    async def run_agent_safe(coro, agent_index: int):
        """单 Agent 容错包装：超时/异常时返回降级结果，不拖垮其余维度

        与 LangGraph 路径语义对齐：单 Agent 失败 → 该维度降级 + 整体强制人工复核。
        """
        name = agent_names[agent_index]
        try:
            return await asyncio.wait_for(
                run_with_progress(coro, agent_index),
                timeout=settings.AGENT_TIMEOUT_SECONDS,
            )
        except Exception as e:
            logging.error(f"{name}评估Agent执行失败: {e}", exc_info=True)
            failed_agents.append(name)
            completed_agents["count"] += 1
            try:
                await manager.send_progress(
                    consultation_id,
                    10 + completed_agents["count"] * 10,
                    f"{name}评估失败，其余维度继续...",
                )
            except Exception:
                pass
            fallback = {
                "score": 0,
                "analysis": f"{name}维度自动评估失败（{type(e).__name__}），该结果需人工复核。",
            }
            return {"raw_response": json.dumps(fallback, ensure_ascii=False)}

    try:
        # 第一阶段：并行调用五个评估智能体
        await manager.send_progress(consultation_id, 5, "正在启动五维评估智能体...")
        await asyncio.sleep(0.1)  # 短暂延迟确保 WebSocket 消息发送
        await manager.send_progress(consultation_id, 10, "病史采集、医学知识、沟通交流、诊断结果、治疗方案评估中...")

        (
            inquiry_result,
            knowledge_result,
            humanistic_result,
            diagnosis_result,
            treatment_result,
        ) = await asyncio.gather(
            run_agent_safe(run_inquiry_analysis(conversation_text, patient_info), 0),
            run_agent_safe(run_knowledge_check(conversation_text, patient_info, doctor_diagnosis, treatment_plan), 1),
            run_agent_safe(run_humanistic_evaluation(conversation_text, patient_info), 2),
            run_agent_safe(run_diagnosis_evaluation(conversation_text, patient_info, doctor_diagnosis), 3),
            run_agent_safe(run_treatment_evaluation(conversation_text, patient_info, doctor_diagnosis, treatment_plan), 4),
        )

        inquiry_data = _extract_json(inquiry_result["raw_response"])
        knowledge_data = _extract_json(knowledge_result["raw_response"])
        humanistic_data = _extract_json(humanistic_result["raw_response"])
        diagnosis_data = _extract_json(diagnosis_result["raw_response"])
        treatment_data = _extract_json(treatment_result["raw_response"])

        # 提取 knowledge_agent 新增的 RAG 审计字段
        retrieval_status = knowledge_result.get("retrieval_status", "not_run")
        evidence_stance = knowledge_result.get("evidence_stance", "undetermined")
        human_review_needed = knowledge_result.get("human_review_needed", False)
        review_reason = knowledge_result.get("review_reason")
        citations = knowledge_result.get("citations", [])
        rag_trace = knowledge_result.get("rag_trace", {})

        # 任一 Agent 执行失败 → 强制人工复核（与 LangGraph 路径语义一致）
        if failed_agents:
            human_review_needed = True
            review_reason = review_reason or ("agent_error: " + "、".join(failed_agents))

        # 拒答逻辑：如果 knowledge_agent 标记需要人工复核，knowledge_score 置 None
        knowledge_score_value = knowledge_data.get("score")
        if human_review_needed:
            knowledge_score_value = None

        # 第二阶段：综合评分智能体
        await manager.send_progress(consultation_id, 70, "综合评分计算中...")
        scoring_result = await run_scoring(
            inquiry_score=inquiry_data.get("score", 0),
            inquiry_analysis=inquiry_data.get("analysis", ""),
            knowledge_score=knowledge_score_value,
            knowledge_analysis=knowledge_data.get("analysis", ""),
            humanistic_score=humanistic_data.get("score", 0),
            humanistic_analysis=humanistic_data.get("analysis", ""),
            diagnosis_score=diagnosis_data.get("score"),
            diagnosis_analysis=diagnosis_data.get("analysis", ""),
            treatment_score=treatment_data.get("score"),
            treatment_analysis=treatment_data.get("analysis", ""),
        )
        scoring_data = _extract_json(scoring_result["raw_response"])

        # 第三阶段：建议指导智能体
        await manager.send_progress(consultation_id, 85, "生成改进建议中...")
        suggestion_result = await run_suggestion(
            conversation_text=conversation_text,
            patient_info=patient_info,
            inquiry_result=inquiry_result["raw_response"],
            knowledge_result=knowledge_result["raw_response"],
            humanistic_result=humanistic_result["raw_response"],
        )
        suggestion_data = _extract_json(suggestion_result["raw_response"])

        await manager.send_progress(consultation_id, 95, "正在保存评估结果...")

        # 拒答时 total_score 也置为 None，并替换摘要避免分数矛盾
        total_score_value = scoring_data.get("total_score")
        summary = scoring_data.get("summary", scoring_result["raw_response"])
        if human_review_needed:
            total_score_value = None
            # 使用预定义拒答摘要，避免摘要中出现与数据库 null 矛盾的总分描述
            inquiry_score_value = inquiry_data.get("score", 0)
            humanistic_score_value = humanistic_data.get("score", 0)
            diagnosis_score_value = diagnosis_data.get("score", 0)
            treatment_score_value = treatment_data.get("score", 0)
            if failed_agents:
                summary = (
                    "部分维度自动评估执行失败（{}），总分待人工复核。"
                    "各维度结果：问诊技巧 {}分，人文关怀 {}分，诊断能力 {}分，治疗方案 {}分。"
                ).format(
                    "、".join(failed_agents),
                    inquiry_score_value, humanistic_score_value,
                    diagnosis_score_value, treatment_score_value,
                )
            else:
                summary = (
                    "知识维度评估证据不足（检索状态：{}，证据立场：{}），总分待人工复核。"
                    "其余维度评估：问诊技巧 {}分，人文关怀 {}分，诊断能力 {}分，治疗方案 {}分。"
                ).format(
                    RETRIEVAL_STATUS_LABELS.get(retrieval_status, retrieval_status),
                    EVIDENCE_STANCE_LABELS.get(evidence_stance, evidence_stance),
                    inquiry_score_value, humanistic_score_value,
                    diagnosis_score_value, treatment_score_value,
                )

        evaluation = Evaluation(
            consultation_id=consultation_id,
            inquiry_score=inquiry_data.get("score"),
            inquiry_analysis=inquiry_data.get("analysis", inquiry_result["raw_response"]),
            knowledge_score=knowledge_score_value,
            knowledge_analysis=knowledge_data.get("analysis", knowledge_result["raw_response"]),
            humanistic_score=humanistic_data.get("score"),
            humanistic_analysis=humanistic_data.get("analysis", humanistic_result["raw_response"]),
            diagnosis_score=diagnosis_data.get("score"),
            diagnosis_analysis=diagnosis_data.get("analysis", diagnosis_result["raw_response"]),
            treatment_score=treatment_data.get("score"),
            treatment_analysis=treatment_data.get("analysis", treatment_result["raw_response"]),
            total_score=total_score_value,
            overall_summary=summary,
            improvement_suggestions=suggestion_data.get(
                "suggestions", suggestion_result["raw_response"]
            ),
            # RAG 审计字段
            citation_data=citations,
            retrieval_status=retrieval_status,
            evidence_stance=evidence_stance,
            human_review_needed=human_review_needed,
            review_reason=review_reason,
            rag_trace_data=rag_trace,
            evaluation_status="needs_review" if human_review_needed else "completed",
        )

        # 非拒答情况下确保其他维度分数不为 None（knowledge_score 和 total_score 允许 None）
        if any(s is None for s in [evaluation.inquiry_score,
                                 evaluation.humanistic_score, evaluation.diagnosis_score,
                                 evaluation.treatment_score]):
             raise EvaluationValidationError("评估 JSON 缺少分数字段", str(scoring_data))

        db.add(evaluation)
        consultation.status = "evaluated"
        await db.commit()
        await db.refresh(evaluation)
        await manager.send_progress(consultation_id, 100, "评估完成")
        return evaluation

    except EvaluationValidationError as e:
        logging.error(f"评估JSON解析失败: {e.message}\n原始返回体: {e.raw_response}")
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail={
                "error_code": "EVALUATION_VALIDATION_ERROR",
                "error_type": "ValidationError",
                "message": "评估格式异常，请稍后重试",
            }
        ) from e
    except Exception as e:
        logging.error(f"评估流程发生异常: {str(e)}")
        await db.rollback()
        raise e


async def get_evaluation_by_consultation(
    db: AsyncSession, consultation_id: int
) -> Optional[Evaluation]:
    result = await db.execute(
        select(Evaluation).where(Evaluation.consultation_id == consultation_id)
    )
    return result.scalar_one_or_none()


def _score_range_label(score: float) -> str:
    if score >= 90:
        return "优秀(90-100)"
    if score >= 80:
        return "良好(80-89)"
    if score >= 60:
        return "一般(60-79)"
    return "不及格(<60)"


async def get_stats(db: AsyncSession, doctor_id: Optional[int] = None) -> dict:
    """统计：doctor_id 为 None 时统计全平台（管理员），否则统计该医生本人"""

    def _q_consultations(base):
        if doctor_id is not None:
            return base.where(Consultation.doctor_id == doctor_id)
        return base

    q_cons = _q_consultations(select(func.count(Consultation.id)))
    total_consultations = await db.execute(q_cons)

    q_evals_count = select(func.count(Evaluation.id)).select_from(Evaluation).join(
        Consultation, Consultation.id == Evaluation.consultation_id
    ).where(Evaluation.evaluation_status != "needs_review")
    if doctor_id is not None:
        q_evals_count = q_evals_count.where(Consultation.doctor_id == doctor_id)
    total_evaluations = await db.execute(q_evals_count)

    q_avgs = (
        select(
            func.avg(Evaluation.inquiry_score),
            func.avg(Evaluation.knowledge_score),
            func.avg(Evaluation.humanistic_score),
            func.avg(Evaluation.diagnosis_score),
            func.avg(Evaluation.treatment_score),
            func.avg(Evaluation.total_score),
        )
        .join(Consultation, Consultation.id == Evaluation.consultation_id)
        .where(Evaluation.evaluation_status != "needs_review")
        .where(Evaluation.total_score.isnot(None))
    )
    if doctor_id is not None:
        q_avgs = q_avgs.where(Consultation.doctor_id == doctor_id)
    avg_scores = await db.execute(q_avgs)
    avgs = avg_scores.one()

    # 按用户分组计算平均分，然后按平均分统计分布（而非按每份报告统计）
    if doctor_id is not None:
        # 个人统计：直接用该用户的平均分归类
        user_avg = avgs[5]  # avg_total_score
        distribution = {"优秀(90-100)": 0, "良好(80-89)": 0, "一般(60-79)": 0, "不及格(<60)": 0}
        if user_avg is not None:
            label = _score_range_label(user_avg)
            distribution[label] = 1
    else:
        # 管理员统计：按用户分组计算每个用户的平均分，再统计分布
        q_user_avgs = (
            select(
                Consultation.doctor_id,
                func.avg(Evaluation.total_score).label("avg_score"),
            )
            .join(Consultation, Consultation.id == Evaluation.consultation_id)
            .where(Evaluation.evaluation_status != "needs_review")
            .where(Evaluation.total_score.isnot(None))
            .group_by(Consultation.doctor_id)
        )
        user_avgs_result = await db.execute(q_user_avgs)
        user_avgs_rows = user_avgs_result.all()

        distribution = {"优秀(90-100)": 0, "良好(80-89)": 0, "一般(60-79)": 0, "不及格(<60)": 0}
        for row in user_avgs_rows:
            if row.avg_score is not None:
                label = _score_range_label(row.avg_score)
                distribution[label] += 1

    score_distribution = [{"range": k, "count": v} for k, v in distribution.items()]

    return {
        "total_consultations": total_consultations.scalar() or 0,
        "total_evaluations": total_evaluations.scalar() or 0,
        "avg_inquiry_score": round(avgs[0] or 0, 1),
        "avg_knowledge_score": round(avgs[1] or 0, 1),
        "avg_humanistic_score": round(avgs[2] or 0, 1),
        "avg_diagnosis_score": round(avgs[3] or 0, 1),
        "avg_treatment_score": round(avgs[4] or 0, 1),
        "avg_total_score": round(avgs[5] or 0, 1),
        "score_distribution": score_distribution,
    }


async def get_user_stats_breakdown(db: AsyncSession) -> List[dict]:
    """管理员视图：按用户分组统计各维度平均分"""
    q = (
        select(
            User.id.label("user_id"),
            User.username,
            User.real_name,
            User.department,
            func.count(func.distinct(Consultation.id)).label("total_consultations"),
            func.count(func.distinct(Evaluation.id)).label("total_evaluations"),
            func.avg(Evaluation.inquiry_score).label("avg_inquiry_score"),
            func.avg(Evaluation.knowledge_score).label("avg_knowledge_score"),
            func.avg(Evaluation.humanistic_score).label("avg_humanistic_score"),
            func.avg(Evaluation.diagnosis_score).label("avg_diagnosis_score"),
            func.avg(Evaluation.treatment_score).label("avg_treatment_score"),
            func.avg(Evaluation.total_score).label("avg_total_score"),
        )
        .select_from(User)
        .outerjoin(Consultation, Consultation.doctor_id == User.id)
        .outerjoin(Evaluation, Evaluation.consultation_id == Consultation.id)
        .where(User.role == "doctor")
        .where(
            or_(
                Evaluation.evaluation_status != "needs_review",
                Evaluation.evaluation_status.is_(None),  # 保留 outerjoin 未匹配的记录
            )
        )
        .group_by(User.id, User.username, User.real_name, User.department)
        .order_by(func.avg(Evaluation.total_score).desc())
    )
    result = await db.execute(q)
    rows = result.all()
    return [
        {
            "user_id": row.user_id,
            "username": row.username,
            "real_name": row.real_name or "",
            "department": row.department or "",
            "total_consultations": row.total_consultations,
            "total_evaluations": row.total_evaluations,
            "avg_inquiry_score": round(row.avg_inquiry_score or 0, 1),
            "avg_knowledge_score": round(row.avg_knowledge_score or 0, 1),
            "avg_humanistic_score": round(row.avg_humanistic_score or 0, 1),
            "avg_diagnosis_score": round(row.avg_diagnosis_score or 0, 1),
            "avg_treatment_score": round(row.avg_treatment_score or 0, 1),
            "avg_total_score": round(row.avg_total_score or 0, 1),
        }
        for row in rows
    ]
