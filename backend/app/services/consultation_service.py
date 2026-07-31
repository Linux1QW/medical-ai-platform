import json
import logging
from datetime import datetime
from typing import AsyncGenerator, Dict, List, Optional

from sqlalchemy import delete, select
from sqlalchemy.ext.asyncio import AsyncSession

from app.models.consultation import Consultation, ConsultationMessage
from app.models.evaluation import Evaluation
from app.models.patient import VirtualPatient
from app.models.user import User
from app.services.qwen_client import call_qwen_chat
from app.services.agents.patient import MemoryState, PatientAgent, extract_facts
from app.services.agents.patient.prompts import PATIENT_ROLE_WRAPPER, build_role_prompt

logger = logging.getLogger(__name__)

# 滑动窗口配置
MEMORY_RECENT_TURNS = 10   # 完整保留最近10轮（20条消息）
MEMORY_COMPRESS_THRESHOLD = 14  # 超过14轮（28条消息）时触发压缩


async def create_consultation(db: AsyncSession, doctor_id: int, patient_id: int) -> Consultation:
    consultation = Consultation(doctor_id=doctor_id, patient_id=patient_id)
    db.add(consultation)
    await db.commit()
    await db.refresh(consultation)
    return consultation


async def get_consultation(db: AsyncSession, consultation_id: int) -> Optional[Consultation]:
    result = await db.execute(select(Consultation).where(Consultation.id == consultation_id))
    return result.scalar_one_or_none()


async def _get_consultation_or_raise(db: AsyncSession, consultation_id: int) -> Consultation:
    """获取问诊记录，不存在时抛出 ValueError（用于必须存在的场景）"""
    consultation = await get_consultation(db, consultation_id)
    if consultation is None:
        raise ValueError(f"问诊记录不存在: {consultation_id}")
    return consultation


async def list_consultations(
    db: AsyncSession,
    doctor_id: Optional[int] = None,
    filters: Dict = None,
    limit: Optional[int] = None,
    offset: int = 0,
) -> List[Dict]:
    """获取问诊记录列表，联表查询患者信息及评分，支持过滤与分页"""
    query = (
        select(
            Consultation,
            VirtualPatient.name.label("patient_name"),
            VirtualPatient.personality_type.label("personality_type"),
            Evaluation.total_score.label("total_score"),
            User.username.label("doctor_username"),
        )
        .join(VirtualPatient, Consultation.patient_id == VirtualPatient.id)
        .join(User, Consultation.doctor_id == User.id)
        .outerjoin(Evaluation, Consultation.id == Evaluation.consultation_id)
    )

    # 基础过滤：特定医生或全平台（管理员）
    if doctor_id is not None:
        query = query.where(Consultation.doctor_id == doctor_id)

    # 额外过滤条件（用于管理员筛选）
    if filters:
        if filters.get("username"):
            query = query.where(User.username.like(f"%{filters['username']}%"))
        if filters.get("personality"):
            query = query.where(VirtualPatient.personality_type == filters["personality"])
        if filters.get("score_min") is not None:
            query = query.where(Evaluation.total_score >= filters["score_min"])
        if filters.get("score_max") is not None:
            query = query.where(Evaluation.total_score <= filters["score_max"])
        if filters.get("start_time"):
            query = query.where(Consultation.started_at >= filters["start_time"])
        if filters.get("end_time"):
            query = query.where(Consultation.started_at <= filters["end_time"])

    query = query.order_by(Consultation.id.desc())
    if offset:
        query = query.offset(offset)
    if limit is not None:
        query = query.limit(limit)

    result = await db.execute(query)
    rows = result.all()

    consultations = []
    for row in rows:
        c = row.Consultation
        # 计算用时（分钟）
        duration = None
        if c.started_at and c.ended_at:
            duration = int((c.ended_at - c.started_at).total_seconds() / 60)

        # 构建返回对象
        consultation_dict = {
            "id": c.id,
            "doctor_id": c.doctor_id,
            "patient_id": c.patient_id,
            "patient_name": row.patient_name,
            "personality_type": row.personality_type,
            "doctor_username": row.doctor_username,
            "status": c.status,
            "started_at": c.started_at,
            "ended_at": c.ended_at,
            "total_score": row.total_score,
            "duration_minutes": duration,
            "summary": c.summary,
            "diagnosis": c.diagnosis,
            "treatment_plan": c.treatment_plan,
            "created_at": c.created_at
        }
        consultations.append(consultation_dict)

    return consultations


async def get_messages(db: AsyncSession, consultation_id: int) -> List[ConsultationMessage]:
    result = await db.execute(
        select(ConsultationMessage)
        .where(ConsultationMessage.consultation_id == consultation_id)
        .order_by(ConsultationMessage.sequence)
    )
    return list(result.scalars().all())


async def _summarize_early_messages(
    early_messages: List[ConsultationMessage],
    patient_profile: str,
) -> str:
    """将早期对话压缩为结构化摘要，小化 LLM context 占用。

    提取已民露症状、已否认症状、重要病史和患者情绪，返回简洁文本块。
    """
    if not early_messages:
        return ""

    history_lines = []
    for m in early_messages:
        role_label = "医生" if m.role == "doctor" else "患者"
        history_lines.append(f"《{role_label}》{m.content}")
    history_text = "\n".join(history_lines)

    prompt = [
        {
            "role": "system",
            "content": (
                "你是一个医学记录助手。请将以下医患对话压缩为结构化摘要，"
                "重点保留：已民露症状/体征、患者否认的症状、重要病史、患者情绪反应。\n"
                "输出格式（严格按格式，无内容用《无》填写）：\n"
                "【已民露症状】...\n"
                "【否认症状】...\n"
                "【重要病史】...\n"
                "【患者情绪】..."
            ),
        },
        {
            "role": "user",
            "content": (
                f"患者基本情况：{patient_profile[:200]}\n\n"
                f"早期问诊对话（共 {len(early_messages)} 条）：\n{history_text}"
            ),
        },
    ]
    try:
        summary = await call_qwen_chat(prompt, temperature=0.1, max_tokens=300)
        return summary
    except Exception as e:
        logger.warning(f"早期对话摘要生成失败，降级为截断模式: {e}")
        return ""


async def _build_history(messages: List[ConsultationMessage], patient_prompt: str) -> List[Dict[str, str]]:
    """滑动窗口 + 早期摘要，返回不含角色包装头的对话历史（两条生成路径共享）"""
    history: List[Dict[str, str]] = []
    recent_window = MEMORY_RECENT_TURNS * 2  # 每轮 2 条消息
    compress_threshold = MEMORY_COMPRESS_THRESHOLD * 2

    if len(messages) > compress_threshold:
        early_messages = messages[:-recent_window]
        recent_messages = messages[-recent_window:]
        summary = await _summarize_early_messages(early_messages, patient_prompt)
        if summary:
            history.append({
                "role": "system",
                "content": f"《早期问诊记录摘要》（口述展示的症状和对话要点，请保持与此一致）\n{summary}",
            })
    else:
        recent_messages = messages[-recent_window:] if len(messages) > recent_window else messages

    for msg in recent_messages:
        history.append({
            "role": "user" if msg.role == "doctor" else "assistant",
            "content": msg.content,
        })
    return history


async def _legacy_generate_patient_reply(
    patient: VirtualPatient, messages: List[ConsultationMessage], content: str
) -> str:
    """无记忆旧路径（回退兼容）：角色包装 + 滑窗历史直接调 LLM"""
    wrapped_prompt = build_role_prompt(patient.system_prompt or "")
    chat_history = [{"role": "system", "content": wrapped_prompt}]
    chat_history.extend(await _build_history(messages, patient.system_prompt or ""))
    chat_history.append({"role": "user", "content": content})
    return await call_qwen_chat(chat_history, temperature=0.3)


async def _generate_patient_reply(
    consultation: Consultation,
    patient: VirtualPatient,
    messages: List[ConsultationMessage],
    content: str,
) -> str:
    """患者回复生成主入口：优先走披露账本智能体，任意异常回退旧路径，绝不中断问诊"""
    try:
        memory = MemoryState.from_json(consultation.memory_state)
        if memory is None:
            facts = await extract_facts(
                patient.chief_complaint or "",
                patient.medical_history or "",
                patient.symptoms or "",
            )
            memory = MemoryState(facts=facts)
        agent = PatientAgent(patient, memory, consultation_id=consultation.id)
        history = await _build_history(messages, patient.system_prompt or "")
        reply = await agent.respond(content, history)
        consultation.memory_state = memory.to_json()  # 调用方统一 commit
        return reply
    except Exception as e:
        logger.warning(f"患者智能体路径失败，回退无记忆旧路径: {e}", exc_info=True)
        return await _legacy_generate_patient_reply(patient, messages, content)


async def send_doctor_message(
    db: AsyncSession, consultation_id: int, content: str
) -> tuple[ConsultationMessage, ConsultationMessage]:
    """医生发送消息并获取虚拟患者回复

    当对话轮数超过 MEMORY_COMPRESS_THRESHOLD 时，将早期对话压缩为结构化摘要，
    仅保留最近 MEMORY_RECENT_TURNS 轮完整对话，有效控制 LLM context 占用。
    """
    messages = await get_messages(db, consultation_id)
    next_seq = len(messages) + 1

    doctor_msg = ConsultationMessage(
        consultation_id=consultation_id,
        role="doctor",
        content=content,
        sequence=next_seq,
    )
    db.add(doctor_msg)

    consultation = await _get_consultation_or_raise(db, consultation_id)
    patient_result = await db.execute(
        select(VirtualPatient).where(VirtualPatient.id == consultation.patient_id)
    )
    patient = patient_result.scalar_one()

    patient_reply = await _generate_patient_reply(consultation, patient, messages, content)

    patient_msg = ConsultationMessage(
        consultation_id=consultation_id,
        role="patient",
        content=patient_reply,
        sequence=next_seq + 1,
    )
    db.add(patient_msg)

    await db.commit()
    await db.refresh(doctor_msg)
    await db.refresh(patient_msg)
    return doctor_msg, patient_msg


def _make_sse_event(event_type: str, data: dict) -> str:
    """构造 SSE 事件字符串"""
    return f"event: {event_type}\ndata: {json.dumps(data, ensure_ascii=False)}\n\n"


async def send_doctor_message_stream(
    db: AsyncSession, consultation_id: int, content: str
) -> AsyncGenerator[str, None]:
    """医生发送消息并流式获取虚拟患者回复（SSE）

    在每个关键步骤发送进度事件，最终发送完整结果。
    事件类型：progress / complete / error
    """
    try:
        # Step 1: 加载对话历史
        yield _make_sse_event("progress", {
            "step": "loading_history",
            "message": "正在加载对话历史...",
            "progress": 10,
        })
        messages = await get_messages(db, consultation_id)
        next_seq = len(messages) + 1

        # Step 2: 保存医生消息
        yield _make_sse_event("progress", {
            "step": "saving_message",
            "message": "正在保存您的消息...",
            "progress": 20,
        })
        doctor_msg = ConsultationMessage(
            consultation_id=consultation_id,
            role="doctor",
            content=content,
            sequence=next_seq,
        )
        db.add(doctor_msg)

        # Step 3: 加载患者信息
        yield _make_sse_event("progress", {
            "step": "loading_patient",
            "message": "正在加载患者信息...",
            "progress": 30,
        })
        consultation = await _get_consultation_or_raise(db, consultation_id)
        patient_result = await db.execute(
            select(VirtualPatient).where(VirtualPatient.id == consultation.patient_id)
        )
        patient = patient_result.scalar_one()

        # Step 4: 构建对话上下文
        yield _make_sse_event("progress", {
            "step": "building_context",
            "message": "正在构建对话上下文...",
            "progress": 40,
        })

        # Step 5: 调用患者智能体生成回复（内部含滑窗/摘要与异常回退）
        yield _make_sse_event("progress", {
            "step": "generating_reply",
            "message": "患者正在思考回复...",
            "progress": 60,
        })
        patient_reply = await _generate_patient_reply(consultation, patient, messages, content)

        # Step 7: 保存患者回复
        yield _make_sse_event("progress", {
            "step": "saving_reply",
            "message": "正在保存患者回复...",
            "progress": 90,
        })
        patient_msg = ConsultationMessage(
            consultation_id=consultation_id,
            role="patient",
            content=patient_reply,
            sequence=next_seq + 1,
        )
        db.add(patient_msg)
        await db.commit()
        await db.refresh(doctor_msg)
        await db.refresh(patient_msg)

        # Step 8: 完成
        yield _make_sse_event("progress", {
            "step": "completed",
            "message": "完成",
            "progress": 100,
        })
        yield _make_sse_event("complete", {
            "doctor_msg": {
                "id": doctor_msg.id,
                "consultation_id": doctor_msg.consultation_id,
                "role": doctor_msg.role,
                "content": doctor_msg.content,
                "sequence": doctor_msg.sequence,
                "created_at": doctor_msg.created_at.isoformat() if doctor_msg.created_at else None,
            },
            "patient_msg": {
                "id": patient_msg.id,
                "consultation_id": patient_msg.consultation_id,
                "role": patient_msg.role,
                "content": patient_msg.content,
                "sequence": patient_msg.sequence,
                "created_at": patient_msg.created_at.isoformat() if patient_msg.created_at else None,
            },
        })

    except Exception as e:
        logger.error(f"SSE 流式消息处理失败: {e}", exc_info=True)
        # 回滚数据库会话
        try:
            await db.rollback()
        except Exception:
            pass
        yield _make_sse_event("error", {
            "message": f"处理失败: {type(e).__name__}: {str(e)[:200]}",
        })


async def end_consultation(db: AsyncSession, consultation_id: int) -> Consultation:
    consultation = await _get_consultation_or_raise(db, consultation_id)
    consultation.status = "completed"
    consultation.ended_at = datetime.utcnow()
    await db.commit()
    await db.refresh(consultation)
    return consultation


async def submit_diagnosis(
    db: AsyncSession, consultation_id: int, diagnosis: str, treatment_plan: str
) -> Consultation:
    """医生提交诊断结果和治疗方案，同时结束问诊"""
    consultation = await _get_consultation_or_raise(db, consultation_id)
    consultation.diagnosis = diagnosis
    consultation.treatment_plan = treatment_plan
    consultation.status = "completed"
    consultation.ended_at = datetime.utcnow()
    await db.commit()
    await db.refresh(consultation)
    return consultation


async def delete_consultation(db: AsyncSession, consultation_id: int, user) -> bool:
    """删除问诊记录（本人或管理员），同时删除消息与评估"""
    consultation = await get_consultation(db, consultation_id)
    if not consultation:
        return False
    if user.role != "admin" and consultation.doctor_id != user.id:
        return False
    await db.execute(delete(ConsultationMessage).where(ConsultationMessage.consultation_id == consultation_id))
    await db.execute(delete(Evaluation).where(Evaluation.consultation_id == consultation_id))
    await db.delete(consultation)
    await db.commit()
    return True
