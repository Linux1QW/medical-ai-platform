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

logger = logging.getLogger(__name__)

# 滑动窗口配置
MEMORY_RECENT_TURNS = 10   # 完整保留最近10轮（20条消息）
MEMORY_COMPRESS_THRESHOLD = 14  # 超过14轮（28条消息）时触发压缩

# 虚拟患者角色扮演约束：规范、符合医学与病情、不随意扩充
PATIENT_ROLE_WRAPPER = """你正在参与临床医学教学模拟，必须严格扮演患者。你的回答必须规范、符合医学常识，且与档案病情一致。

【身份与语气】
- 你就是患者本人，用第一人称描述自己的感受和症状
- 禁止说"我是AI/助手"等破绽语句，禁止使用颜文字、emoji
- 用口语化、自然的患者语言

【人格特点表达（非常重要）】
- 你必须在每次回复中体现出患者档案中描述的人格特点和情绪状态
- 焦虑型患者：回答时要表现出明显的担忧和不安，会主动追问"是不是很严重""不会是什么大病吧"，语气紧张，容易往坏处想
- 沉默型患者：回答极其简短，能一个字不多说一个字，医生不问就沉默，语气冷淡
- 对抗型患者：态度不耐烦，可能质疑医生"问这些有什么用""上次来也没看好"，语气带刺，但不要拒绝回答
- 配合型患者：态度友好，回答完整清楚，语气温和配合
- 人格表达优先于语气平实的要求——焦虑的患者就应该表现得焦虑，对抗的患者就应该表现得不耐烦

【回复长度与信息量控制（极其重要）】
- 每次回复严格控制在1-3句话以内，绝对不要超过50个字
- 每次回复只回答医生当前这一个问题，只提供1个信息点
- 绝对禁止主动提供医生没有问到的症状、病史、检查结果或其他信息
- 绝对禁止一次性罗列多个症状或多条信息，即使档案中有这些内容
- 如果医生问的是是非题（有没有、是不是），只回答"有/没有/是/不是"，最多加一句简短补充
- 不要使用"另外""还有""同时""此外"等连接词来主动补充额外信息
- 宁可回答得太简短，也不要回答得太详细。真实患者不会一口气把所有症状都说完

【无效输入处理（必须严格遵守）】
- 如果医生发送的是纯数字、单个字符、无意义内容或你无法理解的话，你应该回复类似"医生，您刚才说的我没太听懂，能再说一遍吗？"这样的困惑回应
- 绝对不要把无法理解的输入当作提问来回答，更不要因此主动描述症状
- 如果医生只说了"你好""您好"等问候语，你只需简单回应问候（如"医生您好"），不要主动描述任何症状

【与病情一致（必须严格遵守）】
- 只陈述下方「患者档案」中已写明的或与该病情医学上合理伴随的症状与病史，不得编造、不得扩充
- 症状必须符合该疾病/病情的临床常见表现，不得出现医学上与该诊断不符的症状
- 不得添加档案中未提及的检查结果、化验数值、既往诊断、用药名称等
- 若医生问及档案未写明的细节：档案有明确信息的按档案答；无明确信息的只可回答"没有/记不清/没做过"，不得自行编造

【医学与表述规范】
- 回答需符合医学常识：症状描述、时间顺序、诱因与缓解因素等应与档案和该病种的典型表现一致
- 不提供医学建议、不替医生下诊断
- 禁止使用任何专业医学术语，只用老百姓的日常说法（如说"肚子疼"而不是"腹痛"，说"拉肚子"而不是"腹泻"）
- 不要像医学教科书那样系统描述症状，真实患者说话是零散的、不完整的

【正确回复示例】
配合型：
医生：哪里不舒服？
患者：肚子疼，上面这一块。

焦虑型：
医生：哪里不舒服？
患者：头疼，特别担心……医生，不会是脑子里长什么东西吧？

沉默型：
医生：哪里不舒服？
患者：咳嗽。

对抗型：
医生：哪里不舒服？
患者：肚子疼呗，我都来好几回了也没见好。

【错误回复示例（绝对禁止这样回答）】
医生：哪里不舒服？
患者：我最近半个月一直觉得上腹部隐隐作痛，尤其是饭后比较明显，有时候还会反酸烧心，食欲也下降了不少，体重好像也减轻了一些。之前也有过类似的情况，吃了点胃药就好了。（错误：信息量过大，主动提供了太多医生未问到的内容）

患者档案：
{system_prompt}"""


async def create_consultation(db: AsyncSession, doctor_id: int, patient_id: int) -> Consultation:
    consultation = Consultation(doctor_id=doctor_id, patient_id=patient_id)
    db.add(consultation)
    await db.commit()
    await db.refresh(consultation)
    return consultation


async def get_consultation(db: AsyncSession, consultation_id: int) -> Optional[Consultation]:
    result = await db.execute(select(Consultation).where(Consultation.id == consultation_id))
    return result.scalar_one_or_none()


async def list_consultations(db: AsyncSession, doctor_id: Optional[int] = None, filters: Dict = None) -> List[Dict]:
    """获取问诊记录列表，联表查询患者信息及评分，支持过滤"""
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

    consultation = await get_consultation(db, consultation_id)
    patient_result = await db.execute(
        select(VirtualPatient).where(VirtualPatient.id == consultation.patient_id)
    )
    patient = patient_result.scalar_one()

    wrapped_prompt = PATIENT_ROLE_WRAPPER.format(system_prompt=patient.system_prompt or "")
    chat_history = [{"role": "system", "content": wrapped_prompt}]

    recent_window = MEMORY_RECENT_TURNS * 2  # 每轮 2 条消息
    compress_threshold = MEMORY_COMPRESS_THRESHOLD * 2

    if len(messages) > compress_threshold:
        # 将早期对话压缩为摘要，仅保留最近 recent_window 条完整对话
        early_messages = messages[:-recent_window]
        recent_messages = messages[-recent_window:]
        summary = await _summarize_early_messages(early_messages, patient.system_prompt or "")
        if summary:
            chat_history.append({
                "role": "system",
                "content": f"《早期问诊记录摘要》（口述展示的症状和对话要点，请保持与此一致）\n{summary}",
            })
    else:
        recent_messages = messages[-recent_window:] if len(messages) > recent_window else messages

    for msg in recent_messages:
        chat_history.append({
            "role": "user" if msg.role == "doctor" else "assistant",
            "content": msg.content,
        })
    chat_history.append({"role": "user", "content": content})

    patient_reply = await call_qwen_chat(chat_history, temperature=0.3)

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
        consultation = await get_consultation(db, consultation_id)
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
        wrapped_prompt = PATIENT_ROLE_WRAPPER.format(system_prompt=patient.system_prompt or "")
        chat_history = [{"role": "system", "content": wrapped_prompt}]

        recent_window = MEMORY_RECENT_TURNS * 2
        compress_threshold = MEMORY_COMPRESS_THRESHOLD * 2

        # Step 5: 处理长期记忆压缩（如需要）
        if len(messages) > compress_threshold:
            yield _make_sse_event("progress", {
                "step": "compressing_memory",
                "message": "正在压缩早期对话记忆...",
                "progress": 50,
            })
            early_messages = messages[:-recent_window]
            recent_messages = messages[-recent_window:]
            summary = await _summarize_early_messages(early_messages, patient.system_prompt or "")
            if summary:
                chat_history.append({
                    "role": "system",
                    "content": f"《早期问诊记录摘要》（口述展示的症状和对话要点，请保持与此一致）\n{summary}",
                })
        else:
            recent_messages = messages[-recent_window:] if len(messages) > recent_window else messages

        for msg in recent_messages:
            chat_history.append({
                "role": "user" if msg.role == "doctor" else "assistant",
                "content": msg.content,
            })
        chat_history.append({"role": "user", "content": content})

        # Step 6: 调用 LLM 生成患者回复
        yield _make_sse_event("progress", {
            "step": "generating_reply",
            "message": "患者正在思考回复...",
            "progress": 60,
        })
        patient_reply = await call_qwen_chat(chat_history, temperature=0.3)

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
    consultation = await get_consultation(db, consultation_id)
    consultation.status = "completed"
    consultation.ended_at = datetime.utcnow()
    await db.commit()
    await db.refresh(consultation)
    return consultation


async def submit_diagnosis(
    db: AsyncSession, consultation_id: int, diagnosis: str, treatment_plan: str
) -> Consultation:
    """医生提交诊断结果和治疗方案，同时结束问诊"""
    consultation = await get_consultation(db, consultation_id)
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
