from datetime import datetime
from typing import List, Optional, Dict

from sqlalchemy import select, func, and_
from sqlalchemy.ext.asyncio import AsyncSession

from sqlalchemy import delete

from app.models.consultation import Consultation, ConsultationMessage
from app.models.evaluation import Evaluation
from app.models.patient import VirtualPatient
from app.models.user import User
from app.services.qwen_client import call_qwen_chat

# 虚拟患者角色扮演约束：规范、符合医学与病情、不随意扩充
PATIENT_ROLE_WRAPPER = """你正在参与临床医学教学模拟，必须严格扮演患者。你的回答必须规范、符合医学常识，且与档案病情一致。

【身份与语气】
- 你就是患者本人，用第一人称描述自己的感受和症状
- 禁止说"我是AI/助手"等破绽语句，禁止使用颜文字、emoji
- 用口语化、自然的患者语言，语气平实，不夸张不戏剧化

【与病情一致（必须严格遵守）】
- 只陈述下方「患者档案」中已写明的或与该病情医学上合理伴随的症状与病史，不得编造、不得扩充
- 症状必须符合该疾病/病情的临床常见表现，不得出现医学上与该诊断不符的症状（如档案为心绞痛，不得说高热、皮疹等无关表现）
- 不得添加档案中未提及的检查结果、化验数值、既往诊断、用药名称等
- 若医生问及档案未写明的细节：档案有明确信息的按档案答；无明确信息的只可回答“没有/记不清/没做过”，不得自行编造

【医学与表述规范】
- 回答需符合医学常识：症状描述、时间顺序、诱因与缓解因素等应与档案和该病种的典型表现一致
- 不提供医学建议、不替医生下诊断、不使用专业医学术语（用患者能说出口的日常说法）
- 每次回复简洁具体，不冗长、不重复、不随意展开无关内容

【回答风格示例】
示例1：
医生：哪里不舒服？
患者：这几天胸口有点闷，走快一点就更明显，休息一下会好一点。

示例2：
医生：有没有发烧？
患者：没有发烧。

示例3：
医生：做过心电图吗？
患者：这个检查我还没做过。
## 患者档案：
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


async def send_doctor_message(
    db: AsyncSession, consultation_id: int, content: str
) -> tuple[ConsultationMessage, ConsultationMessage]:
    """医生发送消息并获取虚拟患者回复"""
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
    for msg in messages:
        chat_history.append({
            "role": "user" if msg.role == "doctor" else "assistant",
            "content": msg.content,
        })
    chat_history.append({"role": "user", "content": content})

    patient_reply = await call_qwen_chat(chat_history)

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


async def delete_consultation(db: AsyncSession, consultation_id: int, doctor_id: int) -> bool:
    """删除问诊记录（仅允许删除本人的记录），同时删除消息与评估"""
    consultation = await get_consultation(db, consultation_id)
    if not consultation or consultation.doctor_id != doctor_id:
        return False
    await db.execute(delete(ConsultationMessage).where(ConsultationMessage.consultation_id == consultation_id))
    await db.execute(delete(Evaluation).where(Evaluation.consultation_id == consultation_id))
    await db.delete(consultation)
    await db.commit()
    return True
