"""
脱敏患者信息响应模型

使用 Pydantic field_serializer 自动对敏感字段脱敏。
普通 doctor 角色使用这些模型返回数据。
"""

from datetime import datetime

from pydantic import BaseModel, ConfigDict, field_serializer

from app.core.masking import mask_name


class DoctorPatientMaskedOut(BaseModel):
    """医生可见的患者信息（姓名脱敏 + 不含标准答案与系统提示词）"""

    id: int
    name: str
    age: int
    gender: str
    personality_type: str
    chief_complaint: str
    medical_history: str
    symptoms: str
    difficulty_level: int
    created_at: datetime

    model_config = ConfigDict(from_attributes=True)

    @field_serializer("name")
    def _mask_name(self, v: str) -> str:
        return mask_name(v)


class PatientMaskedOut(DoctorPatientMaskedOut):
    """管理员可见的完整患者信息（姓名同样脱敏，但保留诊断和 prompt）"""

    expected_diagnosis: str
    system_prompt: str = ""
