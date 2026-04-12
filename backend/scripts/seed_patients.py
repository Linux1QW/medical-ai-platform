# -*- coding: utf-8 -*-
"""从 dataset 数据集中导入虚拟患者到数据库"""

import asyncio
import json
import sys
from pathlib import Path

# 设置终端输出编码为UTF-8，避免Windows终端乱码
sys.stdout.reconfigure(encoding='utf-8')

# 添加项目路径
sys.path.insert(0, str(Path(__file__).parent.parent))

from sqlalchemy import select, func
from sqlalchemy.ext.asyncio import create_async_engine, AsyncSession
from sqlalchemy.orm import sessionmaker

from app.core.config import settings
from app.models.patient import VirtualPatient


# 人格类型映射：数据集 -> 数据库枚举
PERSONALITY_MAP = {
    "合作": "配合型",
    "偏执": "对抗型",
    "啰嗦": "焦虑型",
    "怀疑": "沉默型",
}

# 从数据集中选取的患者样本（覆盖不同人格类型和症状）
PATIENT_SAMPLES = [
    # (文件夹名, 人格覆盖, 难度等级)
    ("patient100_21", "配合型", 2),  # 29岁女，慢性胃炎，大便不成形
    ("patient110_23", "焦虑型", 3),  # 48岁女，大便不成形，啰嗦型
    ("patient120_25", "对抗型", 4),  # 54岁男，便秘，偏执型，代替家人咨询
    ("patient130_27", "配合型", 3),  # 56岁女，腹胀半年，子宫肌瘤术后
    ("patient140_32", "对抗型", 4),  # 64岁女，右下腹疼痛，偏执型
    ("patient50_13", "配合型", 2),   # 46岁女，食道哽噎感
    ("patient1_5", "配合型", 1),     # 简单病例
    ("patient11_9", "沉默型", 3),    # 中等复杂度
    ("patient103_21", "沉默型", 3),  # 怀疑型
    ("patient105_22", "对抗型", 4),  # 偏执型
]


def generate_system_prompt(patient_data: dict, personality_type: str) -> str:
    """根据患者数据生成系统提示词"""
    basic_info = patient_data.get("基础信息", {})
    personality = patient_data.get("人格", {})
    medical_record = patient_data.get("门诊病历", {})
    
    name = basic_info.get("姓名", "患者")
    age = basic_info.get("年龄", "未知")
    gender = "女性" if basic_info.get("性别") == "女" else "男性"
    
    chief_complaint = medical_record.get("主诉", "")
    current_illness = medical_record.get("现病史", "")
    past_history = medical_record.get("既往史", "既往体健")
    
    # 根据人格类型生成对话风格（所有类型均强调简短回复、不主动提供额外信息）
    personality_styles = {
        "配合型": "您是一位配合度较高的患者，态度友善，愿意回答医生的问题。但您只回答医生问到的内容，不会主动补充其他症状或信息。回答简短直接，像普通人看病一样说话。",
        "焦虑型": "您是一位较为焦虑的患者，会表现出对病情的担忧，可能反复问'严不严重''要不要紧'。但焦虑体现在情绪上，不是信息量上——您仍然只回答医生问到的问题，不会因为紧张就把所有症状一股脑说出来。",
        "沉默型": "您是一位比较沉默寡言的患者，回答问题非常简短，经常只用几个字回答。需要医生反复追问才会提供更多细节。可能对医生的建议持怀疑态度，不太愿意多说。",
        "对抗型": "您是一位有些偏执的患者，可能对之前的就医经历不满意，会质疑医生的诊断和建议。回答问题时态度不太耐烦，但也只回答被问到的内容，不会主动展开。",
    }
    
    style = personality_styles.get(personality_type, personality_styles["配合型"])
    
    prompt = f"""你是一位正在就诊的虚拟患者，请根据以下信息进行角色扮演：

【基本信息】
- 姓名：{name}
- 年龄：{age}岁
- 性别：{gender}

【就诊原因】
- 主诉：{chief_complaint}
- 现病史：{current_illness}
- 既往史：{past_history}

【人格特征】
{style}

【对话要求】
1. 请用第一人称回答医生的问题
2. 根据人格特征调整回答方式和语气
3. 不要主动说出诊断结果，让医生通过问诊来判断
4. 回答要符合普通患者的医学认知水平，不要使用医学术语
5. 每次只回答医生问的那个问题，不要主动补充其他症状或信息
6. 回复要简短，控制在1-3句话以内
7. 医生没问到的内容绝对不要主动提起
"""
    return prompt


async def load_patient_data(folder_name: str) -> dict:
    """从数据集文件夹加载患者数据"""
    dataset_path = Path(__file__).parent.parent.parent / "dataset" / folder_name
    json_file = dataset_path / f"{folder_name}.json"
    
    if not json_file.exists():
        print(f"文件不存在: {json_file}")
        return None
    
    with open(json_file, "r", encoding="utf-8") as f:
        return json.load(f)


async def seed_patients():
    """主函数：导入虚拟患者数据"""
    engine = create_async_engine(settings.DATABASE_URL, echo=False)
    async_session = sessionmaker(engine, class_=AsyncSession, expire_on_commit=False)
    
    async with async_session() as session:
        # 检查现有患者数量
        result = await session.execute(select(func.count(VirtualPatient.id)))
        existing_count = result.scalar()
        print(f"当前数据库中已有 {existing_count} 个虚拟患者")
        
        if existing_count >= 10:
            print("虚拟患者数量已达到10个，无需添加更多")
            return
        
        # 计算需要添加的数量
        needed = 10 - existing_count
        print(f"需要添加 {needed} 个虚拟患者")
        
        added = 0
        for folder_name, target_personality, difficulty in PATIENT_SAMPLES:
            if added >= needed:
                break
            
            patient_data = await load_patient_data(folder_name)
            if not patient_data:
                continue
            
            basic_info = patient_data.get("基础信息", {})
            personality = patient_data.get("人格", {})
            medical_record = patient_data.get("门诊病历", {})
            
            # 提取字段
            name = basic_info.get("姓名", f"患者{added+1}")
            age_str = basic_info.get("年龄", "30")
            age = int(age_str) if age_str.isdigit() else 30
            gender_raw = basic_info.get("性别", "男")
            gender = "female" if gender_raw == "女" else "male"
            
            # 使用目标人格类型（覆盖数据集中的原始类型）
            personality_type = target_personality
            
            chief_complaint = medical_record.get("主诉", "")
            medical_history = medical_record.get("既往史", "既往体健")
            current_illness = medical_record.get("现病史", "")
            expected_diagnosis = patient_data.get("主诊断", "")
            
            # 生成系统提示词
            system_prompt = generate_system_prompt(patient_data, personality_type)
            
            # 创建患者记录
            new_patient = VirtualPatient(
                name=name,
                age=age,
                gender=gender,
                personality_type=personality_type,
                chief_complaint=chief_complaint[:200] if chief_complaint else "门诊就诊",
                medical_history=medical_history or "既往体健",
                symptoms=current_illness or chief_complaint or "详见问诊",
                expected_diagnosis=expected_diagnosis[:200] if expected_diagnosis else "",
                system_prompt=system_prompt,
                difficulty_level=difficulty,
            )
            
            session.add(new_patient)
            added += 1
            print(f"添加患者 {added}: {name}, {age}岁, {gender}, {personality_type}, 难度{difficulty}")
        
        await session.commit()
        print(f"\n成功添加 {added} 个虚拟患者")
        
        # 验证总数
        result = await session.execute(select(func.count(VirtualPatient.id)))
        total = result.scalar()
        print(f"数据库中虚拟患者总数: {total}")
    
    await engine.dispose()


if __name__ == "__main__":
    asyncio.run(seed_patients())
