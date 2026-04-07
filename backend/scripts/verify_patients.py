# -*- coding: utf-8 -*-
"""验证数据集和数据库中患者数据的一致性"""

import asyncio
import json
from pathlib import Path
import sys

# 设置终端输出编码为UTF-8，避免Windows终端乱码
sys.stdout.reconfigure(encoding='utf-8')

sys.path.insert(0, str(Path(__file__).parent.parent))

from sqlalchemy import select
from sqlalchemy.ext.asyncio import create_async_engine, AsyncSession
from sqlalchemy.orm import sessionmaker

from app.core.config import settings
from app.models.patient import VirtualPatient


async def verify_data():
    """验证数据集和数据库中的主诉、预期诊断字段一致性"""
    
    # 数据集文件夹和对应的数据库记录映射
    datasets = [
        ('patient100_21', 9),
        ('patient110_23', 10),
        ('patient120_25', 11),
        ('patient130_27', 12),
        ('patient140_32', 13),
        ('patient50_13', 14),
    ]
    
    engine = create_async_engine(settings.DATABASE_URL)
    async_session = sessionmaker(engine, class_=AsyncSession, expire_on_commit=False)
    
    print("=" * 70)
    print("验证数据集与数据库的一致性")
    print("=" * 70)
    
    all_pass = True
    
    async with async_session() as session:
        for folder, expected_id in datasets:
            # 读取数据集
            json_path = Path(__file__).parent.parent.parent / "dataset" / folder / f"{folder}.json"
            if not json_path.exists():
                print(f"[跳过] {folder}: 文件不存在")
                continue
            
            with open(json_path, "r", encoding="utf-8") as f:
                dataset = json.load(f)
            
            ds_name = dataset.get("基础信息", {}).get("姓名", "N/A")
            ds_chief = dataset.get("门诊病历", {}).get("主诉", "")
            ds_diag = dataset.get("主诊断", "")
            
            # 读取数据库
            result = await session.execute(
                select(VirtualPatient).where(VirtualPatient.id == expected_id)
            )
            patient = result.scalar_one_or_none()
            
            if not patient:
                print(f"[失败] {folder}: 数据库中未找到 ID={expected_id}")
                all_pass = False
                continue
            
            print(f"\n【{folder}】ID={expected_id}")
            print(f"  姓名: 数据集[{ds_name}] -> 数据库[{patient.name}]")
            
            # 验证主诉
            chief_match = ds_chief == patient.chief_complaint
            chief_status = "[OK]" if chief_match else "[FAIL]"
            print(f"  Chief Complaint: {chief_status}")
            print(f"    Dataset: [{ds_chief}]")
            print(f"    Database: [{patient.chief_complaint}]")
            if not chief_match:
                all_pass = False
            
            # 验证预期诊断
            diag_match = ds_diag == patient.expected_diagnosis
            diag_status = "[OK]" if diag_match else "[FAIL]"
            print(f"  Expected Diagnosis: {diag_status}")
            print(f"    Dataset: [{ds_diag}]")
            print(f"    Database: [{patient.expected_diagnosis}]")
            if not diag_match:
                all_pass = False
    
    await engine.dispose()
    
    print("\n" + "=" * 70)
    if all_pass:
        print("Result: ALL FIELDS MATCH")
    else:
        print("Result: SOME FIELDS DO NOT MATCH")
    print("=" * 70)
    
    return all_pass


if __name__ == "__main__":
    result = asyncio.run(verify_data())
    sys.exit(0 if result else 1)
