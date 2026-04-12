# -*- coding: utf-8 -*-
"""医学知识核对智能体 — 基于 RAG 检索增强与一致性评估的医学合理性评估"""

import json
import re
import logging
from app.services.qwen_client import call_qwen_chat
from app.services.rag.retriever import retrieve_medical_evidence, format_evidence_for_verification

# ── System Prompt ──
SYSTEM_PROMPT = """你是一名医学知识核对专家。你的任务是对比医生的诊断和治疗方案与检索到的临床指南证据，评估其一致性。

评估标准：
1. 诊断一致性：医生的诊断是否与医学证据支持的诊断方向一致
2. 治疗合理性：治疗方案是否符合临床指南推荐的标准治疗方案
3. 禁忌症检查：治疗方案中是否存在医学证据明确指出的禁忌或不当之处
4. 遗漏检查：是否遗漏了医学证据建议的必要检查或评估

输出格式（严格JSON）：
{
  "consistency": true/false,  // 诊断和治疗方案与医学证据是否一致
  "confidence": 0.85,         // 置信度 0-1，表示判断的确定程度
  "evidence": "用于核对的核心医学证据内容摘要，包括相关指南的关键推荐内容"
}

注意：
- consistency 为 true 表示诊断和治疗方案与医学证据基本一致
- consistency 为 false 表示存在明显不一致或不当之处
- confidence 表示你对判断的确定程度，证据越充分、判断越明确则 confidence 越高
- evidence 字段应简洁概括用于评估的核心医学证据要点
- 输出纯JSON，不要包含任何markdown格式或额外说明"""

# ── Few-shot 示例 ──
FEWSHOT_USER_1 = """【患者信息】
姓名: 张xx, 年龄: 65, 性别: male
主诉: 咳嗽、咳痰2周，痰中带血3天
病史: 吸烟史40年，每天20支

【问诊对话记录】
医生: 咳嗽多长时间了？
患者: 有两周了，最近3天痰里还有血。
医生: 有胸痛或呼吸困难吗？
患者: 有点胸闷，但不太疼。
医生: 吸烟吗？
患者: 吸了40年了，一天一包。

【医生诊断】
肺癌（疑似中央型肺癌）

【治疗方案】
1. 胸部CT增强扫描
2. 支气管镜检查+活检
3. 根据病理结果制定后续治疗方案（手术/化疗/放疗）

【检索到的医学证据】
1. NCCN非小细胞肺癌指南2025：对于疑似肺癌患者，推荐进行胸部CT增强扫描以评估肿瘤位置和分期。对于中央型病变，支气管镜检查是获取病理诊断的首选方法。
2. CSCO肺癌诊疗指南2025：痰中带血、长期吸烟史是肺癌的高危因素。推荐对可疑病灶进行组织病理学确诊后再制定治疗方案。
3. 临床指南：肺癌诊断需要病理确诊，影像学检查只能作为辅助诊断依据。治疗前应明确病理类型和分期。"""

FEWSHOT_ASSISTANT_1 = """{
  "consistency": true,
  "confidence": 0.92,
  "evidence": "NCCN和CSCO指南均推荐：1）对疑似肺癌患者进行胸部CT增强扫描评估；2）中央型病变首选支气管镜活检获取病理诊断；3）强调病理确诊是制定治疗方案的前提。医生的诊断思路符合指南推荐的诊断流程。"
}"""

FEWSHOT_USER_2 = """【患者信息】
姓名: 李xx, 年龄: 45, 性别: female
主诉: 上腹痛1周，反酸、烧心
病史: 既往体健，无特殊病史

【问诊对话记录】
医生: 腹痛具体位置在哪？
患者: 就在心口窝这，有时候反酸水。
医生: 吃饭后加重还是空腹时重？
患者: 吃完饭更明显。
医生: 有恶心呕吐吗？
患者: 没有。

【医生诊断】
急性胃炎

【治疗方案】
1. 奥美拉唑 20mg bid
2. 铝碳酸镁片 1g tid
3. 多潘立酮 10mg tid（促进胃动力）

【检索到的医学证据】
1. CSCO胃癌诊疗指南2025：对于新发上腹痛的45岁以上患者，应警惕胃癌可能，建议胃镜检查排除恶性病变。单纯按胃炎治疗可能延误诊断。
2. 中国胃食管反流病专家共识：反酸、烧心、餐后加重的上腹痛是GERD的典型表现，PPI治疗有效，但需排除器质性病变。
3. 临床指南：45岁以上新发消化不良症状患者，建议胃镜检查以排除胃癌等器质性病变，尤其对于有报警症状（体重下降、贫血、吞咽困难等）的患者。"""

FEWSHOT_ASSISTANT_2 = """{
  "consistency": false,
  "confidence": 0.85,
  "evidence": "CSCO胃癌诊疗指南明确指出：45岁以上新发上腹痛患者应警惕胃癌可能，建议胃镜检查排除恶性病变。该患者45岁，出现新发上腹痛症状，医生直接诊断为急性胃炎并开始药物治疗，未建议胃镜检查排除器质性病变，存在漏诊风险，不符合指南推荐的诊疗流程。"
}"""


# ── Helper Functions ──

def _extract_json(text: str) -> dict:
    """从 LLM 返回的文本中提取 JSON"""
    if not text or not text.strip():
        raise ValueError("LLM 返回内容为空")
    
    # 1. 尝试直接解析
    try:
        return json.loads(text.strip())
    except (json.JSONDecodeError, ValueError):
        pass
    
    # 2. 尝试移除 markdown 代码块后解析
    cleaned = re.sub(r"```(?:json)?\s*", "", text).strip().rstrip("`")
    try:
        return json.loads(cleaned)
    except (json.JSONDecodeError, ValueError):
        pass
    
    # 3. 尝试正则提取第一个 JSON 对象
    try:
        match = re.search(r"(\{.*\})", cleaned, re.DOTALL)
        if match:
            return json.loads(match.group(1))
    except (json.JSONDecodeError, AttributeError):
        pass
    
    raise ValueError(f"无法解析 JSON: {text[:200]}...")


def _calculate_score(consistency: bool, confidence: float) -> int:
    """
    基于一致性和置信度计算评分
    
    评分规则：
    - 一致时：Score = confidence * 100（置信度越高分越高）
    - 不一致时：Score = (1 - confidence) * 100（置信度越高分越低）
    """
    confidence = max(0.0, min(1.0, confidence))
    if consistency:
        score = confidence * 100
    else:
        score = (1 - confidence) * 100
    return int(round(max(0.0, min(100.0, score))))


def _generate_analysis(
    consistency: bool,
    confidence: float,
    evidence: str,
    doctor_diagnosis: str,
    treatment_plan: str
) -> str:
    """生成详细的分析文本（150-300字）"""
    
    # 一致性判断描述
    if consistency:
        consistency_desc = "诊断和治疗方案与医学证据基本一致"
        quality_desc = "医学知识运用合理"
    else:
        consistency_desc = "诊断和治疗方案与医学证据存在不一致"
        quality_desc = "存在改进空间"
    
    # 置信度描述
    if confidence >= 0.8:
        confidence_desc = "判断置信度高"
    elif confidence >= 0.6:
        confidence_desc = "判断置信度中等"
    else:
        confidence_desc = "判断置信度较低，建议进一步核实"
    
    # 构建分析文本
    analysis_parts = [
        f"医学知识核对结果：{consistency_desc}，{quality_desc}。",
        f"评估置信度为{confidence*100:.0f}%，{confidence_desc}。",
        f"核心医学证据：{evidence}",
    ]
    
    # 添加诊断和治疗方案概述
    if doctor_diagnosis:
        analysis_parts.append(f"医生诊断：{doctor_diagnosis[:50]}{'...' if len(doctor_diagnosis) > 50 else ''}")
    
    analysis = " ".join(analysis_parts)
    
    # 确保长度在 150-300 字之间
    if len(analysis) < 150:
        # 补充说明
        analysis += " 建议医生在后续诊疗中持续关注指南更新，确保诊疗方案符合最新的循证医学证据。"
    
    return analysis[:300] if len(analysis) > 300 else analysis


# ── Main Function ──

async def run_knowledge_check(
    conversation_text: str,
    patient_info: str,
    doctor_diagnosis: str,
    treatment_plan: str,
) -> dict:
    """
    基于 RAG 检索增强与一致性评估的医学知识核对
    
    评估流程：
    1. RAG 检索：从医学知识库检索相关临床指南证据
    2. 一致性评估：LLM 评估诊断/治疗方案与医学证据的一致性
    3. 代码计算：基于一致性和置信度计算最终评分
    4. 生成分析：代码生成详细的分析文本
    
    Args:
        conversation_text: 问诊对话记录
        patient_info: 患者基本信息
        doctor_diagnosis: 医生提交的诊断
        treatment_plan: 医生提交的治疗方案
    
    Returns:
        dict: {"raw_response": json.dumps({"score": int, "analysis": str, "details": {...}})}
    """
    
    # ── Step 1: RAG 检索增强 ──
    evidence_text = ""
    rag_success = False
    try:
        # 基于诊断和治疗方案检索医学证据
        query = f"{doctor_diagnosis} {treatment_plan}".strip()
        if query:
            evidences = await retrieve_medical_evidence(query, top_k=5)
            evidence_text = format_evidence_for_verification(evidences)
            rag_success = True
    except Exception as e:
        logging.error(f"RAG 检索失败: {e}")
        evidence_text = "未检索到医学证据"
    
    # ── Step 2: LLM 一致性评估 ──
    try:
        # 构建用户输入内容
        user_content_parts = [
            f"【患者信息】\n{patient_info}\n",
            f"【问诊对话记录】\n{conversation_text}\n",
            f"【医生诊断】\n{doctor_diagnosis}\n",
            f"【治疗方案】\n{treatment_plan}\n",
        ]
        
        if evidence_text:
            user_content_parts.append(f"【检索到的医学证据】\n{evidence_text}")
        
        user_content = "\n".join(user_content_parts)
        
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": FEWSHOT_USER_1},
            {"role": "assistant", "content": FEWSHOT_ASSISTANT_1},
            {"role": "user", "content": FEWSHOT_USER_2},
            {"role": "assistant", "content": FEWSHOT_ASSISTANT_2},
            {"role": "user", "content": user_content},
        ]
        
        result = await call_qwen_chat(messages, temperature=0.2)
        
        # 解析 LLM 输出
        consistency_data = _extract_json(result)
        
        consistency = consistency_data.get("consistency", False)
        confidence = float(consistency_data.get("confidence", 0.5))
        evidence = consistency_data.get("evidence", "未提供证据摘要")
        
    except Exception as e:
        logging.error(f"一致性评估 LLM 调用失败: {e}")
        # 错误降级：返回默认中等分数
        return {
            "raw_response": json.dumps({
                "score": 50,
                "analysis": "医学知识核对过程中遇到技术问题，无法完成评估。默认给予中等分数，建议人工复核。",
                "details": {
                    "consistency": None,
                    "confidence": 0.5,
                    "error": str(e),
                    "rag_success": rag_success,
                }
            }, ensure_ascii=False)
        }
    
    # ── Step 3: 代码计算评分 ──
    final_score = _calculate_score(consistency, confidence)
    
    # ── Step 4: 生成分析文本 ──
    analysis = _generate_analysis(
        consistency=consistency,
        confidence=confidence,
        evidence=evidence,
        doctor_diagnosis=doctor_diagnosis,
        treatment_plan=treatment_plan,
    )
    
    # 构建返回结果
    result = {
        "score": final_score,
        "analysis": analysis,
        "details": {
            "consistency": consistency,
            "confidence": round(confidence, 2),
            "evidence_summary": evidence,
            "rag_success": rag_success,
        }
    }
    
    return {"raw_response": json.dumps(result, ensure_ascii=False)}
