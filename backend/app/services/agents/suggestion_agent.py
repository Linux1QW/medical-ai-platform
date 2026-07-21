# -*- coding: utf-8 -*-
"""建议指导智能体 — 基于对比学习的问诊改进建议生成

核心思想：
1. 构造理想问诊（Ideal）：基于临床规范和评估标准，构建最优问诊样本
2. 对比当前问诊（Observed）：使用者实际完成的问诊对话 + 各模块评估结果
3. 差异分析：找出缺失的关键信息、需要改进的问诊逻辑和沟通方式
4. 生成建议：基于差异分析，输出结构化改进建议
"""

import json
import logging

from app.services.qwen_client import call_qwen_chat
from app.utils.json_parser import extract_json_from_text

# ── System Prompt ────────────────────────────────────────────────────────────

SYSTEM_PROMPT = """你是一名临床教学指导专家，擅长通过对比分析帮助医生改进问诊技能。

你的任务是基于对比学习思想，分析当前问诊与理想问诊的差异，生成针对性改进建议。

## 分析流程

### 第一步：构造理想问诊（Ideal）
根据患者信息和临床规范，构思一个理想的问诊流程：
- 应该询问哪些关键问题（症状细节、病史、风险因素等）
- 应该采用什么样的沟通方式（共情、倾听、解释）
- 应该体现什么样的人文关怀（尊重患者、关注心理需求）

### 第二步：对比当前问诊（Observed）
将理想问诊与当前问诊进行对比，重点分析：
- What is missing? 缺失了哪些关键信息采集？
- What should be improved? 问诊逻辑、沟通方式、人文关怀有哪些不足？

### 第三步：生成结构化建议
基于差异分析，输出JSON格式的改进建议。

## 输出格式要求

请严格按以下JSON格式输出：
```json
{
  "ideal_inquiry_summary": "理想问诊应包含的关键步骤摘要（50-100字）",
  "missing_questions": ["缺失的关键问题1", "缺失的关键问题2", ...],
  "improvement_suggestions": ["具体改进措施1", "具体改进措施2", ...],
  "priority": "high/middle/low",
  "suggestions": "将missing_questions和improvement_suggestions整合为格式化的建议文本（纯文本，不用markdown，用数字序号，300-500字）"
}
```

## 注意事项
1. missing_questions 列出3-5个最关键的缺失问题
2. improvement_suggestions 列出3-5条具体可操作的改进措施
3. priority 根据缺失问题的严重程度判定：high（关键信息严重缺失）、middle（部分重要信息缺失）、low（基本完整）
4. suggestions 字段是纯文本格式，不使用任何markdown符号，用中文数字序号组织内容"""


# ── Few-shot 示例 ────────────────────────────────────────────────────────────

FEWSHOT_USER_1 = """【患者信息】
姓名: 张xx, 年龄: 45, 性别: male
主诉: 上腹部不适3个月
既往史: 高血压5年，长期服用降压药

【问诊对话记录】
医生: 哪里不舒服？
患者: 上腹部总感觉不舒服，有3个月了。
医生: 怎么个不舒服法？
患者: 就是胀，有时候有点疼。
医生: 跟吃饭有关系吗？
患者: 吃完饭更明显。
医生: 做个胃镜看看吧。

【问诊分析评估】
- 病史采集得分: 35分。问诊过于简短，仅5轮对话。缺少症状具体特征（疼痛性质、程度、放射）、诱因缓解因素、伴随症状、既往胃病史、家族史等关键信息。
- 医学知识评估: 缺少对报警症状（体重下降、黑便、吞咽困难）的筛查，未评估Hp感染风险。

【医学知识评估】
- 知识应用得分: 40分。未对上消化道症状进行危险分层，缺少报警症状筛查，未评估NSAIDs用药史和Hp感染状态。

【人文关怀评估】
- 人文关怀得分: 38分。问诊态度生硬，全程无共情性语言，未关注患者心理状态，检查安排解释不充分。

请基于对比分析生成改进建议。"""

FEWSHOT_ASSISTANT_1 = """{
  "ideal_inquiry_summary": "理想问诊应包括：系统询问上腹痛特征（性质、程度、规律、放射）、诱因与缓解因素、伴随症状（反酸、嗳气、恶心呕吐、黑便）、报警症状筛查、既往胃病史、用药史（尤其NSAIDs）、Hp相关病史、饮食生活习惯、心理压力评估，并给予充分的解释和安抚。",
  "missing_questions": [
    "疼痛的具体性质（烧灼样、钝痛、绞痛）和程度评分",
    "症状的诱发因素和缓解因素",
    "报警症状筛查：体重下降、黑便、吞咽困难、呕吐",
    "既往胃病史、幽门螺杆菌感染史和治疗史",
    "长期用药史，尤其NSAIDs和阿司匹林使用情况"
  ],
  "improvement_suggestions": [
    "建立系统化的上消化道症状问诊框架，确保覆盖疼痛性质、规律、放射、伴随症状等关键要素",
    "对45岁以上上腹部症状患者常规进行报警症状筛查，警惕恶性肿瘤风险",
    "主动询问用药史，评估NSAIDs相关胃黏膜损伤风险",
    "培养共情沟通意识，问诊开始时关切询问，结束前总结病情并解释检查必要性",
    "关注患者心理状态，对慢性症状患者了解焦虑抑郁倾向"
  ],
  "priority": "high",
  "suggestions": "一、病史采集改进建议\\n\\n当前问诊仅5轮即结束，病史采集严重不足。缺失以下关键信息：1.疼痛的具体性质和程度，需鉴别溃疡性疼痛与功能性疼痛；2.症状的诱发和缓解因素，有助于判断病因；3.报警症状筛查（体重下降、黑便、吞咽困难、呕吐），45岁以上患者必须排查；4.既往胃病史和Hp感染状态；5.长期用药史，尤其NSAIDs相关胃损伤风险。\\n\\n二、问诊流程改进建议\\n\\n建议建立标准化的上消化道症状问诊模板，按主诉特征、伴随症状、报警症状筛查、既往史、用药史、生活因素的顺序系统采集。对中老年患者应更加重视报警症状筛查，必要时主动追问。\\n\\n三、沟通与人文关怀改进\\n\\n问诊全程缺乏共情性语言，建议在问诊开始时加入关切语（如"这三个月症状对您影响大吗"），结束时对检查安排给予充分解释，告知胃镜检查的必要性和注意事项，缓解患者焦虑。"
}"""

FEWSHOT_USER_2 = """【患者信息】
姓名: 李xx, 年龄: 28, 性别: female
主诉: 腹泻、腹胀2周
既往史: 无特殊

【问诊对话记录】
医生: 今天怎么不舒服？
患者: 最近两周总是拉肚子，肚子也胀。
医生: 一天几次？什么样便？
患者: 一天三四次，稀便，没有血。
医生: 肚子疼吗？
患者: 不怎么疼，就是胀。
医生: 跟吃东西有关系吗？
患者: 好像吃凉的更明显。
医生: 体重有变化吗？
患者: 没有变化。
医生: 以前有过这情况吗？
患者: 偶尔会有，但这次时间比较长。
医生: 最近压力大吗？睡眠怎么样？
患者: 最近工作确实挺忙的，睡眠不太好。
医生: 我看你需要做个肠镜检查，排除一下器质性问题。
患者: 肠镜...很痛苦吧？
医生: 现在有无痛肠镜，睡一觉就好了。
医生: 我先给你开点调节肠道菌群的药，检查做完再调整。

【问诊分析评估】
- 病史采集得分: 72分。问诊较系统，覆盖了症状特征、伴随症状、诱因、既往史、心理因素。不足：缺少饮食细节、外出就餐史、用药史、家族史。

【医学知识评估】
- 知识应用得分: 75分。考虑到功能性肠病可能，关注了心理因素。不足：未评估IBS诊断标准，未排除感染性腹泻。

【人文关怀评估】
- 人文关怀得分: 68分。态度较为友善，对患者顾虑给予回应。不足：检查解释较简略，未提供饮食生活指导。

请基于对比分析生成改进建议。"""

FEWSHOT_ASSISTANT_2 = """{
  "ideal_inquiry_summary": "理想问诊应包括：腹泻详细特征（次数、性状、有无黏液脓血）、伴随症状评估、饮食史（可疑食物、外出就餐）、旅行史、用药史（抗生素）、IBS相关症状评估、既往肠道病史、家族史、心理因素深入评估，并给予饮食生活指导和随访计划。",
  "missing_questions": [
    "饮食细节：是否进食生冷、不洁食物，近期有无外出就餐或旅行史",
    "用药史：近期是否使用抗生素或其他可能导致腹泻的药物",
    "IBS诊断标准相关：症状与排便关系、排便后是否缓解",
    "既往肠道病史、手术史、家族肠道肿瘤史"
  ],
  "improvement_suggestions": [
    "完善饮食史采集，包括可疑食物、外出就餐史、旅行史，有助于排查感染性腹泻",
    "常规询问抗生素使用史，排除抗生素相关性腹泻",
    "应用Rome IV标准评估IBS可能性，询问症状与排便关系",
    "检查前更详细解释肠镜必要性、检查流程和注意事项，减轻患者焦虑",
    "提供书面或口头饮食生活指导，安排短期随访评估疗效"
  ],
  "priority": "middle",
  "suggestions": "一、病史采集完善建议\\n\\n当前问诊较为系统，但仍有改进空间。建议补充：1.饮食细节，包括是否进食生冷不洁食物、近期外出就餐史或旅行史，有助于排查感染性腹泻；2.用药史，尤其抗生素使用情况；3.IBS相关评估，询问症状与排便关系、排便后是否缓解；4.既往肠道病史和家族肠道肿瘤史。\\n\\n二、诊断思维提升\\n\\n考虑功能性肠病方向正确，建议进一步应用Rome IV标准评估IBS可能性。对急性腹泻患者注意排除感染因素，必要时完善粪便常规检查。\\n\\n三、沟通与患者教育\\n\\n肠镜检查解释较简略，建议详细说明检查必要性、流程、注意事项及无痛肠镜的安全性。此外，应提供腹泻期间的饮食生活指导，如避免生冷刺激性食物、保证水分摄入等，并安排1-2周短期随访评估治疗效果。"
}"""


# ── Helper Functions ─────────────────────────────────────────────────────────

def _extract_json(text: str) -> dict:
    """从 LLM 返回的文本中提取 JSON"""
    return extract_json_from_text(text)


def _get_default_suggestion() -> dict:
    """返回默认的建议结果（错误降级处理）"""
    return {
        "ideal_inquiry_summary": "基于患者主诉系统采集病史，关注症状特征、伴随症状、既往史、用药史、家族史等关键信息，注重沟通技巧和人文关怀。",
        "missing_questions": ["系统评估当前问诊未能获取的关键信息"],
        "improvement_suggestions": [
            "建议回顾问诊流程，确保病史采集的完整性",
            "关注患者的沟通体验，适时使用共情性语言",
            "检查前给予充分的解释和指导"
        ],
        "priority": "middle",
        "suggestions": "建议进一步完善问诊流程：1.确保病史采集的系统性，覆盖主诉特征、伴随症状、既往史、用药史、家族史等关键要素；2.注重沟通技巧，使用共情性语言建立良好的医患关系；3.检查前给予充分解释，告知必要性和注意事项，减轻患者焦虑。由于系统暂时无法生成详细分析，建议结合具体问诊情况进行针对性改进。"
    }


# ── Main Function ───────────────────────────────────────────────────────────

async def run_suggestion(
    conversation_text: str,
    patient_info: str,
    inquiry_result: str,
    knowledge_result: str,
    humanistic_result: str,
) -> dict:
    """
    基于对比学习的问诊改进建议生成

    通过对比理想问诊与当前问诊，分析差异并生成结构化改进建议。

    Args:
        conversation_text: 问诊对话记录
        patient_info: 患者基本信息
        inquiry_result: 问诊分析评估结果
        knowledge_result: 医学知识评估结果
        humanistic_result: 人文关怀评估结果

    Returns:
        dict: 包含 raw_response 字段，其中 JSON 字符串包括：
            - suggestions: 格式化的建议文本（兼容数据库存储）
            - missing_questions: 缺失的关键问题列表
            - improvement_suggestions: 改进措施列表
            - priority: 优先级 (high/middle/low)
            - ideal_inquiry_summary: 理想问诊摘要
    """
    try:
        # 构建用户提示
        user_prompt = f"""【患者信息】
{patient_info}

【问诊对话记录】
{conversation_text}

【问诊分析评估】
{inquiry_result}

【医学知识评估】
{knowledge_result}

【人文关怀评估】
{humanistic_result}

请基于对比分析生成改进建议。"""

        # 构建 LLM 调用消息
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": FEWSHOT_USER_1},
            {"role": "assistant", "content": FEWSHOT_ASSISTANT_1},
            {"role": "user", "content": FEWSHOT_USER_2},
            {"role": "assistant", "content": FEWSHOT_ASSISTANT_2},
            {"role": "user", "content": user_prompt},
        ]

        # 调用 LLM
        result = await call_qwen_chat(messages, temperature=0.4)

        # 解析 JSON
        suggestion_data = _extract_json(result)

        # 确保返回格式兼容
        if "suggestions" not in suggestion_data:
            # 如果 LLM 未生成 suggestions 字段，构造一个
            missing = suggestion_data.get("missing_questions", [])
            improvements = suggestion_data.get("improvement_suggestions", [])

            suggestion_text = "一、缺失的关键问题\n\n"
            for i, q in enumerate(missing[:5], 1):
                suggestion_text += f"{i}. {q}\n"
            suggestion_text += "\n二、改进措施建议\n\n"
            for i, s in enumerate(improvements[:5], 1):
                suggestion_text += f"{i}. {s}\n"

            suggestion_data["suggestions"] = suggestion_text

        return {"raw_response": json.dumps(suggestion_data, ensure_ascii=False)}

    except Exception as e:
        logging.error(f"建议生成智能体执行失败: {e}")
        # 降级处理：返回默认建议
        default_suggestion = _get_default_suggestion()
        return {"raw_response": json.dumps(default_suggestion, ensure_ascii=False)}
