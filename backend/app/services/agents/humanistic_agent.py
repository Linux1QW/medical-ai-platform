# -*- coding: utf-8 -*-
"""人文关怀评估智能体 — 基于结构化建模与可计算指标的医患沟通评估"""

import json
import re
import logging
from app.services.qwen_client import call_qwen_chat

# ── 权重配置 ──
WEIGHTS = {
    "empathy": 0.6,    # 文本共情评估权重
    "behavior": 0.4,   # 对话行为分析权重
}

# ── 对话行为类型定义 ──
BEHAVIOR_TYPES = ["comfort", "explain", "instruction", "ignore"]

# 行为权重系数
BEHAVIOR_WEIGHTS = {
    "comfort": 1.0,      # 安慰 - 正面行为
    "explain": 0.8,      # 解释 - 正面行为
    "instruction": 0.5,  # 指令 - 中性行为
    "ignore": 0.0,       # 忽视 - 负面行为
}

# ── LLM Prompts ──

# 第一次 LLM 调用：文本共情评估
EMPATHY_SYSTEM_PROMPT = """你是一名医学人文关怀与医患沟通评估专家。请分析医生在问诊对话中的沟通表现，从三个子维度进行评分。

评估维度（每个维度 0-10 分）：
1. empathy（共情）：医生对患者情绪的理解与回应程度
   - 10分：充分理解患者情绪，使用共情性语言，给予情感支持
   - 5分：有一定情绪关注，但回应不够深入或及时
   - 0分：完全忽视患者情绪，态度冷漠

2. politeness（礼貌）：医生语言表达的得体性与尊重程度
   - 10分：语言礼貌得体，尊重患者，无打断、无命令式语气
   - 5分：基本礼貌，但偶有生硬表达
   - 0分：语言粗鲁，态度傲慢，频繁打断患者

3. clarity（表达清晰）：医生问诊问题、解释内容的易懂程度
   - 10分：语言通俗易懂，解释清晰，患者易于理解
   - 5分：基本可理解，但偶有专业术语未解释
   - 0分：表达混乱，专业术语堆砌，患者难以理解

输出格式（严格JSON）：
{
  "empathy": 0-10的整数,
  "politeness": 0-10的整数,
  "clarity": 0-10的整数
}

注意：
- 每个维度给出 0-10 的整数评分
- 输出纯JSON，不要包含任何markdown格式或额外说明"""

EMPATHY_FEWSHOT_USER = """【患者信息】
姓名: 姚xx, 年龄: 64, 性别: female
人格类型: 对抗型
主诉: 右下腹疼痛
病史: 既往体健
症状: 右下腹疼痛，大便不成型，持续一个多月
预期诊断: 腹痛

【问诊对话记录】
医生: 怎么了？为啥要看胃肠科呀？
患者: 这是咋的呀，我就胃疼，就这地方唉呀我天一个多月了
医生: 哪块疼？还是不舒服，
患者: 不是不舒服，
医生: 比一下位置
患者: 唉就这就是肚脐子嘛，就这块疼啊，就这肚脐下面.
医生: 做没做过检查呢？
患者: 做过.
医生: 做啥了？
患者: 这两天呢昨天开始这块疼着.
医生: 查啥了
患者: 完还那啥.
医生: 查啥了
患者: 查了肠镜，胃镜，已经都早都做过了
医生: 胃肠镜在哪做的？
患者: 在那个胃肠镜和那什么肠镜。
医生: 我要看报告单，不看片子
患者: 噢
医生: 这个医院结果我们不信，建议在这重新查一个啊
患者: 嗯那我们不知道啊
医生: 就是他看了之后说没啥大事，但是不一定看得全，不一定看得准，我觉得你最好再复查一下，如果实在不想复查的话，咱们就按照他报那个炎症给你开点药吃上看看
患者: 行
医生: 大便几次，一天得好几次是吗？
患者: 没有就一次。
医生: 行，先给你开一周药先吃上看看啊
患者: 嗯那也行
医生: 这样吧我感觉你还是太焦虑了，你应该加点焦虑的药啊。
患者: 啊行。

请对医生的文本共情表现进行评分。"""

EMPATHY_FEWSHOT_ASSISTANT = """{
  "empathy": 2,
  "politeness": 4,
  "clarity": 5
}"""

# 第二次 LLM 调用：对话行为分类
BEHAVIOR_SYSTEM_PROMPT = """你是一名医患沟通行为分析专家。请将医生在问诊过程中的每句发言分类为四种对话行为之一。

对话行为类型定义：
1. comfort（安慰）：对患者进行情绪安抚、鼓励的发言
   - 示例："别担心，这个检查结果很好"、"您的情况会好起来的"

2. explain（解释）：向患者解释病情、检查目的、治疗方案的发言
   - 示例："这个检查是为了排除肿瘤"、"萎缩性胃炎是胃黏膜退化"

3. instruction（指令）：向患者下达检查、用药等指令的发言
   - 示例："去做个CT检查"、"每天三次，每次一片"

4. ignore（忽视）：未回应患者疑问、情绪的发言，或生硬转移话题
   - 示例：患者表达担忧后医生直接问下一个问题而无回应

输出格式（严格JSON）：
{
  "utterances": [
    {"text": "医生发言原文1", "behavior": "explain"},
    {"text": "医生发言原文2", "behavior": "instruction"},
    ...
  ]
}

注意：
- 只包含医生的发言，不包含患者的发言
- 每句医生发言必须分类为四种行为之一
- 输出纯JSON，不要包含任何markdown格式或额外说明"""

BEHAVIOR_FEWSHOT_USER = """【患者信息】
姓名: 宋xx, 年龄: 59, 性别: female
人格类型: 配合型
主诉: 间断反酸10年
病史: 既往体健
症状: 反酸烧心10年，偶有恶心呕吐
预期诊断: 慢性胃炎-萎缩性

【问诊对话记录】
医生: 你怎么不舒服
患者: 这食道就是总叫反酸，反完了就烧心的时候，就这里火燎燎的
医生: 最近反酸烧心的厉害吗？
患者: 还不算咋的，要是不干猫腰活还行的
医生: 那最近哪方面的症状加重了？想让你再来看病的
患者: 就是总叫往上反气打嗝
医生: 现在是这样，肝胆胰彩超没有啥问题，所以说不太考虑是肝脏胆囊胰腺出了问题。然后胃镜的结果是有萎缩性胃炎和一个糜烂的改变。萎缩性胃炎是啥呢？就是胃黏膜已经萎缩退化，被肠黏膜所覆盖了，是一个退行性的改变。萎缩之后会出现消化功能的减弱
患者: 哦
医生: 胃酸分泌消化蠕动代谢功能都会变差，吃东西可能都会变差。再有就是你有一些糜烂的面，糜烂的地方有一些可疑的，给你取了个病理看一下有没有不好的地方，这个需要等一下病理结果
患者: 哦
医生: 首先如果说单纯从萎缩和糜烂来说，需要查幽门螺旋杆菌，你以前查过吗？
患者: 没查
医生: 最近用过胃药和消炎药吗？
患者: 吃过奥美拉唑就是法莫替丁
医生: 你这类的药物建议停两周以上。消炎药吃没吃过？
患者: 没吃
医生: 法莫替丁、奥美拉唑上次吃距现在多长时间了？
患者: 得有十多天吧？
医生: 停药最好两周以上，你现在查可能是假阴性。回当地查都可以，建议查幽门螺杆菌，有菌要除菌。没有菌就只能吃好消化的，适当运动，用一些保护黏膜的药。
患者: 有的时候早上起来喝一口水都噎停
医生: 食管本身没问题，但萎缩后蠕动消化排空功能会变差。再有跟你情绪心理焦虑可能都有关，不知道你有没有这方面因素。适当运动，多想高兴的事，不要让情绪影响消化。没有啥需要特殊治疗，等一等病理。
患者: 好的

请对医生的每句发言进行行为分类。"""

BEHAVIOR_FEWSHOT_ASSISTANT = """{
  "utterances": [
    {"text": "你怎么不舒服", "behavior": "instruction"},
    {"text": "最近反酸烧心的厉害吗？", "behavior": "instruction"},
    {"text": "那最近哪方面的症状加重了？想让你再来看病的", "behavior": "instruction"},
    {"text": "现在是这样，肝胆胰彩超没有啥问题，所以说不太考虑是肝脏胆囊胰腺出了问题。然后胃镜的结果是有萎缩性胃炎和一个糜烂的改变。萎缩性胃炎是啥呢？就是胃黏膜已经萎缩退化，被肠黏膜所覆盖了，是一个退行性的改变。萎缩之后会出现消化功能的减弱", "behavior": "explain"},
    {"text": "胃酸分泌消化蠕动代谢功能都会变差，吃东西可能都会变差。再有就是你有一些糜烂的面，糜烂的地方有一些可疑的，给你取了个病理看一下有没有不好的地方，这个需要等一下病理结果", "behavior": "explain"},
    {"text": "首先如果说单纯从萎缩和糜烂来说，需要查幽门螺旋杆菌，你以前查过吗？", "behavior": "instruction"},
    {"text": "最近用过胃药和消炎药吗？", "behavior": "instruction"},
    {"text": "你这类的药物建议停两周以上。消炎药吃没吃过？", "behavior": "instruction"},
    {"text": "法莫替丁、奥美拉唑上次吃距现在多长时间了？", "behavior": "instruction"},
    {"text": "停药最好两周以上，你现在查可能是假阴性。回当地查都可以，建议查幽门螺杆菌，有菌要除菌。没有菌就只能吃好消化的，适当运动，用一些保护黏膜的药。", "behavior": "explain"},
    {"text": "食管本身没问题，但萎缩后蠕动消化排空功能会变差。再有跟你情绪心理焦虑可能都有关，不知道你有没有这方面因素。适当运动，多想高兴的事，不要让情绪影响消化。没有啥需要特殊治疗，等一等病理。", "behavior": "explain"}
  ]
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


def _calculate_empathy_score(empathy_data: dict) -> float:
    """计算共情得分（归一化到 0-1）"""
    empathy = empathy_data.get("empathy", 5)
    politeness = empathy_data.get("politeness", 5)
    clarity = empathy_data.get("clarity", 5)
    
    # 确保分数在 0-10 范围内
    empathy = max(0, min(10, empathy))
    politeness = max(0, min(10, politeness))
    clarity = max(0, min(10, clarity))
    
    # 计算平均分并归一化到 0-1
    avg_score = (empathy + politeness + clarity) / 3
    return avg_score / 10


def _calculate_behavior_score(behavior_data: dict) -> float:
    """计算行为得分（0-1 范围）"""
    utterances = behavior_data.get("utterances", [])
    
    if not utterances:
        return 0.5  # 默认中等得分
    
    total_weight = 0
    for utterance in utterances:
        behavior_type = utterance.get("behavior", "ignore")
        weight = BEHAVIOR_WEIGHTS.get(behavior_type, 0.0)
        total_weight += weight
    
    return total_weight / len(utterances)


def _generate_analysis(empathy_score: float, behavior_score: float,
                       empathy_data: dict, behavior_data: dict) -> str:
    """生成详细的分析文本"""
    empathy = empathy_data.get("empathy", 5)
    politeness = empathy_data.get("politeness", 5)
    clarity = empathy_data.get("clarity", 5)
    
    utterances = behavior_data.get("utterances", [])
    
    # 统计行为类型
    behavior_counts = {"comfort": 0, "explain": 0, "instruction": 0, "ignore": 0}
    for utterance in utterances:
        behavior_type = utterance.get("behavior", "ignore")
        if behavior_type in behavior_counts:
            behavior_counts[behavior_type] += 1
    
    total_utterances = len(utterances)
    
    # 生成分析文本
    analysis_parts = []
    
    # 1. 共情维度分析
    empathy_desc = f"文本共情评估得分{empathy_score*100:.0f}分（权重60%）。"
    empathy_desc += f"三个子维度：共情{empathy}分"
    
    if empathy >= 8:
        empathy_desc += "（优秀，充分理解患者情绪）"
    elif empathy >= 5:
        empathy_desc += "（一般，有一定情绪关注）"
    else:
        empathy_desc += "（不足，缺乏情感回应）"
    
    empathy_desc += f"，礼貌{politeness}分"
    if politeness >= 8:
        empathy_desc += "（得体尊重）"
    elif politeness >= 5:
        empathy_desc += "（基本礼貌）"
    else:
        empathy_desc += "（生硬欠妥）"
    
    empathy_desc += f"，表达清晰{clarity}分"
    if clarity >= 8:
        empathy_desc += "（通俗易懂）"
    elif clarity >= 5:
        empathy_desc += "（基本可理解）"
    else:
        empathy_desc += "（表达混乱）"
    
    analysis_parts.append(empathy_desc)
    
    # 2. 行为维度分析
    behavior_desc = f"对话行为分析得分{behavior_score*100:.0f}分（权重40%）。"
    
    if total_utterances > 0:
        behavior_desc += f"共{total_utterances}句医生发言："
        behavior_desc += f"安慰{behavior_counts['comfort']}次、"
        behavior_desc += f"解释{behavior_counts['explain']}次、"
        behavior_desc += f"指令{behavior_counts['instruction']}次、"
        behavior_desc += f"忽视{behavior_counts['ignore']}次。"
        
        # 评价行为模式
        positive = behavior_counts['comfort'] + behavior_counts['explain']
        negative = behavior_counts['ignore']
        
        if positive > total_utterances * 0.6:
            behavior_desc += "正面沟通行为占比较高，体现较好的人文关怀。"
        elif negative > total_utterances * 0.3:
            behavior_desc += "存在较多忽视患者的情况，需加强回应意识。"
        else:
            behavior_desc += "沟通行为较为均衡。"
    else:
        behavior_desc += "未能提取到医生发言数据。"
    
    analysis_parts.append(behavior_desc)
    
    return "".join(analysis_parts)


# ── Main Function ──

async def run_humanistic_evaluation(conversation_text: str, patient_info: str) -> dict:
    """
    基于结构化建模与可计算指标的人文关怀评估
    
    评估维度：
    1. Empathy (60%): 文本共情评估（共情、礼貌、表达清晰三个子维度）
    2. Behavior (40%): 对话行为分析（安慰、解释、指令、忽视四类行为）
    """
    # 默认降级数据
    default_empathy_data = {"empathy": 5, "politeness": 5, "clarity": 5}
    default_behavior_data = {"utterances": []}
    
    try:
        # ── Step 1: 文本共情评估（第一次 LLM 调用）──
        empathy_messages = [
            {"role": "system", "content": EMPATHY_SYSTEM_PROMPT},
            {"role": "user", "content": EMPATHY_FEWSHOT_USER},
            {"role": "assistant", "content": EMPATHY_FEWSHOT_ASSISTANT},
            {
                "role": "user",
                "content": f"【患者信息】\n{patient_info}\n\n【问诊对话记录】\n{conversation_text}\n\n请对医生的文本共情表现进行评分。"
            },
        ]
        
        empathy_result = await call_qwen_chat(empathy_messages, temperature=0.2)
        empathy_data = _extract_json(empathy_result)
        
    except Exception as e:
        logging.error(f"共情评估 LLM 调用失败: {e}")
        empathy_data = default_empathy_data
    
    try:
        # ── Step 2: 对话行为分类（第二次 LLM 调用）──
        behavior_messages = [
            {"role": "system", "content": BEHAVIOR_SYSTEM_PROMPT},
            {"role": "user", "content": BEHAVIOR_FEWSHOT_USER},
            {"role": "assistant", "content": BEHAVIOR_FEWSHOT_ASSISTANT},
            {
                "role": "user",
                "content": f"【患者信息】\n{patient_info}\n\n【问诊对话记录】\n{conversation_text}\n\n请对医生的每句发言进行行为分类。"
            },
        ]
        
        behavior_result = await call_qwen_chat(behavior_messages, temperature=0.2)
        behavior_data = _extract_json(behavior_result)
        
    except Exception as e:
        logging.error(f"行为分类 LLM 调用失败: {e}")
        behavior_data = default_behavior_data
    
    # ── Step 3: 数学计算综合评分 ──
    try:
        empathy_score = _calculate_empathy_score(empathy_data)
        behavior_score = _calculate_behavior_score(behavior_data)
        analysis = _generate_analysis(empathy_score, behavior_score, empathy_data, behavior_data)
    except Exception as e:
        logging.error(f"评分或分析生成失败，使用默认数据降级: {e}")
        empathy_data = default_empathy_data
        behavior_data = default_behavior_data
        empathy_score = _calculate_empathy_score(empathy_data)
        behavior_score = _calculate_behavior_score(behavior_data)
        analysis = _generate_analysis(empathy_score, behavior_score, empathy_data, behavior_data)
    
    # 加权计算最终得分
    final_score = (
        WEIGHTS["empathy"] * empathy_score +
        WEIGHTS["behavior"] * behavior_score
    ) * 100
    
    # 确保分数在 0-100 范围内并取整
    final_score = int(round(max(0.0, min(100.0, final_score))))
    
    # 构建返回结果
    result = {
        "score": final_score,
        "analysis": analysis,
        "details": {
            "empathy": {
                "score": round(empathy_score * 100, 1),
                "weight": WEIGHTS["empathy"],
                "sub_scores": {
                    "empathy": empathy_data.get("empathy", 5),
                    "politeness": empathy_data.get("politeness", 5),
                    "clarity": empathy_data.get("clarity", 5)
                }
            },
            "behavior": {
                "score": round(behavior_score * 100, 1),
                "weight": WEIGHTS["behavior"],
                "counts": {
                    "comfort": sum(1 for u in behavior_data.get("utterances", []) if u.get("behavior") == "comfort"),
                    "explain": sum(1 for u in behavior_data.get("utterances", []) if u.get("behavior") == "explain"),
                    "instruction": sum(1 for u in behavior_data.get("utterances", []) if u.get("behavior") == "instruction"),
                    "ignore": sum(1 for u in behavior_data.get("utterances", []) if u.get("behavior") == "ignore")
                }
            }
        }
    }
    
    return {"raw_response": json.dumps(result, ensure_ascii=False)}
