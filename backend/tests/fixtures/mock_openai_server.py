# -*- coding: utf-8 -*-
"""Mock OpenAI-Compatible Server for E2E Testing

提供确定性的 chat/embedding 响应，用于生产闭环 E2E 测试。

特性：
- GET /health → 200
- POST /v1/embeddings → SHA-256 生成固定 1024 维向量，L2 normalize
- POST /v1/chat/completions → 按 prompt 关键词选择响应
- 未匹配 prompt → HTTP 422 + unmatched_prompt_sha256
- 记录 matched_branch / unmatched 计数
"""

import hashlib
import math
import threading
from typing import Any

from fastapi import FastAPI
from fastapi.responses import JSONResponse
from pydantic import BaseModel

# ──────────────────────────────────────────
# 计数器（线程安全）
# ──────────────────────────────────────────
_counters_lock = threading.Lock()
_counters = {"matched": 0, "unmatched": 0}


def reset_counters():
    """重置计数器（测试前调用）"""
    with _counters_lock:
        _counters["matched"] = 0
        _counters["unmatched"] = 0


def get_counters() -> dict:
    """获取当前计数器值"""
    with _counters_lock:
        return dict(_counters)


# ──────────────────────────────────────────
# 固定响应表格
# ──────────────────────────────────────────
PROMPT_RESPONSES: list[tuple[str, str]] = [
    # V1.2 Coach structured suggestion
    (
        "请用自然中文生成一个问诊建议问题",
        '{"question":"建议问题：这些症状从什么时候开始，是否持续加重？",'
        '"rationale":"补充症状起病时间与进展有助于完善问诊信息。",'
        '"confidence":0.9,"risk_level":"low"}',
    ),
    # 槽位填充
    (
        "请提取槽位填充信息。",
        '{"slots":{"chief_complaint":{"symptom":true,"duration":true,"severity":true},"history":{"onset":true,"progression":true,"trigger":false},"past_history":{"disease":true,"surgery":false},"medication":{"current_drugs":false},"allergy":{"drug_allergy":true}},"critical_slots":{"associated_symptom":true,"risk_factor":true}}',
    ),
    # 问诊步骤
    (
        "请提取问诊步骤序列和问题分类。",
        '{"inquiry-steps":["symptom","history","risk_factor","past_history"],"question-classification":[{"question":"哪里不舒服","type":"symptom","category":"relevant"},{"question":"持续多久","type":"history","category":"relevant"},{"question":"是否发热","type":"risk_factor","category":"relevant"}]}',
    ),
    # 共情评分
    (
        "请对医生的文本共情表现进行评分。",
        '{"empathy":8,"politeness":9,"clarity":8}',
    ),
    # 行为分类
    (
        "请对医生的每句发言进行行为分类。",
        '{"utterances":[{"text":"请描述症状","behavior":"instruction"},{"text":"我理解您的担心","behavior":"comfort"},{"text":"建议完成相关检查","behavior":"explain"}]}',
    ),
    # 诊断评估
    (
        "请对医生的诊断结果进行评估。",
        '{"score":80,"analysis":"诊断方向基本合理，仍需结合检查结果确认。"}',
    ),
    # 治疗评估
    (
        "请对医生的治疗方案进行评估。",
        '{"score":78,"analysis":"方案总体合理，需补充用药注意事项和随访计划。"}',
    ),
    # 综合评估摘要
    (
        "请生成综合评估摘要。",
        '{"summary":"问诊结构完整，诊断和治疗方向基本合理；医学知识证据不足，需人工复核。"}',
    ),
]

# 前缀匹配
PREFIX_RESPONSES: list[tuple[str, str]] = [
    # HyDE 扩展
    ("请扩展以下医学查询：", "[]"),
    # 理想段落生成
    (
        "请为以下医学查询生成一段理想临床指南段落：",
        "本段落为合成测试内容，用于验证系统功能。不包含任何真实疾病建议或患者信息。"
        "此段落长度超过50字符，以满足系统对响应长度的最低要求。"
        "测试数据仅供开发验证使用，不应作为临床参考。",
    ),
]


# ──────────────────────────────────────────
# Embedding 生成（确定性）
# ──────────────────────────────────────────
def generate_deterministic_embedding(text: str, dim: int = 1024) -> list[float]:
    """使用 SHA-256 生成确定性 1024 维向量并 L2 normalize"""
    # 计算 SHA-256 hash
    hash_bytes = hashlib.sha256(text.encode("utf-8")).digest()

    # 循环扩展到 dim 个字节
    expanded = []
    for i in range(dim):
        expanded.append(hash_bytes[i % len(hash_bytes)])

    # 转换为 float 并 L2 normalize
    vec = [float(b) / 255.0 for b in expanded]

    # L2 normalize
    l2_norm = math.sqrt(sum(x * x for x in vec))
    if l2_norm > 0:
        vec = [x / l2_norm for x in vec]

    return vec


# ──────────────────────────────────────────
# Prompt 匹配
# ──────────────────────────────────────────
def match_prompt(messages: list[dict]) -> tuple[str | None, str | None]:
    """
    匹配最后一条 user/system message 的关键词

    Returns:
        (response_content, match_type) or (None, None) if unmatched
    """
    # 获取最后一条 user 或 system message
    last_content = ""
    for msg in reversed(messages):
        if msg.get("role") in ("user", "system"):
            last_content = msg.get("content", "")
            break

    if not last_content:
        return None, None

    # 精确匹配
    for trigger, response in PROMPT_RESPONSES:
        if trigger in last_content:
            return response, "exact"

    # 前缀匹配
    for prefix, response in PREFIX_RESPONSES:
        if last_content.startswith(prefix) or prefix in last_content:
            return response, "prefix"

    return None, None


# ──────────────────────────────────────────
# FastAPI App
# ──────────────────────────────────────────
app = FastAPI(title="Mock OpenAI Server")


class EmbeddingRequest(BaseModel):
    input: str | list[str]
    model: str | None = None


class ChatMessage(BaseModel):
    role: str
    content: str


class ChatCompletionRequest(BaseModel):
    model: str | None = None
    messages: list[ChatMessage]
    temperature: float | None = 1.0
    max_tokens: int | None = None
    tools: list[Any] | None = None


@app.get("/health")
async def health():
    """健康检查"""
    return {"status": "ok"}


@app.post("/v1/embeddings")
async def create_embeddings(request: EmbeddingRequest):
    """生成确定性 embedding"""
    inputs = request.input if isinstance(request.input, list) else [request.input]

    data = []
    for i, text in enumerate(inputs):
        embedding = generate_deterministic_embedding(text)
        data.append({
            "object": "embedding",
            "embedding": embedding,
            "index": i,
        })

    return {
        "object": "list",
        "data": data,
        "model": request.model or "mock-embedding",
        "usage": {"prompt_tokens": sum(len(t) for t in inputs), "total_tokens": sum(len(t) for t in inputs)},
    }


@app.post("/v1/chat/completions")
async def create_chat_completion(request: ChatCompletionRequest):
    """按 prompt 关键词路由 chat completion"""
    messages = [{"role": m.role, "content": m.content} for m in request.messages]

    response_content, match_type = match_prompt(messages)

    if response_content is None:
        # 未匹配 → 422
        with _counters_lock:
            _counters["unmatched"] += 1

        # 计算 prompt SHA-256
        last_content = ""
        for msg in reversed(messages):
            if msg.get("role") in ("user", "system"):
                last_content = msg.get("content", "")
                break

        prompt_sha = hashlib.sha256(last_content.encode("utf-8")).hexdigest()

        return JSONResponse(
            status_code=422,
            content={
                "error": "unmatched_prompt",
                "unmatched_prompt_sha256": prompt_sha,
                "message": f"Mock server 未匹配 prompt，请检查 prompt 表格。SHA-256: {prompt_sha}",
            },
        )

    # 匹配成功
    with _counters_lock:
        _counters["matched"] += 1

    return {
        "id": "chatcmpl-mock",
        "object": "chat.completion",
        "created": 1234567890,
        "model": request.model or "mock-chat",
        "choices": [
            {
                "index": 0,
                "message": {
                    "role": "assistant",
                    "content": response_content,
                },
                "finish_reason": "stop",
            }
        ],
        "usage": {"prompt_tokens": 10, "completion_tokens": len(response_content), "total_tokens": 10 + len(response_content)},
    }


# ──────────────────────────────────────────
# 独立运行入口
# ──────────────────────────────────────────
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8080)
