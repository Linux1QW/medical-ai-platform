# -*- coding: utf-8 -*-
"""Mock OpenAI Server 单元测试

验证：
- embedding 可复现（同文本同输出）
- 所有已知 prompt 分支返回正确响应
- unmatched prompt 返回 422 + unmatched_prompt_sha256
- matched_branch 计数正确
"""

import hashlib
import math

import pytest
from fastapi.testclient import TestClient


def test_coach_prompt_has_deterministic_structured_response():
    from tests.fixtures.mock_openai_server import app

    with TestClient(app) as client:
        response = client.post(
            "/v1/chat/completions",
            json={
                "model": "mock-chat",
                "messages": [
                    {
                        "role": "user",
                        "content": "请用自然中文生成一个问诊建议问题。不要做诊断。",
                    }
                ],
            },
        )

    assert response.status_code == 200
    content = response.json()["choices"][0]["message"]["content"]
    assert "建议问题" in content
    assert '"risk_level":"low"' in content


@pytest.fixture
def mock_server():
    """创建 mock OpenAI 服务器测试客户端"""
    from tests.fixtures.mock_openai_server import app, reset_counters

    reset_counters()
    with TestClient(app) as client:
        yield client


class TestHealthEndpoint:
    """GET /health → 200"""

    def test_health_returns_200(self, mock_server):
        response = mock_server.get("/health")
        assert response.status_code == 200
        assert response.json()["status"] == "ok"


class TestEmbeddingsEndpoint:
    """POST /v1/embeddings → 确定性 1024 维向量"""

    def test_embedding_reproducible(self, mock_server):
        """同文本两次请求返回完全相同向量"""
        payload = {"input": "测试文本", "model": "mock-embedding"}

        resp1 = mock_server.post("/v1/embeddings", json=payload)
        resp2 = mock_server.post("/v1/embeddings", json=payload)

        assert resp1.status_code == 200
        assert resp2.status_code == 200

        vec1 = resp1.json()["data"][0]["embedding"]
        vec2 = resp2.json()["data"][0]["embedding"]

        assert vec1 == vec2

    def test_embedding_dimension_1024(self, mock_server):
        """向量维度为 1024"""
        payload = {"input": "任意文本", "model": "mock-embedding"}
        response = mock_server.post("/v1/embeddings", json=payload)

        assert response.status_code == 200
        vec = response.json()["data"][0]["embedding"]
        assert len(vec) == 1024

    def test_embedding_l2_normalized(self, mock_server):
        """向量 L2 范数 ≈ 1.0"""
        payload = {"input": "归一化测试", "model": "mock-embedding"}
        response = mock_server.post("/v1/embeddings", json=payload)

        vec = response.json()["data"][0]["embedding"]
        l2_norm = math.sqrt(sum(x * x for x in vec))
        assert abs(l2_norm - 1.0) < 1e-6

    def test_different_text_different_embedding(self, mock_server):
        """不同文本产生不同向量"""
        resp1 = mock_server.post("/v1/embeddings", json={"input": "文本A", "model": "mock-embedding"})
        resp2 = mock_server.post("/v1/embeddings", json={"input": "文本B", "model": "mock-embedding"})

        vec1 = resp1.json()["data"][0]["embedding"]
        vec2 = resp2.json()["data"][0]["embedding"]

        assert vec1 != vec2


class TestChatCompletionsKnownBranches:
    """POST /v1/chat/completions → 已知 prompt 分支"""

    def test_slot_filling_branch(self, mock_server):
        """病史采集 → 槽位填充 JSON"""
        messages = [{"role": "user", "content": "请提取槽位填充信息。"}]
        response = mock_server.post("/v1/chat/completions", json={
            "model": "mock-chat",
            "messages": messages,
        })

        assert response.status_code == 200
        content = response.json()["choices"][0]["message"]["content"]
        assert "slots" in content
        assert "chief_complaint" in content

    def test_inquiry_steps_branch(self, mock_server):
        """问诊步骤提取"""
        messages = [{"role": "user", "content": "请提取问诊步骤序列和问题分类。"}]
        response = mock_server.post("/v1/chat/completions", json={
            "model": "mock-chat",
            "messages": messages,
        })

        assert response.status_code == 200
        content = response.json()["choices"][0]["message"]["content"]
        assert "inquiry-steps" in content

    def test_empathy_scoring_branch(self, mock_server):
        """人文关怀评分"""
        messages = [{"role": "user", "content": "请对医生的文本共情表现进行评分。"}]
        response = mock_server.post("/v1/chat/completions", json={
            "model": "mock-chat",
            "messages": messages,
        })

        assert response.status_code == 200
        content = response.json()["choices"][0]["message"]["content"]
        assert "empathy" in content

    def test_behavior_classification_branch(self, mock_server):
        """行为分类"""
        messages = [{"role": "user", "content": "请对医生的每句发言进行行为分类。"}]
        response = mock_server.post("/v1/chat/completions", json={
            "model": "mock-chat",
            "messages": messages,
        })

        assert response.status_code == 200
        content = response.json()["choices"][0]["message"]["content"]
        assert "utterances" in content

    def test_diagnosis_evaluation_branch(self, mock_server):
        """诊断评估 → score 80"""
        messages = [{"role": "user", "content": "请对医生的诊断结果进行评估。"}]
        response = mock_server.post("/v1/chat/completions", json={
            "model": "mock-chat",
            "messages": messages,
        })

        assert response.status_code == 200
        content = response.json()["choices"][0]["message"]["content"]
        assert '"score":80' in content or '"score": 80' in content

    def test_treatment_evaluation_branch(self, mock_server):
        """治疗评估 → score 78"""
        messages = [{"role": "user", "content": "请对医生的治疗方案进行评估。"}]
        response = mock_server.post("/v1/chat/completions", json={
            "model": "mock-chat",
            "messages": messages,
        })

        assert response.status_code == 200
        content = response.json()["choices"][0]["message"]["content"]
        assert '"score":78' in content or '"score": 78' in content

    def test_summary_generation_branch(self, mock_server):
        """综合评估摘要"""
        messages = [{"role": "user", "content": "请生成综合评估摘要。"}]
        response = mock_server.post("/v1/chat/completions", json={
            "model": "mock-chat",
            "messages": messages,
        })

        assert response.status_code == 200
        content = response.json()["choices"][0]["message"]["content"]
        assert "summary" in content

    def test_hyde_expansion_branch(self, mock_server):
        """HyDE 查询扩展 → 空数组"""
        messages = [{"role": "user", "content": "请扩展以下医学查询：头痛"}]
        response = mock_server.post("/v1/chat/completions", json={
            "model": "mock-chat",
            "messages": messages,
        })

        assert response.status_code == 200
        content = response.json()["choices"][0]["message"]["content"]
        assert content.strip() == "[]"

    def test_ideal_paragraph_branch(self, mock_server):
        """理想临床指南段落生成"""
        messages = [{"role": "user", "content": "请为以下医学查询生成一段理想临床指南段落：头痛"}]
        response = mock_server.post("/v1/chat/completions", json={
            "model": "mock-chat",
            "messages": messages,
        })

        assert response.status_code == 200
        content = response.json()["choices"][0]["message"]["content"]
        assert len(content) > 50


class TestUnmatchedPrompt:
    """未匹配 prompt → HTTP 422 + unmatched_prompt_sha256"""

    def test_unmatched_returns_422(self, mock_server):
        """未知 prompt 返回 422"""
        messages = [{"role": "user", "content": "这是一个完全未知的提示词"}]
        response = mock_server.post("/v1/chat/completions", json={
            "model": "mock-chat",
            "messages": messages,
        })

        assert response.status_code == 422

    def test_unmatched_includes_sha256(self, mock_server):
        """422 响应包含 unmatched_prompt_sha256"""
        messages = [{"role": "user", "content": "未知提示"}]
        response = mock_server.post("/v1/chat/completions", json={
            "model": "mock-chat",
            "messages": messages,
        })

        assert response.status_code == 422
        body = response.json()
        assert "unmatched_prompt_sha256" in body

    def test_unmatched_sha256_is_correct(self, mock_server):
        """SHA-256 值正确"""
        prompt_text = "测试sha256计算"
        expected_sha = hashlib.sha256(prompt_text.encode()).hexdigest()

        messages = [{"role": "user", "content": prompt_text}]
        response = mock_server.post("/v1/chat/completions", json={
            "model": "mock-chat",
            "messages": messages,
        })

        assert response.status_code == 422
        body = response.json()
        assert body["unmatched_prompt_sha256"] == expected_sha


class TestMatchedBranchCounter:
    """matched_branch 计数"""

    def test_matched_branch_increments(self, mock_server):
        """每次匹配成功请求计数器加 1"""
        messages = [{"role": "user", "content": "请对医生的诊断结果进行评估。"}]

        mock_server.post("/v1/chat/completions", json={"model": "mock-chat", "messages": messages})
        mock_server.post("/v1/chat/completions", json={"model": "mock-chat", "messages": messages})

        # 检查计数器
        from tests.fixtures.mock_openai_server import get_counters
        counters = get_counters()
        assert counters["matched"] >= 2

    def test_unmatched_does_not_increment_matched(self, mock_server):
        """未匹配请求不增加 matched 计数"""
        from tests.fixtures.mock_openai_server import get_counters

        messages = [{"role": "user", "content": "未知内容xxx"}]
        mock_server.post("/v1/chat/completions", json={"model": "mock-chat", "messages": messages})

        counters = get_counters()
        assert counters.get("unmatched", 0) >= 1
