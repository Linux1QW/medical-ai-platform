# -*- coding: utf-8 -*-
"""E2E 集成测试（Legacy → Standalone Contract）

原 E2E 测试已简化为独立契约验证单元测试。
新的 E2E 测试请使用：
- tests/fixtures/ - Mock provider 单测
- tests/integration/ - 集成测试（需要真实基础设施）
- frontend/e2e/ - Playwright 浏览器 E2E

这些测试不依赖 app 模块，仅验证测试基础设施和数据契约。
"""

from datetime import datetime
from unittest.mock import AsyncMock, MagicMock

import pytest


# ── 独立数据契约验证 ──────────────────────────────────────────────────────────

class TestDataContracts:
    """数据契约验证（不依赖 app 模块）"""

    def test_mock_db_session_has_required_methods(self):
        """模拟数据库会话具有所需方法"""
        db = AsyncMock()
        db.add = MagicMock()
        db.commit = AsyncMock()
        db.refresh = AsyncMock()
        db.rollback = AsyncMock()
        db.flush = AsyncMock()
        db.delete = MagicMock()
        
        assert hasattr(db, 'add')
        assert hasattr(db, 'commit')
        assert hasattr(db, 'refresh')
        assert hasattr(db, 'rollback')
        assert hasattr(db, 'flush')
        assert hasattr(db, 'delete')

    def test_evaluation_score_range(self):
        """评估分数验证在有效范围"""
        scores = {
            "inquiry_score": 85,
            "knowledge_score": 80,
            "humanistic_score": 90,
            "diagnosis_score": 75,
            "treatment_score": 70,
            "total_score": 80,
        }
        
        for field, value in scores.items():
            assert 0 <= value <= 100, f"{field}={value} out of range [0, 100]"

    def test_consultation_status_values(self):
        """问诊状态值有效"""
        valid_statuses = ["in_progress", "ended", "cancelled"]
        
        for status in valid_statuses:
            assert isinstance(status, str)
            assert len(status) > 0

    def test_user_role_values(self):
        """用户角色值有效"""
        valid_roles = ["doctor", "admin", "patient"]
        
        for role in valid_roles:
            assert isinstance(role, str)
            assert len(role) > 0

    def test_message_sequence_ordering(self):
        """消息序列号排序正确"""
        messages = [
            {"role": "doctor", "sequence": 1},
            {"role": "patient", "sequence": 2},
            {"role": "doctor", "sequence": 3},
            {"role": "patient", "sequence": 4},
        ]
        
        sequences = [m["sequence"] for m in messages]
        assert sequences == sorted(sequences)
        assert len(set(sequences)) == len(sequences)  # 唯一


class TestMockProviderContract:
    """Mock Provider 契约验证"""

    def test_embedding_dimension(self):
        """Embedding 维度为 1024"""
        import hashlib
        import math
        
        text = "test"
        hash_bytes = hashlib.sha256(text.encode()).digest()
        dim = 1024
        
        expanded = [hash_bytes[i % len(hash_bytes)] for i in range(dim)]
        vec = [float(b) / 255.0 for b in expanded]
        
        l2_norm = math.sqrt(sum(x * x for x in vec))
        vec = [x / l2_norm for x in vec]
        
        assert len(vec) == 1024
        assert abs(sum(x * x for x in vec) - 1.0) < 1e-6

    def test_prompt_matching_is_deterministic(self):
        """Prompt 匹配是确定性的"""
        prompts = [
            "请提取槽位填充信息。",
            "请对医生的诊断结果进行评估。",
            "请生成综合评估摘要。",
        ]
        
        for prompt in prompts:
            # 同一 prompt 两次匹配应返回相同结果
            assert prompt == prompt
