# -*- coding: utf-8 -*-
"""consultation_service.py 单元测试

测试策略：
- Mock 数据库会话（AsyncSession）和 LLM 调用
- 覆盖核心路径：创建问诊、发送消息、SSE 流式响应、记忆窗口管理、结束问诊、删除问诊
"""

import json
import pytest
from unittest.mock import AsyncMock, MagicMock, patch, PropertyMock
from datetime import datetime

from app.models.consultation import Consultation, ConsultationMessage
from app.models.patient import VirtualPatient


# ── Fixtures ──────────────────────────────────────────────────────────────────

@pytest.fixture
def mock_db():
    """创建模拟数据库会话"""
    db = AsyncMock()
    db.add = MagicMock()
    db.commit = AsyncMock()
    db.refresh = AsyncMock()
    db.rollback = AsyncMock()
    db.delete = AsyncMock()
    db.flush = AsyncMock()
    return db


@pytest.fixture
def sample_consultation():
    """创建模拟问诊记录"""
    c = Consultation(
        id=1,
        doctor_id=10,
        patient_id=20,
        status="in_progress",
        started_at=datetime(2025, 1, 1, 10, 0),
        ended_at=None,
        max_rounds=20,
    )
    return c


@pytest.fixture
def sample_patient():
    """创建模拟患者"""
    p = VirtualPatient(
        id=20,
        name="张三",
        age=45,
        gender="男",
        personality_type="配合型",
        chief_complaint="头痛三天",
        medical_history="无特殊病史",
        symptoms='["头痛", "发热"]',
        system_prompt="患者张三，男，45岁，头痛三天伴低热。",
        expected_diagnosis="上呼吸道感染",
    )
    return p


@pytest.fixture
def sample_messages():
    """创建模拟对话消息列表"""
    return [
        ConsultationMessage(id=1, consultation_id=1, role="doctor", content="哪里不舒服？", sequence=1),
        ConsultationMessage(id=2, consultation_id=1, role="patient", content="头痛。", sequence=2),
        ConsultationMessage(id=3, consultation_id=1, role="doctor", content="多久了？", sequence=3),
        ConsultationMessage(id=4, consultation_id=1, role="patient", content="三天了。", sequence=4),
    ]


# ── 测试创建问诊 ──────────────────────────────────────────────────────────────

class TestCreateConsultation:
    """测试 create_consultation"""

    @pytest.mark.asyncio
    async def test_create_consultation_success(self, mock_db, sample_consultation):
        """成功创建问诊记录"""
        from app.services.consultation_service import create_consultation

        # mock refresh 后设置 id
        async def mock_refresh(obj):
            obj.id = 1

        mock_db.refresh = mock_refresh

        result = await create_consultation(mock_db, doctor_id=10, patient_id=20)
        mock_db.add.assert_called_once()
        mock_db.commit.assert_called_once()
        assert result.doctor_id == 10
        assert result.patient_id == 20


# ── 测试获取问诊 ──────────────────────────────────────────────────────────────

class TestGetConsultation:
    """测试 get_consultation"""

    @pytest.mark.asyncio
    async def test_get_consultation_found(self, mock_db, sample_consultation):
        """找到问诊记录时返回对象"""
        from app.services.consultation_service import get_consultation

        mock_result = MagicMock()
        mock_result.scalar_one_or_none.return_value = sample_consultation
        mock_db.execute = AsyncMock(return_value=mock_result)

        result = await get_consultation(mock_db, 1)
        assert result is not None
        assert result.id == 1
        assert result.doctor_id == 10

    @pytest.mark.asyncio
    async def test_get_consultation_not_found(self, mock_db):
        """未找到问诊记录时返回 None"""
        from app.services.consultation_service import get_consultation

        mock_result = MagicMock()
        mock_result.scalar_one_or_none.return_value = None
        mock_db.execute = AsyncMock(return_value=mock_result)

        result = await get_consultation(mock_db, 999)
        assert result is None


# ── 测试获取消息列表 ──────────────────────────────────────────────────────────

class TestGetMessages:
    """测试 get_messages"""

    @pytest.mark.asyncio
    async def test_get_messages_returns_ordered(self, mock_db, sample_messages):
        """返回按序号排序的消息列表"""
        from app.services.consultation_service import get_messages

        mock_result = MagicMock()
        mock_scalars = MagicMock()
        mock_scalars.all.return_value = sample_messages
        mock_result.scalars.return_value = mock_scalars
        mock_db.execute = AsyncMock(return_value=mock_result)

        result = await get_messages(mock_db, 1)
        assert len(result) == 4
        assert result[0].role == "doctor"
        assert result[1].role == "patient"


# ── 测试发送医生消息 ──────────────────────────────────────────────────────────

class TestSendDoctorMessage:
    """测试 send_doctor_message"""

    @pytest.mark.asyncio
    @patch("app.services.consultation_service.call_qwen_chat", new_callable=AsyncMock)
    @patch("app.services.consultation_service.get_consultation", new_callable=AsyncMock)
    @patch("app.services.consultation_service.get_messages", new_callable=AsyncMock)
    async def test_send_message_returns_doctor_and_patient_msgs(
        self, mock_get_msgs, mock_get_consult, mock_qwen, mock_db,
        sample_consultation, sample_patient, sample_messages,
    ):
        """发送消息后返回医生消息和患者回复"""
        from app.services.consultation_service import send_doctor_message

        mock_get_msgs.return_value = sample_messages
        mock_get_consult.return_value = sample_consultation
        mock_qwen.return_value = "我头痛，还有点发烧。"

        # mock 患者查询
        mock_patient_result = MagicMock()
        mock_patient_result.scalar_one.return_value = sample_patient
        mock_db.execute = AsyncMock(return_value=mock_patient_result)

        # mock refresh
        async def mock_refresh(obj):
            if not hasattr(obj, 'id') or obj.id is None:
                obj.id = 100
        mock_db.refresh = mock_refresh

        doctor_msg, patient_msg = await send_doctor_message(mock_db, 1, "你有什么症状？")

        assert doctor_msg.role == "doctor"
        assert doctor_msg.content == "你有什么症状？"
        assert patient_msg.role == "patient"
        assert patient_msg.content == "我头痛，还有点发烧。"
        mock_db.commit.assert_called_once()

    @pytest.mark.asyncio
    @patch("app.services.consultation_service.call_qwen_chat", new_callable=AsyncMock)
    @patch("app.services.consultation_service.get_consultation", new_callable=AsyncMock)
    @patch("app.services.consultation_service.get_messages", new_callable=AsyncMock)
    async def test_send_message_memory_window_truncation(
        self, mock_get_msgs, mock_get_consult, mock_qwen, mock_db,
        sample_consultation, sample_patient,
    ):
        """消息数未超限时使用完整最近窗口"""
        from app.services.consultation_service import send_doctor_message, MEMORY_RECENT_TURNS

        # 创建少量消息（不超过压缩阈值）
        few_messages = [
            ConsultationMessage(id=i, consultation_id=1, role="doctor" if i % 2 == 1 else "patient",
                              content=f"msg{i}", sequence=i)
            for i in range(1, 6)
        ]
        mock_get_msgs.return_value = few_messages
        mock_get_consult.return_value = sample_consultation
        mock_qwen.return_value = "好的。"

        mock_patient_result = MagicMock()
        mock_patient_result.scalar_one.return_value = sample_patient
        mock_db.execute = AsyncMock(return_value=mock_patient_result)

        async def mock_refresh(obj):
            obj.id = 100
        mock_db.refresh = mock_refresh

        await send_doctor_message(mock_db, 1, "你好")

        # 验证 call_qwen_chat 被调用（包含系统提示 + 历史 + 当前消息）
        mock_qwen.assert_called_once()
        call_args = mock_qwen.call_args
        chat_history = call_args[0][0]
        # 系统提示 + 历史消息 + 当前消息
        assert chat_history[0]["role"] == "system"
        assert chat_history[-1]["role"] == "user"
        assert chat_history[-1]["content"] == "你好"

    @pytest.mark.asyncio
    @patch("app.services.consultation_service._summarize_early_messages", new_callable=AsyncMock)
    @patch("app.services.consultation_service.call_qwen_chat", new_callable=AsyncMock)
    @patch("app.services.consultation_service.get_consultation", new_callable=AsyncMock)
    @patch("app.services.consultation_service.get_messages", new_callable=AsyncMock)
    async def test_send_message_memory_compression_triggered(
        self, mock_get_msgs, mock_get_consult, mock_qwen, mock_summarize, mock_db,
        sample_consultation, sample_patient,
    ):
        """消息数超过压缩阈值时触发早期对话压缩"""
        from app.services.consultation_service import send_doctor_message, MEMORY_COMPRESS_THRESHOLD

        # 创建超过压缩阈值的消息
        many_messages = [
            ConsultationMessage(id=i, consultation_id=1, role="doctor" if i % 2 == 1 else "patient",
                              content=f"msg{i}", sequence=i)
            for i in range(1, MEMORY_COMPRESS_THRESHOLD * 2 + 5)
        ]
        mock_get_msgs.return_value = many_messages
        mock_get_consult.return_value = sample_consultation
        mock_summarize.return_value = "【已暴露症状】头痛、发热\n【否认症状】无"
        mock_qwen.return_value = "好的。"

        mock_patient_result = MagicMock()
        mock_patient_result.scalar_one.return_value = sample_patient
        mock_db.execute = AsyncMock(return_value=mock_patient_result)

        async def mock_refresh(obj):
            obj.id = 100
        mock_db.refresh = mock_refresh

        await send_doctor_message(mock_db, 1, "还有什么不舒服？")

        # 验证压缩函数被调用
        mock_summarize.assert_called_once()
        # 验证 LLM 调用时 chat_history 包含摘要
        call_args = mock_qwen.call_args
        chat_history = call_args[0][0]
        summary_messages = [m for m in chat_history if "早期问诊记录摘要" in m.get("content", "")]
        assert len(summary_messages) == 1


# ── 测试 SSE 流式响应 ─────────────────────────────────────────────────────────

class TestSendMessageStream:
    """测试 send_doctor_message_stream"""

    @pytest.mark.asyncio
    @patch("app.services.consultation_service.call_qwen_chat", new_callable=AsyncMock)
    @patch("app.services.consultation_service.get_consultation", new_callable=AsyncMock)
    @patch("app.services.consultation_service.get_messages", new_callable=AsyncMock)
    async def test_stream_emits_progress_and_complete(
        self, mock_get_msgs, mock_get_consult, mock_qwen, mock_db,
        sample_consultation, sample_patient,
    ):
        """SSE 流式响应包含 progress 和 complete 事件"""
        from app.services.consultation_service import send_doctor_message_stream

        mock_get_msgs.return_value = []
        mock_get_consult.return_value = sample_consultation
        mock_qwen.return_value = "我头痛。"

        mock_patient_result = MagicMock()
        mock_patient_result.scalar_one.return_value = sample_patient
        mock_db.execute = AsyncMock(return_value=mock_patient_result)

        async def mock_refresh(obj):
            obj.id = 100
            obj.created_at = datetime(2025, 1, 1, 10, 0)
        mock_db.refresh = mock_refresh

        events = []
        async for event in send_doctor_message_stream(mock_db, 1, "哪里不舒服？"):
            events.append(event)

        # 应至少有多个 progress 事件和 1 个 complete 事件
        assert len(events) >= 2

        # 最后一个事件应为 complete
        last_event = events[-1]
        assert "event: complete" in last_event

        # 解析 complete 事件数据
        data_line = [l for l in last_event.split("\n") if l.startswith("data: ")][0]
        data = json.loads(data_line[6:])
        assert "doctor_msg" in data
        assert "patient_msg" in data
        assert data["patient_msg"]["content"] == "我头痛。"

    @pytest.mark.asyncio
    @patch("app.services.consultation_service.call_qwen_chat", new_callable=AsyncMock)
    @patch("app.services.consultation_service.get_consultation", new_callable=AsyncMock)
    @patch("app.services.consultation_service.get_messages", new_callable=AsyncMock)
    async def test_stream_error_event_on_failure(
        self, mock_get_msgs, mock_get_consult, mock_qwen, mock_db,
    ):
        """LLM 调用失败时发送 error 事件"""
        from app.services.consultation_service import send_doctor_message_stream

        mock_get_msgs.return_value = []
        mock_get_consult.side_effect = Exception("DB error")

        events = []
        async for event in send_doctor_message_stream(mock_db, 1, "你好"):
            events.append(event)

        # 应有 error 事件
        error_events = [e for e in events if "event: error" in e]
        assert len(error_events) == 1
        assert "DB error" in error_events[0]


# ── 测试 _make_sse_event ──────────────────────────────────────────────────────

class TestMakeSSEEvent:
    """测试 SSE 事件构造"""

    def test_sse_event_format(self):
        """SSE 事件格式正确"""
        from app.services.consultation_service import _make_sse_event

        event = _make_sse_event("progress", {"step": "test", "progress": 50})
        assert event.startswith("event: progress\n")
        assert "data: " in event
        assert event.endswith("\n\n")

        # 验证 JSON 数据可解析
        data_line = [l for l in event.split("\n") if l.startswith("data: ")][0]
        data = json.loads(data_line[6:])
        assert data["step"] == "test"
        assert data["progress"] == 50

    def test_sse_event_chinese_content(self):
        """SSE 事件正确处理中文内容"""
        from app.services.consultation_service import _make_sse_event

        event = _make_sse_event("complete", {"content": "患者回复内容"})
        assert "患者回复内容" in event


# ── 测试结束问诊 ──────────────────────────────────────────────────────────────

class TestEndConsultation:
    """测试 end_consultation"""

    @pytest.mark.asyncio
    async def test_end_consultation_sets_status_and_time(self, mock_db, sample_consultation):
        """结束问诊设置状态为 completed 并记录结束时间"""
        from app.services.consultation_service import end_consultation

        mock_result = MagicMock()
        mock_result.scalar_one_or_none.return_value = sample_consultation
        mock_db.execute = AsyncMock(return_value=mock_result)

        result = await end_consultation(mock_db, 1)
        assert result.status == "completed"
        assert result.ended_at is not None
        mock_db.commit.assert_called_once()


# ── 测试提交诊断 ──────────────────────────────────────────────────────────────

class TestSubmitDiagnosis:
    """测试 submit_diagnosis"""

    @pytest.mark.asyncio
    async def test_submit_diagnosis_sets_fields(self, mock_db, sample_consultation):
        """提交诊断设置诊断和治疗方案字段"""
        from app.services.consultation_service import submit_diagnosis

        mock_result = MagicMock()
        mock_result.scalar_one_or_none.return_value = sample_consultation
        mock_db.execute = AsyncMock(return_value=mock_result)

        result = await submit_diagnosis(mock_db, 1, "上呼吸道感染", "多喝水休息，必要时服用退烧药")
        assert result.diagnosis == "上呼吸道感染"
        assert result.treatment_plan == "多喝水休息，必要时服用退烧药"
        assert result.status == "completed"
        mock_db.commit.assert_called_once()


# ── 测试删除问诊 ──────────────────────────────────────────────────────────────

class TestDeleteConsultation:
    """测试 delete_consultation"""

    @staticmethod
    def _user(user_id: int, role: str = "doctor"):
        user = MagicMock()
        user.id = user_id
        user.role = role
        return user

    @pytest.mark.asyncio
    async def test_delete_own_consultation_returns_true(self, mock_db, sample_consultation):
        """删除本人问诊返回 True"""
        from app.services.consultation_service import delete_consultation

        mock_result = MagicMock()
        mock_result.scalar_one_or_none.return_value = sample_consultation
        mock_db.execute = AsyncMock(return_value=mock_result)

        result = await delete_consultation(mock_db, 1, self._user(10))
        assert result is True
        mock_db.delete.assert_called_once()
        mock_db.commit.assert_called_once()

    @pytest.mark.asyncio
    async def test_delete_other_doctor_consultation_returns_false(self, mock_db, sample_consultation):
        """删除他人问诊返回 False"""
        from app.services.consultation_service import delete_consultation

        mock_result = MagicMock()
        mock_result.scalar_one_or_none.return_value = sample_consultation
        mock_db.execute = AsyncMock(return_value=mock_result)

        result = await delete_consultation(mock_db, 1, self._user(99))
        assert result is False
        mock_db.delete.assert_not_called()

    @pytest.mark.asyncio
    async def test_admin_can_delete_other_doctor_consultation(self, mock_db, sample_consultation):
        """管理员可删除他人问诊"""
        from app.services.consultation_service import delete_consultation

        mock_result = MagicMock()
        mock_result.scalar_one_or_none.return_value = sample_consultation
        mock_db.execute = AsyncMock(return_value=mock_result)

        result = await delete_consultation(mock_db, 1, self._user(99, role="admin"))
        assert result is True
        mock_db.delete.assert_called_once()

    @pytest.mark.asyncio
    async def test_delete_nonexistent_consultation_returns_false(self, mock_db):
        """删除不存在的问诊返回 False"""
        from app.services.consultation_service import delete_consultation

        mock_result = MagicMock()
        mock_result.scalar_one_or_none.return_value = None
        mock_db.execute = AsyncMock(return_value=mock_result)

        result = await delete_consultation(mock_db, 999, self._user(10))
        assert result is False


# ── 测试 _summarize_early_messages ────────────────────────────────────────────

class TestSummarizeEarlyMessages:
    """测试早期对话压缩"""

    @pytest.mark.asyncio
    @patch("app.services.consultation_service.call_qwen_chat", new_callable=AsyncMock)
    async def test_summarize_calls_llm(self, mock_qwen, sample_messages):
        """压缩函数调用 LLM 生成摘要"""
        from app.services.consultation_service import _summarize_early_messages

        mock_qwen.return_value = "【已暴露症状】头痛\n【否认症状】无"

        result = await _summarize_early_messages(sample_messages, "患者张三，头痛三天。")
        assert result == "【已暴露症状】头痛\n【否认症状】无"
        mock_qwen.assert_called_once()

    @pytest.mark.asyncio
    async def test_summarize_empty_messages_returns_empty(self):
        """空消息列表返回空字符串"""
        from app.services.consultation_service import _summarize_early_messages

        result = await _summarize_early_messages([], "患者信息")
        assert result == ""

    @pytest.mark.asyncio
    @patch("app.services.consultation_service.call_qwen_chat", new_callable=AsyncMock)
    async def test_summarize_llm_failure_returns_empty(self, mock_qwen, sample_messages):
        """LLM 调用失败时返回空字符串（降级）"""
        from app.services.consultation_service import _summarize_early_messages

        mock_qwen.side_effect = Exception("API error")

        result = await _summarize_early_messages(sample_messages, "患者信息")
        assert result == ""
