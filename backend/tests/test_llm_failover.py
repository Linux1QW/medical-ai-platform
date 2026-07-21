"""LLM Failover Manager 测试"""
import json
import pytest
import threading
from unittest.mock import patch, MagicMock

# 测试用配置
MOCK_PROVIDERS = [
    {"name": "primary", "api_key": "key1", "base_url": "url1", "model": "model1"},
    {"name": "secondary", "api_key": "key2", "base_url": "url2", "model": "model2"},
    {"name": "tertiary", "api_key": "key3", "base_url": "url3", "model": "model3"},
]


def _create_manager(providers=None, threshold=3):
    """创建测试用 Manager 实例"""
    from app.services.llm_failover import LLMFailoverManager
    with patch("app.services.llm_failover.settings") as mock_settings:
        mock_settings.LLM_PROVIDERS = json.dumps(providers or MOCK_PROVIDERS)
        mock_settings.LLM_CIRCUIT_BREAKER_THRESHOLD = threshold
        mock_settings.QWEN_API_BASE_URL = "default_url"
        mock_settings.QWEN_MODEL = "default_model"
        mock_settings.get_llm_providers.return_value = providers or MOCK_PROVIDERS
        return LLMFailoverManager()


# ── 初始化 ──────────────────────────────────────────────

class TestLLMFailoverManagerInit:
    def test_single_provider(self):
        manager = _create_manager(providers=[MOCK_PROVIDERS[0]])
        assert manager.get_current_provider()["name"] == "primary"

    def test_multiple_providers(self):
        manager = _create_manager()
        assert manager.get_current_provider()["name"] == "primary"

    def test_default_threshold(self):
        manager = _create_manager(threshold=5)
        assert manager._circuit_breaker_threshold == 5

    def test_provider_count(self):
        manager = _create_manager()
        assert manager.provider_count == 3

    def test_lock_is_rlock(self):
        """验证使用 RLock 而非 Lock，防止死锁"""
        manager = _create_manager()
        assert isinstance(manager._lock, type(threading.RLock()))


# ── 失败报告 ────────────────────────────────────────────

class TestReportFailure:
    def test_failure_count_increments(self):
        manager = _create_manager()
        manager.report_failure()
        assert manager._failure_counts[0] == 1
        manager.report_failure()
        assert manager._failure_counts[0] == 2

    def test_failure_count_per_provider(self):
        manager = _create_manager()
        manager.report_failure()
        manager.switch_to_next()  # 切换时重置 primary 的计数
        assert manager._failure_counts[0] == 0  # primary 被重置
        manager.report_failure()
        assert manager._failure_counts[1] == 1  # secondary 新计数


# ── 成功报告 ────────────────────────────────────────────

class TestReportSuccess:
    def test_success_resets_failure_count(self):
        manager = _create_manager()
        manager.report_failure()
        manager.report_failure()
        manager.report_success()
        assert manager._failure_counts.get(0, 0) == 0

    def test_success_on_zero_is_noop(self):
        manager = _create_manager()
        manager.report_success()  # 计数已经为 0，不应报错
        assert manager._failure_counts[0] == 0


# ── 切换判断 ────────────────────────────────────────────

class TestShouldSwitch:
    def test_below_threshold(self):
        manager = _create_manager(threshold=3)
        manager.report_failure()
        manager.report_failure()
        assert not manager.should_switch()

    def test_at_threshold(self):
        manager = _create_manager(threshold=3)
        manager.report_failure()
        manager.report_failure()
        manager.report_failure()
        assert manager.should_switch()

    def test_above_threshold(self):
        manager = _create_manager(threshold=2)
        manager.report_failure()
        manager.report_failure()
        manager.report_failure()
        assert manager.should_switch()


# ── 切换逻辑 ────────────────────────────────────────────

class TestSwitchToNext:
    def test_switch_to_next_provider(self):
        manager = _create_manager()
        result = manager.switch_to_next()
        assert result["name"] == "secondary"

    def test_switch_wraps_around(self):
        """从最后一个切回第一个"""
        manager = _create_manager()
        manager.switch_to_next()  # → secondary
        manager.switch_to_next()  # → tertiary
        result = manager.switch_to_next()  # → primary
        assert result["name"] == "primary"

    def test_switch_skips_circuit_broken_provider(self):
        manager = _create_manager(threshold=2)
        manager.switch_to_next()  # → secondary
        manager.report_failure()
        manager.report_failure()
        # secondary 已熔断，应跳到 tertiary
        result = manager.switch_to_next()
        assert result["name"] == "tertiary"

    def test_all_circuited_resets(self):
        manager = _create_manager(threshold=1)
        manager.report_failure()
        manager.switch_to_next()  # → secondary
        manager.report_failure()
        manager.switch_to_next()  # → tertiary
        manager.report_failure()
        # 所有熔断，应重置并返回非 None
        result = manager.switch_to_next()
        assert result is not None
        assert result["name"] == "primary"  # 重置回第一个

    def test_switch_resets_old_provider_failure_count(self):
        """切换后旧 provider 的失败计数被清零"""
        manager = _create_manager(threshold=5)
        manager.report_failure()
        manager.report_failure()
        assert manager._failure_counts[0] == 2
        manager.switch_to_next()
        assert manager._failure_counts[0] == 0  # 旧 provider 被重置


# ── 死锁回归（最关键） ──────────────────────────────────

class TestDeadlockRegression:
    def test_switch_deadlock_regression(self):
        """验证 switch_to_next() 内部调用 get_current_provider() 不会死锁。

        如果 _lock 是 threading.Lock（不可重入），同一线程在持有锁时再次
        获取锁会永久阻塞。使用 RLock 后此问题消除。
        """
        manager = _create_manager(threshold=1)
        manager.report_failure()

        result_holder = [None]
        error_holder = [None]

        def _switch():
            try:
                result_holder[0] = manager.switch_to_next()
            except Exception as e:
                error_holder[0] = e

        t = threading.Thread(target=_switch)
        t.start()
        t.join(timeout=5)  # 5 秒超时，死锁时会触发

        assert not t.is_alive(), "switch_to_next() 死锁了！线程未能在 5s 内完成"
        assert error_holder[0] is None, f"switch_to_next() 抛出异常: {error_holder[0]}"
        assert result_holder[0] is not None
        assert result_holder[0]["name"] == "secondary"

    def test_repeated_switch_no_deadlock(self):
        """多次连续切换不应死锁"""
        manager = _create_manager(threshold=1)
        for _ in range(10):
            manager.report_failure()
            result = manager.switch_to_next()
            assert result is not None


# ── 状态查询 ────────────────────────────────────────────

class TestGetStatus:
    def test_status_structure(self):
        manager = _create_manager()
        status = manager.get_status()
        assert "current_index" in status
        assert "current_provider" in status
        assert "failure_counts" in status
        assert "threshold" in status
        assert "total_providers" in status

    def test_status_values(self):
        manager = _create_manager(threshold=5)
        status = manager.get_status()
        assert status["current_index"] == 0
        assert status["current_provider"] == "primary"
        assert status["threshold"] == 5
        assert status["total_providers"] == 3
        assert status["failure_counts"] == {0: 0, 1: 0, 2: 0}

    def test_status_after_failure(self):
        manager = _create_manager()
        manager.report_failure()
        manager.report_failure()
        status = manager.get_status()
        assert status["failure_counts"][0] == 2


# ── 线程安全 ────────────────────────────────────────────

class TestThreadSafety:
    def test_concurrent_failures(self):
        """10 线程 × 100 次 report_failure，验证计数不丢失"""
        manager = _create_manager(threshold=10000)
        barrier = threading.Barrier(10)

        def _worker():
            barrier.wait()  # 确保所有线程同时开始
            for _ in range(100):
                manager.report_failure()

        threads = [threading.Thread(target=_worker) for _ in range(10)]
        for t in threads:
            t.start()
        for t in threads:
            t.join(timeout=10)

        assert manager._failure_counts[0] == 1000  # 10 × 100

    def test_concurrent_switch_no_deadlock(self):
        """多线程同时调用 switch_to_next 不应死锁"""
        manager = _create_manager(threshold=2)
        errors = []

        def _worker():
            try:
                for _ in range(20):
                    manager.report_failure()
                    manager.switch_to_next()
            except Exception as e:
                errors.append(e)

        threads = [threading.Thread(target=_worker) for _ in range(5)]
        for t in threads:
            t.start()
        for t in threads:
            t.join(timeout=10)

        alive = [t for t in threads if t.is_alive()]
        assert not alive, f"{len(alive)} 个线程死锁"
        assert not errors, f"出现异常: {errors}"
