"""Celery 应用实例配置"""

from types import SimpleNamespace
from uuid import uuid4

try:
    from celery import Celery
    from celery.signals import worker_process_init, worker_process_shutdown
except ModuleNotFoundError:  # pragma: no cover - only used by dependency-light tests
    class _Signal:
        def connect(self, func):
            return func

    class _FallbackTask:
        def __init__(self, func, *, bind=False, name=None):
            self.func = func
            self.bind = bind
            self.name = name or func.__name__
            self.delay = self._delay

        def _delay(self, *args, **kwargs):
            return SimpleNamespace(
                id=uuid4().hex,
                state="PENDING",
                status="PENDING",
                info=None,
                result=None,
            )

        def __call__(self, *args, **kwargs):
            if self.bind:
                return self.func(self, *args, **kwargs)
            return self.func(*args, **kwargs)

        def run(self, *args, **kwargs):
            return self(*args, **kwargs)

    class _FallbackConf(SimpleNamespace):
        def update(self, **values):
            for key, value in values.items():
                setattr(self, key, value)

    class Celery:  # type: ignore[no-redef]
        def __init__(self, *_args, **_kwargs):
            self.conf = _FallbackConf(
                beat_schedule={},
                include=[],
            )
            self.control = SimpleNamespace(revoke=lambda *_a, **_kw: None)

        def task(self, *, bind=False, name=None, **_kwargs):
            def decorate(func):
                return _FallbackTask(func, bind=bind, name=name)

            return decorate

        def autodiscover_tasks(self, *_args, **_kwargs):
            return None

        def AsyncResult(self, task_id):
            return AsyncResult(task_id, app=self)

    class AsyncResult:  # type: ignore[no-redef]
        def __init__(self, task_id, app=None):
            self.id = task_id
            self.state = "PENDING"
            self.status = "PENDING"
            self.info = None
            self.result = None

    worker_process_init = _Signal()
    worker_process_shutdown = _Signal()

from app.core.config import settings

celery_app = Celery(
    "medical_ai",
    broker=settings.CELERY_BROKER_URL,
    backend=settings.CELERY_RESULT_BACKEND,
)

celery_app.conf.update(
    task_serializer="json",
    accept_content=["json"],
    result_serializer="json",
    timezone="Asia/Shanghai",
    enable_utc=True,
    task_track_started=True,
    task_time_limit=600,        # 10 分钟硬超时
    task_soft_time_limit=300,   # 5 分钟软超时
    worker_prefetch_multiplier=1,  # 每个 worker 预取 1 个任务
)

# Beat 定时任务调度
celery_app.conf.beat_schedule = {
    "cleanup-expired-records": {
        "task": "cleanup_expired_records",
        "schedule": 86400.0,  # 每天执行一次
    },
}

# 自动发现 tasks 模块
celery_app.autodiscover_tasks(["app"])

# 显式 include（autodiscover 不会遍历 app/tasks/ 子包内的模块）
celery_app.conf.include = [
    "app.tasks.data_cleanup",
    "app.tasks.evaluation_task",
    "app.tasks.rag_index_task",
]


@worker_process_init.connect
def _start_rag_generation_listener(**_: object) -> None:
    """Subscribe each forked Worker process to generation switch events."""
    from app.services.rag.indexing.versioning import start_index_switch_listener

    start_index_switch_listener()


@worker_process_shutdown.connect
def _stop_rag_generation_listener(**_: object) -> None:
    from app.services.rag.indexing.versioning import stop_index_switch_listener

    stop_index_switch_listener()
