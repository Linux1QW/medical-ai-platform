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

    class AsyncResult:
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
    # Late ack：任务完成后才确认，worker 崩溃时任务可重投
    task_acks_late=True,
    # worker 崩溃时拒绝任务，让 broker 重新分配
    task_reject_on_worker_lost=True,
    # Broker visibility timeout（秒）：worker 崩溃后多久重新分发
    broker_transport_options={"visibility_timeout": 900},
    # 结果过期时间（秒）：24小时
    result_expires=86400,
    # 固定使用 prefork pool
    worker_pool="prefork",
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
def _init_worker_runtime(**_: object) -> None:
    """Initialize the per-prefork-child async runtime and shared resources.

    Each forked Celery worker process gets its own event loop owned by the
    main thread. Process-level async resources (checkpointer, graph, HTTP
    clients, Redis clients, DB engine) are cleared from parent-inherited
    references and will be lazily re-initialized inside the child runtime.
    """
    import logging
    logger = logging.getLogger(__name__)

    # 1. Clear parent-inherited singleton references (stale across fork)
    _clear_parent_inherited_singletons()

    # 2. Start the async runtime for this child process
    from app.tasks.async_runtime import get_worker_runtime
    rt = get_worker_runtime()
    rt.start()

    # 3. Initialize checkpointer + graph in the child runtime's loop
    from app.core.config import settings
    if settings.LANGGRAPH_ENABLED:
        from app.orchestration.checkpointer import init_checkpointer
        from app.orchestration.graph import get_graph

        async def _init_graph_resources():
            await init_checkpointer(
                redis_url=settings.REDIS_CHECKPOINT_URL,
                ttl=settings.REDIS_CHECKPOINT_TTL,
            )
            await get_graph()

        rt.run(_init_graph_resources())

    # 4. Start the RAG generation switch listener
    from app.services.rag.indexing.versioning import start_index_switch_listener
    start_index_switch_listener()

    logger.info("Worker process initialized: async runtime started")


@worker_process_shutdown.connect
def _shutdown_worker_runtime(**_: object) -> None:
    """Shut down the per-prefork-child async runtime and close all resources."""
    import logging
    logger = logging.getLogger(__name__)

    # 1. Stop the RAG generation switch listener
    from app.services.rag.indexing.versioning import stop_index_switch_listener
    stop_index_switch_listener()

    # 2. Stop the runtime and close all resources
    from app.tasks.async_runtime import get_worker_runtime
    from app.tasks.worker_resources import close_worker_resources

    rt = get_worker_runtime()
    rt.stop(close_resources=close_worker_resources)

    logger.info("Worker process shut down: async runtime stopped")


def _clear_parent_inherited_singletons() -> None:
    """Clear parent-process singleton references that are stale after fork.

    After os.fork(), the child inherits module-level singletons (async clients,
    DB engines, etc.) that are bound to the parent's event loop. These must be
    cleared so the child can lazily re-initialize them in its own loop.
    """
    # Clear checkpointer / graph
    try:
        from app.orchestration import checkpointer as cp_mod
        cp_mod._checkpointer = None
        cp_mod._exit_stack = None
    except Exception:
        pass

    try:
        from app.orchestration import graph as graph_mod
        graph_mod._compiled_graph = None
    except Exception:
        pass

    # Clear LLM cache Redis
    try:
        from app.services import llm_cache as lc_mod
        lc_mod._redis_client = None
    except Exception:
        pass

    # Clear retrieval cache Redis
    try:
        from app.services.rag import retrieval_cache as rc_mod
        rc_mod._redis_client = None
    except Exception:
        pass

    # Clear evaluation control Redis
    try:
        from app.services import evaluation_cancel as ec_mod
        ec_mod._control_redis = None
    except Exception:
        pass

    # Clear Qwen client
    try:
        from app.services import qwen_client as qc_mod
        qc_mod.client = None
        qc_mod._active_adapter = None
        qc_mod._active_model = None
        qc_mod._semaphore = None
    except Exception:
        pass

    # Clear DB engine
    try:
        from app.db import session as sess_mod
        if hasattr(sess_mod, '_engine'):
            sess_mod._engine = None
    except Exception:
        pass
