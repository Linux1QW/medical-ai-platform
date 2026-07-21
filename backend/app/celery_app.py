"""Celery 应用实例配置"""

from celery import Celery

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
]
