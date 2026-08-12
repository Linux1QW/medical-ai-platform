# ADR-001: Celery Prefork Async Runtime — 主线程持久 Event Loop

## 状态

已接受（Accepted）

## 背景

`backend/app/tasks/evaluation_task.py` 原本每个 Celery task 调用 `asyncio.run()`，
但 SQLAlchemy async engine、Redis clients、OpenAI AsyncClient、embedding client 和
LangGraph checkpointer 都是进程级单例对象。`asyncio.run()` 在返回时会关闭事件循环，
导致第二个任务复用这些绑定到旧 loop 的对象时出现：

- `Event loop is closed`
- 连接池跨 loop 失效
- Redis checkpoint 连接不可用

之前只对 checkpointer 做了逐任务重建（`_ensure_worker_checkpointer()`），未覆盖其他异步资源。

## 决策

每个 Celery prefork 子进程在 `worker_process_init` 信号处理中，于当前主线程创建一个
**唯一的事件循环**（`asyncio.new_event_loop()`），由 `WorkerAsyncRuntime` 持有。

同步 Celery task 在同一线程通过 `loop.run_until_complete(asyncio.wait_for(task, timeout))`
执行异步 coroutine。

### 关键约束

1. **不使用 `run_coroutine_threadsafe()`**：prefork 子进程一次只执行一个 task，
   不需要额外 daemon thread，所有操作在主线程完成。
2. **PID/线程所有权校验**：`.run()` 检测 PID 和 thread ID 不匹配时立即抛
   `WorkerRuntimeOwnershipError`。
3. **Task 清理**：正常/timeout/exception 退出时，如 task 未完成则 cancel 并
   `run_until_complete(gather(task, return_exceptions=True))`，确保下一个 Celery task
   不继承悬挂 coroutine。
4. **Hard kill 处理**：`task_acks_late=True` + `task_reject_on_worker_lost=True` +
   `visibility_timeout=900`，确保 worker 崩溃时未 ack 任务可重投。

### Linux prefork 与 Windows 边界

- **Linux (production)**：Celery prefork 使用 `os.fork()`，子进程继承父进程内存但
  事件循环不可跨 fork 共享。`worker_process_init` 中清空父进程继承的 client 引用，
  在子进程 runtime 内懒初始化。
- **Windows (local dev)**：Windows 不支持 `os.fork()`，Celery 使用 `spawn` 启动子进程。
  本地调试不走 prefork 路径，Windows 原生调试不进入此发布门禁。
- **Docker/WSL2**：production 使用 Docker Linux 容器，WSL2 开发环境同 Linux 行为。

### 资源配置

- `LLM_CACHE_REDIS_URL=redis://localhost:6380/0`（独立 redis-cache 实例）
- `RETRIEVAL_CACHE_REDIS_URL=redis://localhost:6380/1`
- `REDIS_CHECKPOINT_URL=redis://localhost:6379/0`（独立 redis-state 实例；RedisVL/RediSearch 要求 DB 0）
- 本地 6379 保留给 redis-state（broker/backend/checkpoint/control），6380 对应独立 redis-cache。

### Shutdown 顺序

`close_worker_resources()` 按以下顺序关闭：
1. Graph（清除编译缓存）
2. Checkpointer（关闭 Redis 连接）
3. HTTP clients（Qwen AsyncOpenAI, embedding client）
4. Redis clients（LLM cache, retrieval cache, evaluation control）
5. DB engine（SQLAlchemy async engine）

## 回退方案

若 staging soak test（100 次顺序 coroutine）不通过：

1. **回退到 `EvaluationTaskResourceScope`**：每个任务创建并关闭独立的 async 资源
   （engine/clients/checkpointer），但保持同一 event loop。
2. **不可回退到**：每任务 `asyncio.run()` + 全局 client 的模式，这是当前的 bug 来源。

## 后果

### 正面

- 所有异步资源在同一 event loop 中创建和使用，无跨 loop 失效问题
- 消除了逐任务重建 checkpointer 的开销
- Late ack + visibility timeout 保证任务不丢失
- Worker 崩溃后任务自动重投

### 负面

- 需要确保所有 async 资源在 worker init 时初始化或懒初始化
- Windows 本地调试无法测试 prefork 路径
- 如果 event loop 意外被关闭，所有资源失效

### 中性

- `worker_pool` 固定为 `prefork`，不接受 solo/threads/gevent/eventlet
- 新增两个 Redis 实例配置（LLM_CACHE_REDIS_URL, RETRIEVAL_CACHE_REDIS_URL）
