"""
Patch starlette BaseHTTPMiddleware to prevent anyio MemoryObjectReceiveStream leaks.

Root cause: when an exception propagates through call_next() (e.g. from an endpoint
that raises an unhandled error), BaseHTTPMiddleware.__call__ does not call
response_sent.set(), so the background task close_recv_stream_on_response_sent is
cancelled before it can close recv_stream.  The orphaned stream is later detected
by gc / __del__ and emitted as a ResourceWarning, which pytest -W error promotes
to a test failure.

The patch makes two surgical changes to __call__:
  1. close_recv_stream_on_response_sent uses try/finally so recv_stream.close()
     runs even if the task is cancelled.
  2. response_sent.set() is moved into a finally block so it is always called,
     allowing the cleanup task to finish before the task group exits.

Import this module **before** any @app.middleware("http") decorator runs —
app/main.py does this at the top.
"""

from __future__ import annotations

import typing

import anyio
from anyio.abc import ObjectReceiveStream, ObjectSendStream

from starlette._utils import collapse_excgroups
from starlette.middleware.base import BaseHTTPMiddleware, _CachedRequest, _StreamingResponse
from starlette.requests import Request
from starlette.types import ASGIApp, Message, Receive, Scope, Send

T = typing.TypeVar("T")
RequestResponseEndpoint = typing.Callable[[Request], typing.Awaitable[typing.Any]]


async def _patched_call(self: BaseHTTPMiddleware, scope: Scope, receive: Receive, send: Send) -> None:
    if scope["type"] != "http":
        await self.app(scope, receive, send)
        return

    request = _CachedRequest(scope, receive)
    wrapped_receive = request.wrapped_receive
    response_sent = anyio.Event()

    async def call_next(request: Request) -> typing.Any:
        app_exc: Exception | None = None
        send_stream: ObjectSendStream[typing.MutableMapping[str, typing.Any]]
        recv_stream: ObjectReceiveStream[typing.MutableMapping[str, typing.Any]]
        send_stream, recv_stream = anyio.create_memory_object_stream()

        async def receive_or_disconnect() -> Message:
            if response_sent.is_set():
                return {"type": "http.disconnect"}

            async with anyio.create_task_group() as task_group:

                async def wrap(func: typing.Callable[[], typing.Awaitable[T]]) -> T:
                    result = await func()
                    task_group.cancel_scope.cancel()
                    return result

                task_group.start_soon(wrap, response_sent.wait)
                message = await wrap(wrapped_receive)

            if response_sent.is_set():
                return {"type": "http.disconnect"}

            return message

        # FIX (1): try/finally ensures recv_stream.close() runs even on cancellation
        async def close_recv_stream_on_response_sent() -> None:
            try:
                await response_sent.wait()
            finally:
                recv_stream.close()

        async def send_no_error(message: Message) -> None:
            try:
                await send_stream.send(message)
            except anyio.BrokenResourceError:
                return

        async def coro() -> None:
            nonlocal app_exc
            async with send_stream:
                try:
                    await self.app(scope, receive_or_disconnect, send_no_error)
                except Exception as exc:
                    app_exc = exc

        task_group.start_soon(close_recv_stream_on_response_sent)
        task_group.start_soon(coro)

        try:
            message = await recv_stream.receive()
            info = message.get("info", None)
            if message["type"] == "http.response.debug" and info is not None:
                message = await recv_stream.receive()
        except anyio.EndOfStream:
            if app_exc is not None:
                raise app_exc
            raise RuntimeError("No response returned.")

        assert message["type"] == "http.response.start"

        async def body_stream() -> typing.AsyncGenerator[bytes, None]:
            async with recv_stream:
                async for message in recv_stream:
                    assert message["type"] == "http.response.body"
                    body = message.get("body", b"")
                    if body:
                        yield body
                    if not message.get("more_body", False):
                        break

            if app_exc is not None:
                raise app_exc

        response = _StreamingResponse(status_code=message["status"], content=body_stream(), info=info)
        response.raw_headers = message["headers"]
        return response

    # FIX (2): try/finally ensures response_sent.set() is always called
    with collapse_excgroups():
        async with anyio.create_task_group() as task_group:
            try:
                response = await self.dispatch_func(request, call_next)
                await response(scope, wrapped_receive, send)
            finally:
                response_sent.set()


# Apply the monkey-patch
BaseHTTPMiddleware.__call__ = _patched_call  # type: ignore[assignment]
