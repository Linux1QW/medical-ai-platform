"""
Patch starlette BaseHTTPMiddleware to prevent anyio MemoryObjectReceiveStream leaks.

Root cause: when an exception propagates through call_next() (e.g. from an endpoint
that raises an unhandled error), BaseHTTPMiddleware.__call__ does not call
response_sent.set(), so the background task close_recv_stream_on_response_sent is
cancelled before it can close recv_stream.  The orphaned stream is later detected
by gc / __del__ and emitted as a ResourceWarning, which pytest -W error promotes
to a test failure.

The patch makes two surgical changes to __call__:
  1. close_recv_stream_on_response_sent uses try/finally so recv_stream.close()  # type: ignore[attr-defined]
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
from starlette.types import Message, Receive, Scope, Send

T = typing.TypeVar("T")


async def _wrap_and_cancel(
    func: typing.Callable[[], typing.Awaitable[T]],
    task_group: anyio.abc.TaskGroup,
) -> T:
    """Wrap a callable to cancel the task group upon completion."""
    result = await func()
    task_group.cancel_scope.cancel()
    return result


async def _receive_or_disconnect(
    response_sent: anyio.Event,
    wrapped_receive: typing.Callable[..., typing.Awaitable[Message]],
) -> Message:
    """Return the next receive message or an http.disconnect if the response was sent."""
    if response_sent.is_set():
        return {"type": "http.disconnect"}

    async with anyio.create_task_group() as tg:
        tg.start_soon(_wrap_and_cancel, response_sent.wait, tg)
        message = await _wrap_and_cancel(wrapped_receive, tg)

    if response_sent.is_set():
        return {"type": "http.disconnect"}

    return message


async def _close_recv_stream_on_response_sent(
    response_sent: anyio.Event,
    recv_stream: ObjectReceiveStream[typing.MutableMapping[str, typing.Any]],
) -> None:
    """Wait for response_sent then close recv_stream (guaranteed via try/finally)."""
    try:
        await response_sent.wait()
    finally:
        recv_stream.close()  # type: ignore[attr-defined]


async def _send_no_error(
    send_stream: ObjectSendStream[typing.MutableMapping[str, typing.Any]],
    message: Message,
) -> None:
    """Send a message, silently ignoring BrokenResourceError."""
    try:
        await send_stream.send(message)
    except anyio.BrokenResourceError:
        return


async def _make_body_stream(
    recv_stream: ObjectReceiveStream[typing.MutableMapping[str, typing.Any]],
    app_exc: Exception | None,
) -> typing.AsyncGenerator[bytes, None]:
    """Read the response body from recv_stream, then re-raise app exception if any."""
    async with recv_stream:
        async for message in recv_stream:
            assert message["type"] == "http.response.body"
            body = message.get("body", b"")
            if body:
                yield body
            if not message.get("more_body", False):
                break

    if app_exc is not None:
        raise app_exc  # noqa: B904


def _build_call_next(
    app: typing.Any,
    scope: Scope,
    task_group: anyio.abc.TaskGroup,
    response_sent: anyio.Event,
    wrapped_receive: typing.Callable[..., typing.Awaitable[Message]],
) -> typing.Callable[..., typing.Awaitable[typing.Any]]:
    """Build a call_next closure bound to the current task group and scope."""

    async def call_next(request: typing.Any) -> typing.Any:
        app_exc: Exception | None = None
        send_stream: ObjectSendStream[typing.MutableMapping[str, typing.Any]]
        recv_stream: ObjectReceiveStream[typing.MutableMapping[str, typing.Any]]
        send_stream, recv_stream = anyio.create_memory_object_stream()

        async def coro() -> None:
            nonlocal app_exc
            async with send_stream:
                try:
                    await app(scope, receive_or_disconnect, send_no_error)
                except Exception as exc:
                    app_exc = exc

        receive_or_disconnect = lambda: _receive_or_disconnect(response_sent, wrapped_receive)  # noqa: E731
        send_no_error = lambda msg: _send_no_error(send_stream, msg)  # noqa: E731

        task_group.start_soon(_close_recv_stream_on_response_sent, response_sent, recv_stream)
        task_group.start_soon(coro)

        try:
            message = await recv_stream.receive()
            info = message.get("info", None)
            if message["type"] == "http.response.debug" and info is not None:
                message = await recv_stream.receive()
        except anyio.EndOfStream:
            if app_exc is not None:
                raise app_exc from None
            raise RuntimeError("No response returned.") from None

        assert message["type"] == "http.response.start"

        response = _StreamingResponse(
            status_code=message["status"],
            content=_make_body_stream(recv_stream, app_exc),
            info=info,
        )
        response.raw_headers = message["headers"]
        return response

    return call_next


async def _patched_call(self: BaseHTTPMiddleware, scope: Scope, receive: Receive, send: Send) -> None:
    """Patched BaseHTTPMiddleware.__call__ that prevents MemoryObjectReceiveStream leaks."""
    if scope["type"] != "http":
        await self.app(scope, receive, send)
        return

    request = _CachedRequest(scope, receive)
    wrapped_receive = request.wrapped_receive
    response_sent = anyio.Event()

    # FIX (1): _close_recv_stream_on_response_sent uses try/finally so recv_stream.close()
    #          runs even if the task is cancelled.
    # FIX (2): try/finally ensures response_sent.set() is always called.
    with collapse_excgroups():
        async with anyio.create_task_group() as task_group:
            call_next = _build_call_next(self.app, scope, task_group, response_sent, wrapped_receive)
            try:
                response = await self.dispatch_func(request, call_next)
                await response(scope, wrapped_receive, send)
            finally:
                response_sent.set()


# Apply the monkey-patch
BaseHTTPMiddleware.__call__ = _patched_call  # type: ignore[method-assign]
