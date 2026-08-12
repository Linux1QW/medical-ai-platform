"""Regression tests for structural/live Coach gate runtime wiring."""
from __future__ import annotations

from types import SimpleNamespace

import pytest

from scripts.ci import v12_coach_gate


@pytest.mark.asyncio
async def test_live_gate_uses_factory_and_closes_checkpointer(monkeypatch) -> None:
    calls: list[str] = []
    graph = object()
    expected_results = [object()]

    async def fake_init() -> None:
        calls.append("init")

    async def fake_close() -> None:
        calls.append("close")

    async def fake_create(self) -> SimpleNamespace:
        calls.append("factory")
        return SimpleNamespace(graph=graph)

    async def fake_evaluate(cases, *, graph: object):
        calls.append("evaluate")
        assert cases == ["case"]
        assert graph is not None
        return expected_results

    monkeypatch.setattr("app.orchestration.checkpointer.init_checkpointer", fake_init)
    monkeypatch.setattr("app.orchestration.checkpointer.close_checkpointer", fake_close)
    monkeypatch.setattr(
        "app.services.coach_runtime_factory.CoachRuntimeFactory.create",
        fake_create,
    )
    monkeypatch.setattr(v12_coach_gate, "evaluate_all_async", fake_evaluate)

    results = await v12_coach_gate._evaluate_live_cases(["case"])

    assert results == expected_results
    assert calls == ["init", "factory", "evaluate", "close"]


@pytest.mark.asyncio
async def test_live_gate_closes_checkpointer_when_evaluation_fails(monkeypatch) -> None:
    closed = False

    async def fake_init() -> None:
        return None

    async def fake_close() -> None:
        nonlocal closed
        closed = True

    async def fake_create(self) -> SimpleNamespace:
        return SimpleNamespace(graph=object())

    async def fake_evaluate(cases, *, graph: object):
        raise RuntimeError("benchmark failed")

    monkeypatch.setattr("app.orchestration.checkpointer.init_checkpointer", fake_init)
    monkeypatch.setattr("app.orchestration.checkpointer.close_checkpointer", fake_close)
    monkeypatch.setattr(
        "app.services.coach_runtime_factory.CoachRuntimeFactory.create",
        fake_create,
    )
    monkeypatch.setattr(v12_coach_gate, "evaluate_all_async", fake_evaluate)

    with pytest.raises(RuntimeError, match="benchmark failed"):
        await v12_coach_gate._evaluate_live_cases(["case"])

    assert closed is True
