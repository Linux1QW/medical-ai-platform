"""Task 7 contracts for the asynchronous knowledge-base API."""

from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

import pytest
from fastapi.testclient import TestClient

from app.api.v1 import knowledge_base
from app.core.deps import get_current_admin
from app.main import app


@pytest.fixture
def client():
    app.dependency_overrides[get_current_admin] = lambda: SimpleNamespace(
        id=1, role="admin"
    )
    try:
        yield TestClient(app, raise_server_exceptions=False)
    finally:
        app.dependency_overrides.pop(get_current_admin, None)


@pytest.fixture
def admin_headers():
    return {}


def test_rebuild_returns_task_id(client, admin_headers, monkeypatch):
    from app.tasks import rag_index_task

    task = Mock()
    task.return_value.id = "task-1"
    monkeypatch.setattr(rag_index_task.rebuild_rag_index, "delay", task)

    response = client.post(
        "/api/v1/knowledge-base/rebuild", headers=admin_headers
    )

    assert response.status_code == 202
    assert response.json()["task_id"] == "task-1"


@pytest.mark.parametrize("force_replace", [False, True])
def test_add_pdf_returns_task_id(
    client, admin_headers, monkeypatch, tmp_path, force_replace
):
    pdf = tmp_path / "guide.pdf"
    pdf.write_bytes(b"%PDF-1.7")
    monkeypatch.setattr(knowledge_base, "PDF_DIR", tmp_path)
    from app.tasks import rag_index_task

    task = Mock()
    task.return_value.id = "task-add"
    task_function = (
        rag_index_task.replace_rag_index
        if force_replace
        else rag_index_task.add_rag_index
    )
    monkeypatch.setattr(task_function, "delay", task)

    response = client.post(
        "/api/v1/knowledge-base/add-pdf",
        json={"filename": pdf.name, "force_replace": force_replace},
        headers=admin_headers,
    )

    assert response.status_code == 202
    assert response.json()["task_id"] == "task-add"
    task.assert_called_once()
    assert Path(task.call_args.args[0]).resolve() == pdf.resolve()
    assert task.call_args.kwargs["source_name"].startswith("source-")


def test_delete_source_returns_task_id(client, admin_headers, monkeypatch):
    from app.tasks import rag_index_task

    task = Mock()
    task.return_value.id = "task-delete"
    monkeypatch.setattr(rag_index_task.delete_rag_index, "delay", task)

    response = client.delete(
        "/api/v1/knowledge-base/sources/source-123", headers=admin_headers
    )

    assert response.status_code == 202
    assert response.json()["task_id"] == "task-delete"
    task.assert_called_once_with("source-123")


@pytest.mark.parametrize("filename", ["../outside.pdf", "C:\\outside.pdf"])
def test_add_pdf_rejects_traversal_and_absolute_path(
    client, admin_headers, monkeypatch, tmp_path, filename
):
    monkeypatch.setattr(knowledge_base, "PDF_DIR", tmp_path)
    from app.tasks import rag_index_task

    task = Mock()
    monkeypatch.setattr(rag_index_task.add_rag_index, "delay", task)

    response = client.post(
        "/api/v1/knowledge-base/add-pdf",
        json={"filename": filename},
        headers=admin_headers,
    )

    assert response.status_code == 400
    task.assert_not_called()


def test_rebuild_status_reads_celery_result_backend(client, admin_headers, monkeypatch):
    from app.celery_app import celery_app

    result = Mock()
    result.state = "PROGRESS"
    result.info = {
        "phase": "embed",
        "generation": "rag-20260808112233-01234567",
    }
    result.result = result.info
    async_result = Mock(return_value=result)
    monkeypatch.setattr(knowledge_base, "AsyncResult", async_result, raising=False)

    response = client.get(
        "/api/v1/knowledge-base/rebuild/status?task_id=task-1",
        headers=admin_headers,
    )

    assert response.status_code == 200
    assert response.json()["state"] == "PROGRESS"
    assert response.json()["phase"] == "embed"
    async_result.assert_called_once_with("task-1", app=celery_app)


def test_api_has_no_process_local_rebuild_state():
    assert not hasattr(knowledge_base, "_rebuild_lock")
    assert not hasattr(knowledge_base, "_rebuild_status")
