import inspect
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock

import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[2]


def test_backend_runtime_configuration_supports_app_imports_and_celery():
    dockerfile = (PROJECT_ROOT / "Dockerfile.backend").read_text(encoding="utf-8")
    setup_py = (PROJECT_ROOT / "backend" / "setup.py").read_text(encoding="utf-8")
    compose = (PROJECT_ROOT / "docker-compose.yml").read_text(encoding="utf-8")

    production_config = dockerfile.split("FROM python:3.10-slim AS base", 1)[1]
    assert "PYTHONPATH=/app/backend" in production_config
    assert '"celery[redis]==5.6.3"' in setup_py
    assert compose.count("celery -A app.celery_app") == 2


def test_patient_export_route_is_registered_before_dynamic_patient_route():
    from app.api.v1.patients import router

    paths = [route.path for route in router.routes if hasattr(route, "path")]
    assert paths.index("/export") < paths.index("/{patient_id}")


def test_virtual_patient_has_stable_case_id_mapping():
    from app.models.patient import VirtualPatient

    assert "case_id" in VirtualPatient.__table__.columns
    assert any(
        index.columns.keys() == ["case_id"] and index.unique
        for index in VirtualPatient.__table__.indexes
    )


@pytest.mark.asyncio
async def test_case_recommender_attaches_database_patient_ids():
    from app.services.case_recommender import _attach_patient_db_ids

    result = MagicMock()
    result.all.return_value = [(42, "patient100_21")]
    db = MagicMock()
    db.execute = AsyncMock(return_value=result)
    cases = [{"case_id": "patient100_21"}, {"case_id": "patient101_21"}]

    await _attach_patient_db_ids(db, cases)

    assert cases[0]["patient_db_id"] == 42
    assert cases[1]["patient_db_id"] is None


def test_consultation_queries_can_request_row_locks():
    from app.services.consultation_service import get_consultation

    assert "for_update" in inspect.signature(get_consultation).parameters
