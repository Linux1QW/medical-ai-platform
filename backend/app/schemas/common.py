"""Common schema for V1.1 error responses."""
from typing import Any
from pydantic import BaseModel


class ErrorOut(BaseModel):
    """Standardized error response body."""
    error_code: str
    message: str
    detail: str
    request_id: str
    error_type: str | None = None
    context: dict[str, Any] | None = None
