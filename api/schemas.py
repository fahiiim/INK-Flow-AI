"""Transport-specific response schemas for the HTTP API."""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field


class HealthResponse(BaseModel):
    """Service health or readiness response."""

    model_config = ConfigDict(extra="forbid")

    status: Literal["ok", "ready"] = Field(
        description="Current service state.",
    )
    service: str = Field(
        description="Stable service identifier.",
    )
    version: str = Field(
        description="Deployed API version.",
    )


class ErrorResponse(BaseModel):
    """Standard API error response."""

    model_config = ConfigDict(extra="forbid")

    detail: str = Field(
        description="Safe, client-facing error explanation.",
    )
