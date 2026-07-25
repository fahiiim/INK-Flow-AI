"""Transport-specific response schemas for the HTTP API."""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, StrictStr

from ai_brain.schemas import AIExtractionOutput, TattooInquiryInput

__all__ = [
    "AIExtractionOutput",
    "ErrorResponse",
    "HealthResponse",
    "TattooInquiryInput",
    "TelegramSummaryResponse",
]


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


class TelegramSummaryResponse(BaseModel):
    """Staff-facing high-risk summary and client draft reply."""

    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)

    risk_level: Literal["high"]
    summary: StrictStr = Field(min_length=1, max_length=5000)
    draft_reply: StrictStr = Field(min_length=1, max_length=2000)
    telegram_message: StrictStr = Field(min_length=1, max_length=7000)
