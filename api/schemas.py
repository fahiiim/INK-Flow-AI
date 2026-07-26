"""Transport-specific response schemas for the HTTP API."""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, StrictStr, model_validator

from ai_brain.schemas import AIExtractionOutput, Message, TattooInquiryInput

__all__ = [
    "AIExtractionOutput",
    "ErrorResponse",
    "HealthResponse",
    "TattooInquiryInput",
    "TelegramSummaryInput",
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


class TelegramSummaryInput(TattooInquiryInput):
    """Summary input with compatibility for the legacy backend payload."""

    current_message: str = Field(
        default="",
        description=(
            "Latest client text, or the latest user history message when "
            "omitted."
        ),
    )

    @model_validator(mode="before")
    @classmethod
    def normalize_legacy_payload(cls, value: Any) -> Any:
        """Map legacy summary keys and infer the latest user message."""
        if not isinstance(value, dict):
            return value

        data = dict(value)
        current_message = data.get("current_message")
        for alias in ("message", "text"):
            alias_value = data.pop(alias, None)
            if not current_message and alias_value:
                current_message = alias_value
        if current_message is not None:
            data["current_message"] = current_message

        history = data.get("recent_chat_history")
        for alias in ("chat_history", "history"):
            alias_value = data.pop(alias, None)
            if history is None and alias_value is not None:
                history = alias_value
        if history is not None:
            data["recent_chat_history"] = history

        lead = data.pop("lead", None)
        if "existing_db_state" not in data and lead is not None:
            data["existing_db_state"] = lead

        if not data.get("current_message") and isinstance(history, (list, tuple)):
            for item in reversed(history):
                if isinstance(item, Message):
                    role = item.role
                    content = item.content
                elif isinstance(item, dict):
                    role = item.get("role")
                    content = item.get("content")
                else:
                    continue
                if role == "user" and isinstance(content, str) and content.strip():
                    data["current_message"] = content
                    break

        data.setdefault("current_message", "")
        return data


class TelegramSummaryResponse(BaseModel):
    """Staff-facing high-risk summary and client draft reply."""

    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)

    risk_level: Literal["high"]
    summary: StrictStr = Field(min_length=1, max_length=5000)
    draft_reply: StrictStr = Field(min_length=1, max_length=2000)
    telegram_message: StrictStr = Field(min_length=1, max_length=7000)
