"""Strict data contracts for the Tattoo Studio AI Brain."""

from __future__ import annotations

from datetime import date as calendar_date
from datetime import datetime
from typing import Annotated, Any, Literal, Self

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    StringConstraints,
    field_validator,
    model_validator,
)

STYLE_TAG_OPTIONS: tuple[str, ...] = (
    "fine-line",
    "watercolor",
    "minimal",
    "floral",
    "micro-realism",
    "black-and-grey",
    "calligraphy",
    "traditional",
    "geometric",
    "unknown",
)

MISSING_INFORMATION_OPTIONS: tuple[str, ...] = (
    "tattoo idea",
    "size in cm",
    "placement",
    "reference images",
    "tattoo style",
    "color preference",
    "preferred date",
    "preferred time",
)

MESSAGE_SOURCE_OPTIONS: tuple[str, ...] = (
    "whatsapp",
    "outlook",
    "vcita",
    "other",
)
_MESSAGE_SOURCE_ALIASES = {
    "email": "outlook",
    "microsoft outlook": "outlook",
    "ms outlook": "outlook",
}

StyleTag = Literal[
    "fine-line",
    "watercolor",
    "minimal",
    "floral",
    "micro-realism",
    "black-and-grey",
    "calligraphy",
    "traditional",
    "geometric",
    "unknown",
]

SuggestedArtist = Annotated[
    str,
    StringConstraints(
        strip_whitespace=True,
        min_length=1,
        max_length=80,
    ),
]
ConfidenceLevel = Literal["high", "medium", "low"]
RiskLevel = Literal["low", "high"]
MessageSource = Literal["whatsapp", "outlook", "vcita", "other"]
VisualColorPreference = Literal["black-and-grey", "color", "unknown"]

MissingInformationItem = Literal[
    "tattoo idea",
    "size in cm",
    "placement",
    "reference images",
    "tattoo style",
    "color preference",
    "preferred date",
    "preferred time",
]


class Message(BaseModel):
    """Single recent conversation message supplied by the backend."""

    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)

    role: Literal["user", "assistant"] = Field(
        description="Conversation participant that produced the message.",
    )
    content: str = Field(
        min_length=1,
        description="Message text used to resolve conversational context.",
    )


class TattooInquiryInput(BaseModel):
    """Hybrid-context inquiry payload assembled by the backend."""

    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)

    current_message: str = Field(
        description=(
            "Latest client text, or an empty string for an image-only message."
        ),
    )
    new_image_urls: list[str] = Field(
        default_factory=list,
        description="Image URLs attached only to the latest client message.",
    )
    existing_db_state: dict[str, Any] = Field(
        default_factory=dict,
        description="Previously collected inquiry data supplied by the backend.",
    )
    recent_chat_history: list[Message] = Field(
        default_factory=list,
        max_length=30,
        description="Up to 30 recent messages used for context resolution.",
    )
    message_source: MessageSource = Field(
        default="whatsapp",
        description=(
            "Source channel used to select WhatsApp chat or Outlook email "
            "reply formatting."
        ),
    )

    @model_validator(mode="before")
    @classmethod
    def infer_nested_message_source(cls, value: Any) -> Any:
        """Use the backend's nested lead/intake source when not explicit."""
        if not isinstance(value, dict) or "message_source" in value:
            return value

        existing_db_state = value.get("existing_db_state")
        if not isinstance(existing_db_state, dict):
            return value

        candidates: list[Any] = []
        for key in ("intake", "lead"):
            record = existing_db_state.get(key)
            if isinstance(record, dict):
                candidates.append(record.get("source"))
        candidates.append(existing_db_state.get("source"))

        for candidate in candidates:
            normalized = cls._normalize_source_value(candidate)
            if normalized in MESSAGE_SOURCE_OPTIONS:
                data = dict(value)
                data["message_source"] = normalized
                return data
        return value

    @field_validator("message_source", mode="before")
    @classmethod
    def normalize_message_source(cls, value: Any) -> Any:
        """Accept source names without making casing a backend concern."""
        return cls._normalize_source_value(value)

    @staticmethod
    def _normalize_source_value(value: Any) -> Any:
        """Normalize known backend source values and email aliases."""
        if not isinstance(value, str):
            return value
        normalized = value.strip().casefold()
        return _MESSAGE_SOURCE_ALIASES.get(normalized, normalized)

    @field_validator("new_image_urls")
    @classmethod
    def normalize_image_urls(cls, value: list[str]) -> list[str]:
        """Drop empty URL items and trim surrounding whitespace."""
        return [item.strip() for item in value if item and item.strip()]

    @field_validator("recent_chat_history", mode="before")
    @classmethod
    def keep_latest_chat_history(cls, value: Any) -> Any:
        """Drop empty media-only text entries and keep the latest 30."""
        if not isinstance(value, (list, tuple)):
            return value

        non_empty_messages: list[Any] = []
        for item in value:
            if isinstance(item, Message):
                if item.content.strip():
                    non_empty_messages.append(item)
                continue
            if isinstance(item, dict):
                content = item.get("content")
                if isinstance(content, str) and not content.strip():
                    continue
            non_empty_messages.append(item)
        return non_empty_messages[-30:]

    @model_validator(mode="after")
    def require_text_or_image(self) -> Self:
        """Require usable text or at least one attached image URL."""
        if not self.current_message and not self.new_image_urls:
            raise ValueError(
                "current_message or at least one new_image_url is required."
            )
        return self


class TattooVisionOutput(BaseModel):
    """Strict style and color signals extracted from reference images."""

    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)

    style_tags: list[StyleTag] = Field(min_length=1)
    color_preference: VisualColorPreference


class TattooExtractionDraft(BaseModel):
    """Intermediate extraction output before routing and risk enrichment."""

    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)

    tattoo_idea: str = Field(
        description="Core tattoo concept extracted from client text.",
    )
    style_tags: list[StyleTag] = Field(
        default_factory=list,
        description="Tattoo style tags from vision analysis.",
    )
    placement: str = Field(
        description="Requested body placement for the tattoo.",
    )
    size_estimate_cm: str = Field(
        description="Approximate tattoo size in centimeters.",
    )
    color_preference: str = Field(
        description="Client color preference for the tattoo.",
    )
    date: str = Field(
        default="",
        description="Preferred appointment date in YYYY-MM-DD format.",
    )
    time: str = Field(
        default="",
        description="Preferred appointment time in 24-hour HH:MM format.",
    )
    missing_information: list[MissingInformationItem] = Field(
        default_factory=list,
        description=(
            "Missing intake items from the required checklist: "
            "tattoo idea, size in cm, placement, reference images, "
            "tattoo style, color preference, preferred date, preferred time."
        ),
    )

    @field_validator("date")
    @classmethod
    def validate_date_format(cls, value: str) -> str:
        """Require a real calendar date in the public YYYY-MM-DD format."""
        return _validate_preferred_date(value)

    @field_validator("time")
    @classmethod
    def validate_time_format(cls, value: str) -> str:
        """Require a real clock time in the public 24-hour HH:MM format."""
        return _validate_preferred_time(value)


class AIExtractionOutput(BaseModel):
    """Final strict JSON contract returned to backend services."""

    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)

    tattoo_idea: str = Field(
        description="Short summary of the tattoo request from the client.",
    )
    style_tags: list[StyleTag] = Field(
        min_length=1,
        description="Detected style tags from the approved style taxonomy.",
    )
    placement: str = Field(
        description="Body placement requested by the client.",
    )
    size_estimate_cm: str = Field(
        description="Tattoo size estimate expressed in centimeters.",
    )
    color_preference: str = Field(
        description="Color preference such as black-and-grey or full color.",
    )
    date: str = Field(
        default="",
        description="Preferred appointment date in YYYY-MM-DD format.",
    )
    time: str = Field(
        default="",
        description="Preferred appointment time in 24-hour HH:MM format.",
    )
    suggested_artist: SuggestedArtist = Field(
        description="Configured artist display name or Unclear.",
    )
    confidence_level: ConfidenceLevel = Field(
        description="Confidence in routing and extraction quality.",
    )
    ai_reasoning: str = Field(
        description="Brief operational reasoning behind routing and risk output.",
    )
    missing_information: list[MissingInformationItem] = Field(
        default_factory=list,
        description="Checklist items still required before booking follow-up.",
    )
    risk_level: RiskLevel = Field(
        description="Low or high risk triage label for this inquiry.",
    )
    draft_reply: str = Field(
        description="Draft text requiring the delivery controls below.",
    )
    auto_reply_allowed: bool = Field(
        description="Whether backend automation may send the draft directly.",
    )
    telegram_review_required: bool = Field(
        description="Whether the draft must be routed to staff in Telegram.",
    )

    @field_validator("date")
    @classmethod
    def validate_date_format(cls, value: str) -> str:
        """Require a real calendar date in the public YYYY-MM-DD format."""
        return _validate_preferred_date(value)

    @field_validator("time")
    @classmethod
    def validate_time_format(cls, value: str) -> str:
        """Require a real clock time in the public 24-hour HH:MM format."""
        return _validate_preferred_time(value)

    @model_validator(mode="before")
    @classmethod
    def default_delivery_controls(cls, value: Any) -> Any:
        """Derive safe delivery defaults from the declared risk level."""
        if not isinstance(value, dict):
            return value
        payload = dict(value)
        is_high_risk = payload.get("risk_level") == "high"
        payload.setdefault("auto_reply_allowed", not is_high_risk)
        payload.setdefault("telegram_review_required", is_high_risk)
        return payload

    @model_validator(mode="after")
    def enforce_high_risk_delivery(self) -> Self:
        """Forbid high-risk outputs from entering the auto-reply channel."""
        if self.risk_level == "high" and self.auto_reply_allowed:
            raise ValueError("High-risk drafts cannot allow auto-replies.")
        if self.risk_level == "high" and not self.telegram_review_required:
            raise ValueError("High-risk drafts require Telegram review.")
        return self


def _validate_preferred_date(value: str) -> str:
    """Validate and preserve an optional ISO calendar date string."""
    if not value:
        return value
    try:
        parsed = calendar_date.fromisoformat(value)
    except ValueError as exc:
        raise ValueError("date must use YYYY-MM-DD format.") from exc
    if parsed.isoformat() != value:
        raise ValueError("date must use YYYY-MM-DD format.")
    return value


def _validate_preferred_time(value: str) -> str:
    """Validate and preserve an optional 24-hour clock time string."""
    if not value:
        return value
    try:
        parsed = datetime.strptime(value, "%H:%M")
    except ValueError as exc:
        raise ValueError("time must use 24-hour HH:MM format.") from exc
    if parsed.strftime("%H:%M") != value:
        raise ValueError("time must use 24-hour HH:MM format.")
    return value
