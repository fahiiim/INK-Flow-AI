"""Strict data contracts for the Tattoo Studio AI Brain."""

from __future__ import annotations

from typing import Any, Literal

from pydantic import (
    AliasChoices,
    BaseModel,
    ConfigDict,
    Field,
    field_validator,
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
    "size in cm",
    "placement",
    "reference images",
    "color preference",
    "preferred date",
)

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

SuggestedArtist = Literal["Nina", "Hoss", "Unclear"]
ConfidenceLevel = Literal["high", "medium", "low"]
RiskLevel = Literal["low", "high"]

MissingInformationItem = Literal[
    "size in cm",
    "placement",
    "reference images",
    "color preference",
    "preferred date",
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
        min_length=1,
        validation_alias=AliasChoices("current_message", "client_text"),
        description="Latest client message, which overrides conflicting state.",
    )
    new_image_urls: list[str] = Field(
        default_factory=list,
        validation_alias=AliasChoices("new_image_urls", "image_urls"),
        description="Image URLs attached only to the latest client message.",
    )
    existing_db_state: dict[str, Any] = Field(
        default_factory=dict,
        description="Previously collected inquiry data supplied by the backend.",
    )
    recent_chat_history: list[Message] = Field(
        default_factory=list,
        max_length=7,
        description="Up to seven recent messages used for context resolution.",
    )

    @field_validator("new_image_urls")
    @classmethod
    def normalize_image_urls(cls, value: list[str]) -> list[str]:
        """Drop empty URL items and trim surrounding whitespace."""
        return [item.strip() for item in value if item and item.strip()]

    @field_validator("recent_chat_history", mode="before")
    @classmethod
    def keep_latest_chat_history(cls, value: Any) -> Any:
        """Keep only the latest seven raw messages before validation."""
        if isinstance(value, (list, tuple)):
            return list(value[-7:])
        return value

    @property
    def client_text(self) -> str:
        """Return the legacy text attribute during the staged migration."""
        return self.current_message

    @property
    def image_urls(self) -> list[str]:
        """Return the legacy image attribute during the staged migration."""
        return self.new_image_urls


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
    missing_information: list[MissingInformationItem] = Field(
        default_factory=list,
        description=(
            "Missing intake items from the required checklist: "
            "size in cm, placement, reference images, "
            "color preference, preferred date."
        ),
    )


class AIExtractionOutput(BaseModel):
    """Final strict JSON contract returned to backend services."""

    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)

    tattoo_idea: str = Field(
        description="Short summary of the tattoo request from the client.",
    )
    style_tags: list[StyleTag] = Field(
        default_factory=list,
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
    suggested_artist: SuggestedArtist = Field(
        description="Artist recommendation: Nina, Hoss, or Unclear.",
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
        description="Polite suggested response that staff can send to client.",
    )
