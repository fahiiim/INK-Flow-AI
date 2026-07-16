"""Strict data contracts for the Tattoo Studio AI Brain."""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator

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


class TattooInquiryInput(BaseModel):
    """Raw inquiry payload received from backend channels."""

    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)

    client_text: str = Field(
        min_length=1,
        description="Client message text exactly as received.",
    )
    image_urls: list[str] = Field(
        default_factory=list,
        description="Reference image URLs attached to the inquiry.",
    )

    @field_validator("image_urls")
    @classmethod
    def normalize_image_urls(cls, value: list[str]) -> list[str]:
        """Drop empty URL items and trim surrounding whitespace."""
        return [item.strip() for item in value if item and item.strip()]


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
