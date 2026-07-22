"""AI-only contracts for studio decisions, feedback, and pricing."""

from __future__ import annotations

from datetime import datetime
from decimal import Decimal
from typing import Annotated, Literal, Self

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    StringConstraints,
    field_validator,
    model_validator,
)

from .schemas import AIExtractionOutput, ConfidenceLevel, Message, StyleTag

ArtistKey = Annotated[
    str,
    StringConstraints(
        pattern=r"^[a-z0-9][a-z0-9_-]{0,63}$",
        strip_whitespace=True,
    ),
]
CurrencyCode = Annotated[
    str,
    StringConstraints(
        pattern=r"^[A-Z]{3}$",
        strip_whitespace=True,
    ),
]
InquiryChannel = Literal["whatsapp", "vcita", "outlook", "other"]
DecisionAction = Literal[
    "accept_request",
    "request_more_information",
    "offer_consultation",
    "reject_politely",
    "artist_review",
    "pricing_review",
    "ready_to_book",
]
DecisionOutcome = Literal["correct", "corrected", "rejected"]
DecisionSource = Literal["default_rules", "verified_history", "unresolved"]
PriceSource = Literal["pricing_rule", "approved_history"]


class StrictDecisionModel(BaseModel):
    """Base model that rejects unknown decision fields."""

    model_config = ConfigDict(
        extra="forbid",
        str_strip_whitespace=True,
    )


class ArtistOption(StrictDecisionModel):
    """Artist that may be recommended for this request."""

    key: ArtistKey = Field(
        description="Stable artist identifier supplied by the backend.",
    )
    display_name: str = Field(
        min_length=1,
        max_length=80,
        description="Artist name displayed to studio staff.",
    )


class MoneyRange(StrictDecisionModel):
    """Validated monetary range associated with a studio decision."""

    currency: CurrencyCode = Field(description="ISO 4217 currency code.")
    minimum: Decimal = Field(ge=0)
    maximum: Decimal = Field(ge=0)

    @model_validator(mode="after")
    def validate_range(self) -> Self:
        """Require the upper amount to be at least the lower amount."""
        if self.maximum < self.minimum:
            raise ValueError("maximum must be greater than or equal to minimum.")
        return self


class NextAction(StrictDecisionModel):
    """Recommended operational action for studio staff."""

    action: DecisionAction
    reason: str = Field(min_length=1, max_length=500)
    priority: Literal["low", "normal", "high"] = "normal"
    requires_human_review: bool = True


class DecisionHistoryExample(StrictDecisionModel):
    """Verified prior studio decision supplied for this invocation."""

    example_id: str = Field(
        min_length=1,
        max_length=80,
        pattern=r"^[A-Za-z0-9._-]+$",
    )
    channel: InquiryChannel
    style_tags: list[StyleTag] = Field(default_factory=list, max_length=10)
    placement: str = Field(default="", max_length=100)
    size_estimate_cm: str = Field(default="", max_length=100)
    color_preference: str = Field(default="", max_length=100)
    original_ai_artist_key: ArtistKey | None = None
    final_artist_key: ArtistKey | None = None
    original_ai_action: DecisionAction
    final_action: DecisionAction
    ai_suggestion_outcome: DecisionOutcome
    correction_reason: str | None = Field(default=None, max_length=500)
    approved_price_range: MoneyRange | None = None


class PricingRule(StrictDecisionModel):
    """Authoritative internal pricing rule supplied by the studio."""

    rule_id: str = Field(
        min_length=1,
        max_length=80,
        pattern=r"^[A-Za-z0-9._-]+$",
    )
    description: str = Field(min_length=1, max_length=500)
    priority: int = Field(default=0, ge=0, le=1000)
    artist_keys: list[ArtistKey] = Field(default_factory=list, max_length=20)
    style_tags: list[StyleTag] = Field(default_factory=list, max_length=10)
    placement_terms: list[str] = Field(default_factory=list, max_length=20)
    min_size_cm: Decimal | None = Field(default=None, ge=0)
    max_size_cm: Decimal | None = Field(default=None, ge=0)
    price_range: MoneyRange
    requires_consultation: bool = False

    @model_validator(mode="after")
    def validate_size_range(self) -> Self:
        """Require maximum size to be no smaller than minimum size."""
        if (
            self.min_size_cm is not None
            and self.max_size_cm is not None
            and self.max_size_cm < self.min_size_cm
        ):
            raise ValueError(
                "max_size_cm must be greater than or equal to min_size_cm."
            )
        return self


class StudioDecisionContext(StrictDecisionModel):
    """Request-scoped policies and verified examples supplied by backend."""

    channel: InquiryChannel
    artist_options: list[ArtistOption] = Field(min_length=1, max_length=20)
    decision_history: list[DecisionHistoryExample] = Field(
        default_factory=list,
        max_length=12,
    )
    pricing_rules: list[PricingRule] = Field(
        default_factory=list,
        max_length=50,
    )

    @model_validator(mode="after")
    def validate_references(self) -> Self:
        """Validate unique IDs and artist references within this context."""
        artist_keys = [artist.key for artist in self.artist_options]
        if len(artist_keys) != len(set(artist_keys)):
            raise ValueError("artist option keys must be unique.")

        example_ids = [item.example_id for item in self.decision_history]
        if len(example_ids) != len(set(example_ids)):
            raise ValueError("decision example IDs must be unique.")

        rule_ids = [rule.rule_id for rule in self.pricing_rules]
        if len(rule_ids) != len(set(rule_ids)):
            raise ValueError("pricing rule IDs must be unique.")

        available = set(artist_keys)
        for example in self.decision_history:
            if (
                example.final_artist_key is not None
                and example.final_artist_key not in available
            ):
                raise ValueError(
                    "example final artist must exist in artist_options."
                )
        for rule in self.pricing_rules:
            if set(rule.artist_keys).difference(available):
                raise ValueError(
                    "pricing rule artist keys must exist in artist_options."
                )
        return self


class ArtistSuggestion(StrictDecisionModel):
    """Internal artist recommendation, including request-scoped artists."""

    artist_key: ArtistKey | None = None
    artist_name: str | None = Field(default=None, max_length=80)
    confidence_level: ConfidenceLevel
    reasoning: str = Field(min_length=1, max_length=500)
    source: DecisionSource

    @model_validator(mode="after")
    def validate_artist_pair(self) -> Self:
        """Require artist key and name to be supplied together."""
        if (self.artist_key is None) != (self.artist_name is None):
            raise ValueError(
                "artist_key and artist_name must be provided together."
            )
        return self


class InternalPriceEstimate(StrictDecisionModel):
    """Staff-only estimate that cannot be approved for client sharing."""

    price_range: MoneyRange
    confidence_level: ConfidenceLevel
    source: PriceSource
    reasoning: str = Field(min_length=1, max_length=500)
    applied_rule_ids: list[str] = Field(default_factory=list, max_length=10)
    applied_example_ids: list[str] = Field(default_factory=list, max_length=12)
    visibility: Literal["internal_only"] = "internal_only"
    requires_consultation: bool = False
    requires_human_approval: Literal[True] = True
    client_safe_to_share: Literal[False] = False


class StudioDecisionOutput(StrictDecisionModel):
    """Internal decision result returned only to studio-facing callers."""

    analysis: AIExtractionOutput
    artist_suggestion: ArtistSuggestion
    suggested_next_action: NextAction
    internal_price_estimate: InternalPriceEstimate | None = None
    applied_history_example_ids: list[str] = Field(
        default_factory=list,
        max_length=12,
    )


class StudioDecisionFeedback(StrictDecisionModel):
    """Human decision label used to improve later invocations."""

    decided_by: str = Field(min_length=1, max_length=80)
    final_action: DecisionAction
    final_artist_key: ArtistKey | None = None
    ai_suggestion_outcome: DecisionOutcome
    correction_reason: str | None = Field(default=None, max_length=500)
    final_action_notes: str | None = Field(default=None, max_length=500)
    approved_price_range: MoneyRange | None = None
    occurred_at: datetime

    @field_validator("occurred_at")
    @classmethod
    def require_timezone(cls, value: datetime) -> datetime:
        """Require an unambiguous feedback timestamp."""
        if value.tzinfo is None or value.utcoffset() is None:
            raise ValueError("occurred_at must include timezone information.")
        return value


class StudioLearningRecord(StrictDecisionModel):
    """Serializable AI-only record for backend persistence."""

    channel: InquiryChannel
    original_client_message: str = Field(min_length=1)
    recent_chat_history: list[Message] = Field(default_factory=list, max_length=7)
    reference_image_urls: list[str] = Field(default_factory=list)
    decision: StudioDecisionOutput
    human_feedback: StudioDecisionFeedback
