"""Tests for feedback-conditioned studio decisions and internal pricing."""

from __future__ import annotations

from datetime import datetime, timezone
from decimal import Decimal

from ai_brain.decision import StudioDecisionEngine
from ai_brain.decision_schemas import (
    ArtistOption,
    DecisionHistoryExample,
    MoneyRange,
    PricingRule,
    StudioDecisionContext,
    StudioDecisionFeedback,
)
from ai_brain.processor import StudioAIBrain
from ai_brain.schemas import (
    AIExtractionOutput,
    Message,
    TattooExtractionDraft,
    TattooInquiryInput,
)


class StaticVisionAnalyzer:
    """Return deterministic style tags without downloading images."""

    def analyze_styles(self, image_urls: list[str]) -> list[str]:
        """Return fine-line style tags for the orchestration test."""
        return ["fine-line", "minimal"]


class StaticTextExtractor:
    """Return deterministic structured inquiry facts."""

    def extract(
        self,
        current_message: str,
        style_tags: list[str],
        new_image_urls: list[str] | None = None,
        existing_db_state: dict[str, object] | None = None,
        recent_chat_history: list[Message] | None = None,
    ) -> TattooExtractionDraft:
        """Return facts that match the verified Lana example."""
        return TattooExtractionDraft(
            tattoo_idea="Matching fine-line sister tattoos",
            style_tags=["fine-line", "minimal"],
            placement="inner wrist",
            size_estimate_cm="6cm",
            color_preference="black-and-grey",
            missing_information=[],
        )


class StaticRouter:
    """Return the existing client-facing analysis contract."""

    def route(self, extracted: TattooExtractionDraft) -> AIExtractionOutput:
        """Return a deterministic legacy routing result."""
        return _analysis()


def _analysis() -> AIExtractionOutput:
    """Build a complete fine-line analysis using legacy default routing."""
    return AIExtractionOutput(
        tattoo_idea="Matching fine-line sister tattoos",
        style_tags=["fine-line", "minimal"],
        placement="inner wrist",
        size_estimate_cm="6cm",
        color_preference="black-and-grey",
        suggested_artist="Nina",
        confidence_level="medium",
        ai_reasoning="Default fine-line routing selected Nina.",
        missing_information=[],
        risk_level="low",
        draft_reply="Thanks. The studio will review your request.",
    )


def _history_example() -> DecisionHistoryExample:
    """Build a verified correction assigning similar work to Lana."""
    return DecisionHistoryExample(
        example_id="fine-line-lana-001",
        channel="whatsapp",
        style_tags=["fine-line", "minimal"],
        placement="inner wrist",
        size_estimate_cm="5cm",
        color_preference="black-and-grey",
        original_ai_artist_key="nina",
        final_artist_key="lana",
        original_ai_action="artist_review",
        final_action="ready_to_book",
        ai_suggestion_outcome="corrected",
        correction_reason="Lana handles this reference style.",
        approved_price_range=MoneyRange(
            currency="EUR",
            minimum=Decimal("120"),
            maximum=Decimal("150"),
        ),
    )


def _context(*, include_rule: bool = True) -> StudioDecisionContext:
    """Build request-scoped artists, feedback, and optional price policy."""
    rules: list[PricingRule] = []
    if include_rule:
        rules.append(
            PricingRule(
                rule_id="small-fine-line",
                description="Small fine-line work up to 10cm.",
                priority=100,
                artist_keys=["lana"],
                style_tags=["fine-line", "minimal"],
                max_size_cm=Decimal("10"),
                price_range=MoneyRange(
                    currency="EUR",
                    minimum=Decimal("130"),
                    maximum=Decimal("170"),
                ),
            )
        )
    return StudioDecisionContext(
        channel="whatsapp",
        artist_options=[
            ArtistOption(key="nina", display_name="Nina"),
            ArtistOption(key="hoss", display_name="Hoss"),
            ArtistOption(key="lana", display_name="Lana"),
        ],
        decision_history=[_history_example()],
        pricing_rules=rules,
    )


def test_verified_correction_can_suggest_request_scoped_lana() -> None:
    """A relevant verified correction overrides legacy default routing."""
    result = StudioDecisionEngine().decide(
        analysis=_analysis(),
        current_message="I want this matching tattoo with my sister.",
        context=_context(),
    )

    assert result.artist_suggestion.artist_key == "lana"
    assert result.artist_suggestion.artist_name == "Lana"
    assert result.artist_suggestion.source == "verified_history"
    assert result.suggested_next_action.action == "ready_to_book"
    assert result.applied_history_example_ids == ["fine-line-lana-001"]

    estimate = result.internal_price_estimate
    assert estimate is not None
    assert estimate.source == "pricing_rule"
    assert estimate.price_range.minimum == Decimal("130")
    assert estimate.price_range.maximum == Decimal("170")
    assert estimate.visibility == "internal_only"
    assert estimate.requires_human_approval is True
    assert estimate.client_safe_to_share is False
    assert "130" not in result.analysis.draft_reply


def test_approved_history_can_supply_internal_price_without_rule() -> None:
    """Verified approved prices act as conservative fallback evidence."""
    result = StudioDecisionEngine().decide(
        analysis=_analysis(),
        current_message="I want matching wrist tattoos.",
        context=_context(include_rule=False),
    )

    estimate = result.internal_price_estimate
    assert estimate is not None
    assert estimate.source == "approved_history"
    assert estimate.price_range.minimum == Decimal("120")
    assert estimate.price_range.maximum == Decimal("150")
    assert estimate.applied_example_ids == ["fine-line-lana-001"]


def test_pricing_request_requires_review_and_context_does_not_leak() -> None:
    """Prices remain internal and direct price questions require review."""
    engine = StudioDecisionEngine()
    first = engine.decide(
        analysis=_analysis(),
        current_message="How much will the matching tattoos cost?",
        context=_context(),
    )
    empty_context = StudioDecisionContext(
        channel="outlook",
        artist_options=[
            ArtistOption(key="nina", display_name="Nina"),
            ArtistOption(key="hoss", display_name="Hoss"),
        ],
    )
    second = engine.decide(
        analysis=_analysis(),
        current_message="I want a fine-line wrist tattoo.",
        context=empty_context,
    )

    assert first.suggested_next_action.action == "pricing_review"
    assert first.internal_price_estimate is not None
    assert second.artist_suggestion.artist_key == "nina"
    assert second.artist_suggestion.source == "default_rules"
    assert second.internal_price_estimate is None
    assert second.applied_history_example_ids == []


def test_learning_record_is_serializable_and_contains_human_feedback() -> None:
    """The AI builds a record but performs no persistence itself."""
    engine = StudioDecisionEngine()
    context = _context()
    inquiry = TattooInquiryInput(
        current_message="Actually make it 6cm.",
        new_image_urls=["https://example.com/reference.jpg"],
        existing_db_state={"size": "5cm"},
        recent_chat_history=[
            Message(role="user", content="I originally asked for 5cm."),
        ],
    )
    decision = engine.decide(
        analysis=_analysis(),
        current_message=inquiry.current_message,
        context=context,
    )
    feedback = StudioDecisionFeedback(
        decided_by="Nina",
        final_action="ready_to_book",
        final_artist_key="lana",
        ai_suggestion_outcome="corrected",
        correction_reason="Reference style belongs with Lana.",
        approved_price_range=MoneyRange(
            currency="EUR",
            minimum=Decimal("140"),
            maximum=Decimal("160"),
        ),
        occurred_at=datetime.now(timezone.utc),
    )

    record = engine.build_learning_record(
        inquiry=inquiry,
        context=context,
        decision=decision,
        feedback=feedback,
    )

    payload = record.model_dump(mode="json")
    assert payload["channel"] == "whatsapp"
    assert payload["original_client_message"] == "Actually make it 6cm."
    assert payload["human_feedback"]["final_artist_key"] == "lana"
    assert payload["decision"]["internal_price_estimate"][
        "visibility"
    ] == "internal_only"

    example = engine.build_history_example(
        record=record,
        example_id="recorded-decision-001",
    )
    assert example.final_artist_key == "lana"
    assert example.ai_suggestion_outcome == "corrected"
    assert example.approved_price_range is not None
    assert example.approved_price_range.minimum == Decimal("140")


def test_studio_ai_brain_exposes_internal_decision_entry_point() -> None:
    """The orchestrator runs AI-only decisions with supplied context."""
    brain = StudioAIBrain(
        vision_analyzer=StaticVisionAnalyzer(),
        text_extractor=StaticTextExtractor(),
        router=StaticRouter(),
    )
    inquiry = TattooInquiryInput(
        current_message="Matching tattoo with my sister.",
        new_image_urls=["https://example.com/reference.jpg"],
    )

    result = brain.process_studio_decision(
        inquiry=inquiry,
        context=_context(),
    )

    assert result.artist_suggestion.artist_key == "lana"
    assert result.internal_price_estimate is not None