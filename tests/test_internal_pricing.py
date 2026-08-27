"""Tests for YAML-backed, staff-only internal pricing policy."""

from __future__ import annotations

from decimal import Decimal
from pathlib import Path

from ai_brain.decision_schemas import (
    ArtistOption,
    DecisionHistoryExample,
    MoneyRange,
    PricingRule,
    StudioDecisionContext,
)
from ai_brain.pricing import InternalPricingEstimator, StudioPricingConfig
from ai_brain.schemas import AIExtractionOutput


def _analysis() -> AIExtractionOutput:
    """Build a request matching the temporary configured rule."""
    return AIExtractionOutput(
        tattoo_idea="Fine-line floral tattoo",
        style_tags=["fine-line", "floral"],
        placement="inner wrist",
        size_estimate_cm="8cm",
        color_preference="black-and-grey",
        suggested_artist="Lana",
        confidence_level="high",
        ai_reasoning="Configured specialty match.",
        missing_information=[],
        risk_level="low",
        draft_reply="The studio will review your request.",
    )


def _context() -> StudioDecisionContext:
    """Build a context without request-scoped price evidence."""
    return StudioDecisionContext(
        channel="whatsapp",
        artist_options=[ArtistOption(key="lana", display_name="Lana")],
    )


def test_file_backed_pricing_rule_is_applied(tmp_path: Path) -> None:
    """Studio pricing changes can be deployed through YAML only."""
    config_path = tmp_path / "pricing_rules.yaml"
    config_path.write_text(
        "\n".join(
            [
                "config_version: 1",
                "rules:",
                "  - rule_id: small-fine-line",
                "    description: Approved small fine-line range",
                "    priority: 100",
                "    artist_keys: [lana]",
                "    style_tags: [fine-line]",
                "    placement_terms: [wrist]",
                "    min_size_cm: 3",
                "    max_size_cm: 10",
                "    price_range:",
                "      currency: EUR",
                "      minimum: 130",
                "      maximum: 170",
                "    requires_consultation: false",
            ]
        ),
        encoding="utf-8",
    )
    estimator = InternalPricingEstimator(config_path=config_path)

    estimate = estimator.estimate(
        analysis=_analysis(),
        context=_context(),
        artist_key="lana",
    )

    assert estimate is not None
    assert estimate.source == "pricing_rule"
    assert estimate.internal_only is True
    assert estimate.price_range.minimum == Decimal("130")
    assert estimate.price_range.maximum == Decimal("170")


def test_conflicting_history_requires_consultation() -> None:
    """History cannot override a rule and conflicts force staff review."""
    rule = PricingRule(
        rule_id="small-fine-line",
        description="Approved small fine-line range",
        priority=100,
        artist_keys=["lana"],
        style_tags=["fine-line"],
        max_size_cm=Decimal("10"),
        price_range=MoneyRange(
            currency="EUR",
            minimum=Decimal("130"),
            maximum=Decimal("170"),
        ),
    )
    history = DecisionHistoryExample(
        example_id="conflicting-price-case",
        channel="whatsapp",
        style_tags=["fine-line"],
        placement="inner wrist",
        size_estimate_cm="8cm",
        color_preference="black-and-grey",
        original_ai_artist_key="lana",
        final_artist_key="lana",
        original_ai_action="pricing_review",
        final_action="ready_to_book",
        ai_suggestion_outcome="correct",
        approved_price_range=MoneyRange(
            currency="EUR",
            minimum=Decimal("300"),
            maximum=Decimal("400"),
        ),
    )
    estimator = InternalPricingEstimator(
        pricing_config=StudioPricingConfig(rules=[rule])
    )

    estimate = estimator.estimate(
        analysis=_analysis(),
        context=_context(),
        artist_key="lana",
        history_examples=[history],
    )

    assert estimate is not None
    assert estimate.source == "pricing_rule"
    assert estimate.price_range.maximum == Decimal("170")
    assert estimate.requires_consultation is True
    assert estimate.applied_example_ids == ["conflicting-price-case"]
