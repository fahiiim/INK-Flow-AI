"""Deterministic staff-only price estimation from supplied evidence."""

from __future__ import annotations

import re
from collections.abc import Sequence
from decimal import Decimal, InvalidOperation
from pathlib import Path
from typing import Any, Self

import yaml
from pydantic import BaseModel, ConfigDict, Field, ValidationError, model_validator

from .decision_schemas import (
    DecisionHistoryExample,
    InternalPriceEstimate,
    MoneyRange,
    PricingRule,
    StudioDecisionContext,
)
from .schemas import AIExtractionOutput, ConfidenceLevel

_SIZE_PATTERN = re.compile(r"\d+(?:\.\d+)?")
DEFAULT_PRICING_CONFIG_PATH = (
    Path(__file__).resolve().parent.parent
    / "config"
    / "pricing_rules.yaml"
)


class PricingConfigError(ValueError):
    """Raised when studio pricing configuration cannot be loaded."""


class StudioPricingConfig(BaseModel):
    """Validated deterministic pricing policy for one studio deployment."""

    model_config = ConfigDict(
        extra="forbid",
        str_strip_whitespace=True,
    )

    config_version: int = Field(default=1, ge=1)
    rules: list[PricingRule] = Field(default_factory=list, max_length=100)

    @model_validator(mode="after")
    def validate_unique_rule_ids(self) -> Self:
        """Reject ambiguous duplicate pricing-rule identifiers."""
        rule_ids = [rule.rule_id for rule in self.rules]
        if len(rule_ids) != len(set(rule_ids)):
            raise ValueError("Pricing rule IDs must be unique.")
        return self


def load_pricing_config(
    path: str | Path = DEFAULT_PRICING_CONFIG_PATH,
) -> StudioPricingConfig:
    """Load and validate a studio pricing policy from safe YAML."""
    resolved_path = Path(path).expanduser().resolve()
    if not resolved_path.is_file():
        raise PricingConfigError(
            f"Pricing config file not found: {resolved_path}"
        )
    try:
        payload: Any = yaml.safe_load(
            resolved_path.read_text(encoding="utf-8")
        )
        return StudioPricingConfig.model_validate(payload)
    except (OSError, ValidationError, yaml.YAMLError) as exc:
        raise PricingConfigError(
            f"Invalid pricing configuration: {resolved_path}"
        ) from exc


def parse_size_cm(value: str) -> Decimal | None:
    """Return the largest numeric dimension from a size description."""
    numbers: list[Decimal] = []
    for raw_number in _SIZE_PATTERN.findall(value):
        try:
            numbers.append(Decimal(raw_number))
        except InvalidOperation:
            continue
    return max(numbers) if numbers else None


class InternalPricingEstimator:
    """Calculate internal estimates without inventing unsupported prices."""

    def __init__(
        self,
        pricing_config: StudioPricingConfig | None = None,
        config_path: str | Path | None = None,
    ) -> None:
        """Initialize the estimator from injected or file-backed rules.

        Args:
            pricing_config: Optional validated configuration for dependency
                injection and deterministic tests.
            config_path: Optional YAML path. It cannot be combined with an
                injected configuration.
        """
        if pricing_config is not None and config_path is not None:
            raise ValueError(
                "pricing_config and config_path cannot both be provided."
            )
        self._config = (
            pricing_config.model_copy(deep=True)
            if pricing_config is not None
            else load_pricing_config(
                config_path or DEFAULT_PRICING_CONFIG_PATH
            )
        )

    def estimate(
        self,
        analysis: AIExtractionOutput,
        context: StudioDecisionContext,
        artist_key: str | None,
        history_examples: Sequence[DecisionHistoryExample] | None = None,
    ) -> InternalPriceEstimate | None:
        """Use configured rules first, then verified approved examples."""
        size_cm = parse_size_cm(analysis.size_estimate_cm)
        rule_estimate = self._estimate_from_rules(
            analysis=analysis,
            rules=self._resolve_rules(context.pricing_rules),
            artist_key=artist_key,
            size_cm=size_cm,
        )
        history_estimate = self._estimate_from_history(
            analysis=analysis,
            examples=(
                context.decision_history
                if history_examples is None
                else history_examples
            ),
            channel=context.channel,
            artist_key=artist_key,
            size_cm=size_cm,
        )
        if rule_estimate is not None:
            return self._combine_rule_and_history(
                rule_estimate,
                history_estimate,
            )
        return history_estimate

    def _combine_rule_and_history(
        self,
        rule_estimate: InternalPriceEstimate,
        history_estimate: InternalPriceEstimate | None,
    ) -> InternalPriceEstimate:
        """Keep the rule range and attach compatible historical evidence."""
        if history_estimate is None:
            return rule_estimate

        rule_range = rule_estimate.price_range
        history_range = history_estimate.price_range
        currency_matches = rule_range.currency == history_range.currency
        ranges_overlap = (
            currency_matches
            and history_range.maximum >= rule_range.minimum
            and history_range.minimum <= rule_range.maximum
        )
        payload = rule_estimate.model_dump(mode="python")
        payload["applied_example_ids"] = (
            history_estimate.applied_example_ids
        )
        if ranges_overlap:
            payload["confidence_level"] = "high"
            payload["reasoning"] = (
                "Matched an authoritative studio pricing rule supported by "
                "similar approved historical prices."
            )
        else:
            payload["confidence_level"] = "medium"
            payload["requires_consultation"] = True
            payload["reasoning"] = (
                "The authoritative pricing rule conflicts with similar "
                "approved history and requires staff review."
            )
        return InternalPriceEstimate.model_validate(payload)

    def _resolve_rules(
        self,
        request_rules: Sequence[PricingRule],
    ) -> list[PricingRule]:
        """Merge configured rules with request-scoped rule overrides."""
        merged = {
            rule.rule_id: rule.model_copy(deep=True)
            for rule in self._config.rules
        }
        for rule in request_rules:
            merged[rule.rule_id] = rule.model_copy(deep=True)
        return list(merged.values())

    def _estimate_from_rules(
        self,
        analysis: AIExtractionOutput,
        rules: list[PricingRule],
        artist_key: str | None,
        size_cm: Decimal | None,
    ) -> InternalPriceEstimate | None:
        """Return the highest-priority unambiguous matching rule."""
        ranked: list[tuple[tuple[int, int], PricingRule]] = []
        for rule in rules:
            if not self._rule_matches(
                rule=rule,
                analysis=analysis,
                artist_key=artist_key,
                size_cm=size_cm,
            ):
                continue
            ranked.append((self._rule_rank(rule), rule))

        if not ranked:
            return None
        best_rank = max(rank for rank, _ in ranked)
        best_rules = [rule for rank, rule in ranked if rank == best_rank]
        bands = {
            (
                rule.price_range.currency,
                rule.price_range.minimum,
                rule.price_range.maximum,
            )
            for rule in best_rules
        }
        if len(bands) != 1:
            return None

        confidence: ConfidenceLevel = "high"
        if best_rank[1] < 2:
            confidence = "medium"
        return InternalPriceEstimate(
            price_range=best_rules[0].price_range,
            confidence_level=confidence,
            source="pricing_rule",
            reasoning=(
                "Matched the highest-priority authoritative studio pricing rule."
            ),
            applied_rule_ids=[rule.rule_id for rule in best_rules],
            requires_consultation=any(
                rule.requires_consultation for rule in best_rules
            ),
        )

    def _rule_matches(
        self,
        rule: PricingRule,
        analysis: AIExtractionOutput,
        artist_key: str | None,
        size_cm: Decimal | None,
    ) -> bool:
        """Check whether every configured rule condition is satisfied."""
        if rule.artist_keys and artist_key not in rule.artist_keys:
            return False

        analysis_tags = set(analysis.style_tags)
        if rule.style_tags and not analysis_tags.intersection(rule.style_tags):
            return False

        placement = analysis.placement.lower()
        if rule.placement_terms and not any(
            term.lower() in placement for term in rule.placement_terms
        ):
            return False

        if rule.min_size_cm is not None:
            if size_cm is None or size_cm < rule.min_size_cm:
                return False
        if rule.max_size_cm is not None:
            if size_cm is None or size_cm > rule.max_size_cm:
                return False
        return True

    def _rule_rank(self, rule: PricingRule) -> tuple[int, int]:
        """Rank rules by explicit priority and condition specificity."""
        specificity = 0
        specificity += int(bool(rule.artist_keys))
        specificity += int(bool(rule.style_tags))
        specificity += int(bool(rule.placement_terms))
        specificity += int(rule.min_size_cm is not None)
        specificity += int(rule.max_size_cm is not None)
        return rule.priority, specificity

    def _estimate_from_history(
        self,
        analysis: AIExtractionOutput,
        examples: Sequence[DecisionHistoryExample],
        channel: str,
        artist_key: str | None,
        size_cm: Decimal | None,
    ) -> InternalPriceEstimate | None:
        """Build a conservative range from best matching approved examples."""
        ranked: list[tuple[int, DecisionHistoryExample]] = []
        for example in examples:
            if example.approved_price_range is None:
                continue
            score = self._history_score(
                example=example,
                analysis=analysis,
                channel=channel,
                artist_key=artist_key,
                size_cm=size_cm,
            )
            if score >= 3:
                ranked.append((score, example))

        if not ranked:
            return None
        best_score = max(score for score, _ in ranked)
        best_examples = [item for score, item in ranked if score == best_score]
        currencies = {
            item.approved_price_range.currency
            for item in best_examples
            if item.approved_price_range is not None
        }
        if len(currencies) != 1:
            return None

        ranges = [
            item.approved_price_range
            for item in best_examples
            if item.approved_price_range is not None
        ]
        price_range = MoneyRange(
            currency=next(iter(currencies)),
            minimum=min(item.minimum for item in ranges),
            maximum=max(item.maximum for item in ranges),
        )
        confidence: ConfidenceLevel = "medium"
        if len(best_examples) == 1:
            confidence = "low"
        return InternalPriceEstimate(
            price_range=price_range,
            confidence_level=confidence,
            source="approved_history",
            reasoning=(
                "Derived from the closest verified studio-approved price examples."
            ),
            applied_example_ids=[item.example_id for item in best_examples],
        )

    def _history_score(
        self,
        example: DecisionHistoryExample,
        analysis: AIExtractionOutput,
        channel: str,
        artist_key: str | None,
        size_cm: Decimal | None,
    ) -> int:
        """Score the relevance of one verified historical decision."""
        style_overlap = set(example.style_tags).intersection(analysis.style_tags)
        placement_match = bool(
            example.placement
            and analysis.placement
            and example.placement.lower() == analysis.placement.lower()
        )
        if not style_overlap and not placement_match:
            return 0

        score = len(style_overlap) * 4
        score += int(placement_match) * 2
        score += int(example.channel == channel)
        score += int(
            artist_key is not None and example.final_artist_key == artist_key
        ) * 2

        example_size = parse_size_cm(example.size_estimate_cm)
        if size_cm is not None and example_size is not None:
            difference = abs(size_cm - example_size)
            if difference <= Decimal("2"):
                score += 2
            elif difference <= Decimal("5"):
                score += 1
        return score
