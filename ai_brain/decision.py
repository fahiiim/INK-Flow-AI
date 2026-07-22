"""Feedback-conditioned internal studio decision engine."""

from __future__ import annotations

from collections import defaultdict
from decimal import Decimal

from .decision_schemas import (
    ArtistOption,
    ArtistSuggestion,
    DecisionAction,
    DecisionHistoryExample,
    InternalPriceEstimate,
    NextAction,
    StudioDecisionContext,
    StudioDecisionFeedback,
    StudioDecisionOutput,
    StudioLearningRecord,
)
from .pricing import InternalPricingEstimator, parse_size_cm
from .schemas import AIExtractionOutput, ConfidenceLevel, TattooInquiryInput

_PRICE_TERMS = ("price", "pricing", "cost", "quote", "how much")


class StudioDecisionEngine:
    """Use request-scoped verified decisions without retaining state."""

    def __init__(
        self,
        pricing_estimator: InternalPricingEstimator | None = None,
    ) -> None:
        self._pricing_estimator = pricing_estimator or InternalPricingEstimator()

    def decide(
        self,
        analysis: AIExtractionOutput,
        current_message: str,
        context: StudioDecisionContext,
    ) -> StudioDecisionOutput:
        """Create staff-only artist, action, and price recommendations."""
        ranked = self._rank_examples(analysis, context)
        artist, artist_ids = self._suggest_artist(analysis, context, ranked)
        estimate = self._pricing_estimator.estimate(
            analysis=analysis,
            context=context,
            artist_key=artist.artist_key,
        )
        action, action_ids = self._suggest_action(
            analysis=analysis,
            current_message=current_message,
            ranked_examples=ranked,
            price_estimate=estimate,
        )
        applied_ids = sorted(set(artist_ids).union(action_ids))
        return StudioDecisionOutput(
            analysis=analysis,
            artist_suggestion=artist,
            suggested_next_action=action,
            internal_price_estimate=estimate,
            applied_history_example_ids=applied_ids,
        )

    def build_learning_record(
        self,
        inquiry: TattooInquiryInput,
        context: StudioDecisionContext,
        decision: StudioDecisionOutput,
        feedback: StudioDecisionFeedback,
    ) -> StudioLearningRecord:
        """Build a serializable record for caller-managed persistence."""
        if feedback.final_artist_key is not None:
            available = {item.key for item in context.artist_options}
            if feedback.final_artist_key not in available:
                raise ValueError(
                    "feedback final artist must exist in artist_options."
                )
        return StudioLearningRecord(
            channel=context.channel,
            original_client_message=inquiry.current_message,
            recent_chat_history=inquiry.recent_chat_history,
            reference_image_urls=inquiry.new_image_urls,
            decision=decision,
            human_feedback=feedback,
        )
    
    def build_history_example(
        self,
        record: StudioLearningRecord,
        example_id: str,
    ) -> DecisionHistoryExample:
        """Convert a completed feedback record into verified evidence."""
        analysis = record.decision.analysis
        feedback = record.human_feedback
        return DecisionHistoryExample(
            example_id=example_id,
            channel=record.channel,
            style_tags=analysis.style_tags,
            placement=analysis.placement,
            size_estimate_cm=analysis.size_estimate_cm,
            color_preference=analysis.color_preference,
            original_ai_artist_key=(
                record.decision.artist_suggestion.artist_key
            ),
            final_artist_key=feedback.final_artist_key,
            original_ai_action=(
                record.decision.suggested_next_action.action
            ),
            final_action=feedback.final_action,
            ai_suggestion_outcome=feedback.ai_suggestion_outcome,
            correction_reason=feedback.correction_reason,
            approved_price_range=feedback.approved_price_range,
        )

    def _rank_examples(
        self,
        analysis: AIExtractionOutput,
        context: StudioDecisionContext,
    ) -> list[tuple[int, DecisionHistoryExample]]:
        """Rank verified examples by structured inquiry similarity."""
        ranked: list[tuple[int, DecisionHistoryExample]] = []
        for example in context.decision_history:
            score = self._example_score(example, analysis, context.channel)
            if score >= 3:
                ranked.append((score, example))
        ranked.sort(key=lambda item: (-item[0], item[1].example_id))
        return ranked

    def _example_score(
        self,
        example: DecisionHistoryExample,
        analysis: AIExtractionOutput,
        channel: str,
    ) -> int:
        """Score an example using style, placement, size, and channel."""
        overlap = set(example.style_tags).intersection(analysis.style_tags)
        same_placement = bool(
            example.placement
            and analysis.placement
            and example.placement.lower() == analysis.placement.lower()
        )
        if not overlap and not same_placement:
            return 0

        score = len(overlap) * 4
        score += int(same_placement) * 2
        score += int(example.channel == channel)
        score += int(
            bool(example.color_preference)
            and example.color_preference.lower()
            == analysis.color_preference.lower()
        )
        current_size = parse_size_cm(analysis.size_estimate_cm)
        previous_size = parse_size_cm(example.size_estimate_cm)
        if current_size is not None and previous_size is not None:
            score += self._size_score(current_size, previous_size)
        return score

    def _size_score(self, current: Decimal, previous: Decimal) -> int:
        """Return a small similarity bonus for comparable sizes."""
        difference = abs(current - previous)
        if difference <= Decimal("2"):
            return 2
        if difference <= Decimal("5"):
            return 1
        return 0

    def _suggest_artist(
        self,
        analysis: AIExtractionOutput,
        context: StudioDecisionContext,
        ranked: list[tuple[int, DecisionHistoryExample]],
    ) -> tuple[ArtistSuggestion, list[str]]:
        """Prefer verified final assignments when evidence is unambiguous."""
        artist_map = {item.key: item for item in context.artist_options}
        votes: dict[str, int] = defaultdict(int)
        support: dict[str, list[str]] = defaultdict(list)
        for score, example in ranked:
            key = example.final_artist_key
            if key is None or key not in artist_map:
                continue
            votes[key] += score
            support[key].append(example.example_id)

        if votes:
            best = max(votes.values())
            winners = [key for key, score in votes.items() if score == best]
            if len(winners) == 1:
                key = winners[0]
                artist = artist_map[key]
                confidence: ConfidenceLevel = "medium"
                if best >= 8 or len(support[key]) > 1:
                    confidence = "high"
                return (
                    ArtistSuggestion(
                        artist_key=key,
                        artist_name=artist.display_name,
                        confidence_level=confidence,
                        reasoning=(
                            "Verified similar decisions support this artist."
                        ),
                        source="verified_history",
                    ),
                    support[key],
                )
        return self._default_artist(analysis, context.artist_options), []

    def _default_artist(
        self,
        analysis: AIExtractionOutput,
        options: list[ArtistOption],
    ) -> ArtistSuggestion:
        """Map legacy routing output into the supplied artist catalog."""
        if analysis.suggested_artist != "Unclear":
            for artist in options:
                if artist.display_name.casefold() == (
                    analysis.suggested_artist.casefold()
                ):
                    return ArtistSuggestion(
                        artist_key=artist.key,
                        artist_name=artist.display_name,
                        confidence_level=analysis.confidence_level,
                        reasoning=analysis.ai_reasoning,
                        source="default_rules",
                    )
        return ArtistSuggestion(
            artist_key=None,
            artist_name=None,
            confidence_level="low",
            reasoning="No unambiguous artist assignment evidence was found.",
            source="unresolved",
        )

    def _suggest_action(
        self,
        analysis: AIExtractionOutput,
        current_message: str,
        ranked_examples: list[tuple[int, DecisionHistoryExample]],
        price_estimate: InternalPriceEstimate | None,
    ) -> tuple[NextAction, list[str]]:
        """Choose a verified historical action or a safe default."""
        if self._contains_pricing_request(current_message):
            return (
                NextAction(
                    action="pricing_review",
                    reason=(
                        "Pricing requires studio approval before client use."
                    ),
                    priority="high",
                ),
                [],
            )
        if price_estimate is not None and price_estimate.requires_consultation:
            return (
                NextAction(
                    action="offer_consultation",
                    reason="The matching pricing rule requires consultation.",
                ),
                [],
            )

        votes: dict[DecisionAction, int] = defaultdict(int)
        support: dict[DecisionAction, list[str]] = defaultdict(list)
        for score, example in ranked_examples:
            votes[example.final_action] += score
            support[example.final_action].append(example.example_id)
        if votes:
            best = max(votes.values())
            winners = [key for key, score in votes.items() if score == best]
            if len(winners) == 1:
                action = winners[0]
                return (
                    NextAction(
                        action=action,
                        reason=(
                            "Verified similar decisions support this action."
                        ),
                    ),
                    support[action],
                )
        return self._default_action(analysis), []

    def _default_action(self, analysis: AIExtractionOutput) -> NextAction:
        """Derive a conservative action from the existing analysis."""
        if analysis.risk_level == "high":
            return NextAction(
                action="artist_review",
                reason="High-risk requests require manual studio review.",
                priority="high",
            )
        if analysis.missing_information:
            missing = ", ".join(analysis.missing_information)
            return NextAction(
                action="request_more_information",
                reason=f"Request missing details before proceeding: {missing}.",
            )
        if analysis.suggested_artist == "Unclear":
            return NextAction(
                action="offer_consultation",
                reason="Artist suitability remains unclear from current details.",
            )
        return NextAction(
            action="ready_to_book",
            reason="Request is complete enough for staff booking review.",
        )

    def _contains_pricing_request(self, current_message: str) -> bool:
        """Detect direct client requests for price information."""
        normalized = current_message.lower()
        return any(term in normalized for term in _PRICE_TERMS)
