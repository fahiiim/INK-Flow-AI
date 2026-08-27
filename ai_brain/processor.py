"""Unified AI Brain processor used by backend services."""

from __future__ import annotations

from typing import Any

from pydantic import ValidationError

from .decision import StudioDecisionEngine
from .decision_schemas import (
    ArtistSuggestion,
    DecisionHistoryExample,
    NextAction,
    StudioDecisionContext,
    StudioDecisionFeedback,
    StudioDecisionOutput,
    StudioLearningRecord,
)
from .errors import AnalysisPipelineError
from .extraction import TattooTextExtractor
from .routing import TattooRouter
from .schemas import AIExtractionOutput, Message, TattooInquiryInput
from .vision import TattooVisionAnalyzer

_AUTOMATION_PAUSED_DRAFT = "A staff member will reply to you shortly."


class StudioAIBrain:
    """Single entry-point for tattoo inquiry analysis."""

    def __init__(
        self,
        vision_analyzer: TattooVisionAnalyzer | None = None,
        text_extractor: TattooTextExtractor | None = None,
        router: TattooRouter | None = None,
        decision_engine: StudioDecisionEngine | None = None,
    ) -> None:
        self.vision_analyzer = vision_analyzer or TattooVisionAnalyzer()
        self.text_extractor = text_extractor or TattooTextExtractor()
        self.router = router or TattooRouter()
        self.decision_engine = decision_engine or StudioDecisionEngine()

    def process_inquiry(
        self,
        current_message: str | None = None,
        new_image_urls: list[str] | None = None,
        existing_db_state: dict[str, Any] | None = None,
        recent_chat_history: list[Message] | None = None,
        *,
        text: str | None = None,
        image_urls: list[str] | None = None,
    ) -> AIExtractionOutput:
        """Process latest-message and supplied hybrid context.

        The text and image_urls keywords remain temporarily supported until
        the API endpoint migrates to the canonical payload in Step 4.
        """
        inquiry = self._build_inquiry(
            current_message=current_message,
            new_image_urls=new_image_urls,
            existing_db_state=existing_db_state,
            recent_chat_history=recent_chat_history,
            text=text,
            image_urls=image_urls,
        )
        if self._is_automation_paused(inquiry):
            return self._build_automation_paused_output(inquiry)

        vision_output = self.vision_analyzer.analyze_images(
            inquiry.new_image_urls
        )
        extracted = self.text_extractor.extract(
            current_message=inquiry.current_message,
            style_tags=vision_output.style_tags,
            visual_color_preference=vision_output.color_preference,
            new_image_urls=inquiry.new_image_urls,
            existing_db_state=inquiry.existing_db_state,
            recent_chat_history=inquiry.recent_chat_history,
        )
        return self.router.route(
            extracted,
            current_message=inquiry.current_message,
            recent_chat_history=inquiry.recent_chat_history,
            existing_db_state=inquiry.existing_db_state,
        )

    def process_studio_decision(
        self,
        inquiry: TattooInquiryInput,
        context: StudioDecisionContext,
    ) -> StudioDecisionOutput:
        """Create an internal decision from caller-supplied studio context."""
        analysis = self.process_inquiry(
            current_message=inquiry.current_message,
            new_image_urls=inquiry.new_image_urls,
            existing_db_state=inquiry.existing_db_state,
            recent_chat_history=inquiry.recent_chat_history,
        )
        if self._is_automation_paused(inquiry):
            return self._build_automation_paused_decision(analysis)
        return self.decision_engine.decide(
            analysis=analysis,
            current_message=inquiry.current_message,
            context=context,
        )

    def _is_automation_paused(self, inquiry: TattooInquiryInput) -> bool:
        """Return whether backend state explicitly disables automation."""
        return inquiry.existing_db_state.get("automation_paused") is True

    def _build_automation_paused_output(
        self,
        inquiry: TattooInquiryInput,
    ) -> AIExtractionOutput:
        """Return a deterministic hold response without calling AI services."""
        return AIExtractionOutput(
            tattoo_idea=inquiry.current_message,
            style_tags=["unknown"],
            placement="",
            size_estimate_cm="",
            color_preference="",
            suggested_artist="Unclear",
            confidence_level="low",
            ai_reasoning=(
                "Automation paused: manual staff response required."
            ),
            missing_information=[],
            risk_level="high",
            draft_reply=_AUTOMATION_PAUSED_DRAFT,
            auto_reply_allowed=False,
            telegram_review_required=True,
        )

    def _build_automation_paused_decision(
        self,
        analysis: AIExtractionOutput,
    ) -> StudioDecisionOutput:
        """Return a staff-only decision without retrieval or pricing work."""
        reasoning = "Automation paused: manual staff response required."
        return StudioDecisionOutput(
            analysis=analysis,
            artist_suggestion=ArtistSuggestion(
                artist_key=None,
                artist_name=None,
                confidence_level="low",
                reasoning=reasoning,
                source="unresolved",
            ),
            suggested_next_action=NextAction(
                action="artist_review",
                reason=reasoning,
                priority="high",
                requires_human_review=True,
            ),
            internal_price_estimate=None,
            applied_history_example_ids=[],
        )

    def build_learning_record(
        self,
        inquiry: TattooInquiryInput,
        context: StudioDecisionContext,
        decision: StudioDecisionOutput,
        feedback: StudioDecisionFeedback,
    ) -> StudioLearningRecord:
        """Build a feedback record without storing or transmitting it."""
        return self.decision_engine.build_learning_record(
            inquiry=inquiry,
            context=context,
            decision=decision,
            feedback=feedback,
        )

    def build_history_example(
        self,
        record: StudioLearningRecord,
        example_id: str,
    ) -> DecisionHistoryExample:
        """Convert a completed record into future decision evidence."""
        return self.decision_engine.build_history_example(
            record=record,
            example_id=example_id,
        )

    def _build_inquiry(
        self,
        current_message: str | None,
        new_image_urls: list[str] | None,
        existing_db_state: dict[str, Any] | None,
        recent_chat_history: list[Message] | None,
        text: str | None,
        image_urls: list[str] | None,
    ) -> TattooInquiryInput:
        """Validate canonical values with temporary legacy fallbacks."""
        resolved_message = current_message
        if resolved_message is None:
            resolved_message = text

        resolved_images = new_image_urls
        if resolved_images is None:
            resolved_images = image_urls or []

        payload: dict[str, Any] = {
            "current_message": resolved_message,
            "new_image_urls": resolved_images,
            "existing_db_state": existing_db_state or {},
            "recent_chat_history": recent_chat_history or [],
        }
        try:
            return TattooInquiryInput.model_validate(payload)
        except ValidationError as exc:
            raise AnalysisPipelineError("Invalid inquiry payload.") from exc


if __name__ == "__main__":
    brain = StudioAIBrain()
    mock_text = (
        "Hey, I want a tiny fine-line lotus on my inner wrist. "
        "Maybe around 4 cm. Black ink only. Can I come next Tuesday?"
    )
    mock_images = [
        "https://example.com/reference-1.jpg",
        "https://example.com/reference-2.jpg",
    ]

    try:
        output = brain.process_inquiry(
            current_message=mock_text,
            new_image_urls=mock_images,
            existing_db_state={},
            recent_chat_history=[],
        )
        print(output.model_dump_json(indent=2))
    except Exception as exc:
        print(f"Processor test run failed: {exc}")
