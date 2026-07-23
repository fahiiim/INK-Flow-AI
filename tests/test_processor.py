"""Tests for the StudioAIBrain unified processor."""

from __future__ import annotations

from ai_brain.processor import StudioAIBrain
from ai_brain.schemas import AIExtractionOutput, Message, TattooExtractionDraft


class StubVisionAnalyzer:
    """Deterministic vision analyzer stub for processor tests."""

    def __init__(self, tags: list[str]) -> None:
        self._tags = tags
        self.calls: list[list[str]] = []

    def analyze_styles(self, image_urls: list[str]) -> list[str]:
        """Return pre-defined tags and track invocation input."""
        self.calls.append(image_urls)
        return self._tags


class StubTextExtractor:
    """Deterministic text extractor stub for processor tests."""

    def __init__(self, draft: TattooExtractionDraft) -> None:
        self._draft = draft
        self.calls: list[dict[str, object]] = []

    def extract(
        self,
        current_message: str,
        style_tags: list[str],
        new_image_urls: list[str] | None = None,
        existing_db_state: dict[str, object] | None = None,
        recent_chat_history: list[Message] | None = None,
    ) -> TattooExtractionDraft:
        """Return pre-defined draft and track invocation arguments."""
        self.calls.append(
            {
                "current_message": current_message,
                "style_tags": style_tags,
                "new_image_urls": new_image_urls or [],
                "existing_db_state": existing_db_state or {},
                "recent_chat_history": recent_chat_history or [],
            }
        )
        return self._draft


class StubRouter:
    """Deterministic router stub for processor tests."""

    def __init__(self, output: AIExtractionOutput) -> None:
        self._output = output
        self.calls: list[TattooExtractionDraft] = []
        self.context_calls: list[tuple[str, list[Message]]] = []

    def route(
        self,
        extracted: TattooExtractionDraft,
        current_message: str = "",
        recent_chat_history: list[Message] | None = None,
    ) -> AIExtractionOutput:
        """Return pre-defined output and track invocation input."""
        self.calls.append(extracted)
        self.context_calls.append(
            (current_message, list(recent_chat_history or []))
        )
        return self._output


def test_process_inquiry_low_risk_fine_line_request() -> None:
    """Validate low-risk fine-line flow and orchestration calls."""
    vision = StubVisionAnalyzer(tags=["fine-line", "minimal"])
    extraction = StubTextExtractor(
        draft=TattooExtractionDraft(
            tattoo_idea="Small lotus linework",
            style_tags=["fine-line", "minimal"],
            placement="inner wrist",
            size_estimate_cm="10",
            color_preference="black-and-grey",
            missing_information=[],
        )
    )
    router = StubRouter(
        output=AIExtractionOutput(
            tattoo_idea="Small lotus linework",
            style_tags=["fine-line", "minimal"],
            placement="inner wrist",
            size_estimate_cm="10",
            color_preference="black-and-grey",
            suggested_artist="Nina",
            confidence_level="high",
            ai_reasoning="Fine-line and minimal style align strongly with Nina.",
            missing_information=[],
            risk_level="low",
            draft_reply=(
                "Thanks for sharing the details. This looks like a great "
                "fine-line piece for Nina."
            ),
        )
    )

    brain = StudioAIBrain(
        vision_analyzer=vision,
        text_extractor=extraction,
        router=router,
    )
    history = [
        Message(role="user", content="I originally wanted a 5cm lotus."),
        Message(role="assistant", content="A 5cm wrist piece is possible."),
    ]

    result = brain.process_inquiry(
        current_message="Actually make the fine-line lotus 10cm.",
        new_image_urls=["https://example.com/new-reference.jpg"],
        existing_db_state={"size": "5cm", "placement": "inner wrist"},
        recent_chat_history=history,
    )

    assert result.suggested_artist == "Nina"
    assert result.risk_level == "low"
    assert result.style_tags == ["fine-line", "minimal"]
    assert result.size_estimate_cm == "10"
    assert vision.calls == [["https://example.com/new-reference.jpg"]]
    assert extraction.calls[0]["style_tags"] == ["fine-line", "minimal"]
    assert extraction.calls[0]["current_message"] == (
        "Actually make the fine-line lotus 10cm."
    )
    assert extraction.calls[0]["existing_db_state"] == {
        "size": "5cm",
        "placement": "inner wrist",
    }
    assert extraction.calls[0]["recent_chat_history"] == history
    assert len(router.calls) == 1
    assert router.context_calls == [
        ("Actually make the fine-line lotus 10cm.", history)
    ]
    assert result.draft_reply == (
        "Thanks for sharing the details. This looks like a great "
        "fine-line piece for Nina."
    )


def test_process_inquiry_high_risk_pricing_request() -> None:
    """Validate high-risk pricing intent reaches final output unchanged."""
    vision = StubVisionAnalyzer(tags=["traditional"])
    extraction = StubTextExtractor(
        draft=TattooExtractionDraft(
            tattoo_idea="Traditional wolf chest piece with budget question",
            style_tags=["traditional"],
            placement="chest",
            size_estimate_cm="22",
            color_preference="full color",
            missing_information=["preferred date"],
        )
    )
    router = StubRouter(
        output=AIExtractionOutput(
            tattoo_idea="Traditional wolf chest piece with budget question",
            style_tags=["traditional"],
            placement="chest",
            size_estimate_cm="22",
            color_preference="full color",
            suggested_artist="Hoss",
            confidence_level="high",
            ai_reasoning=(
                "Traditional large-scale work routes to Hoss. Pricing intent "
                "raises risk."
            ),
            missing_information=["preferred date"],
            risk_level="high",
            draft_reply=(
                "Thanks for your message. Pricing and scope details will be "
                "reviewed by our senior team first."
            ),
        )
    )

    brain = StudioAIBrain(
        vision_analyzer=vision,
        text_extractor=extraction,
        router=router,
    )

    result = brain.process_inquiry(
        current_message=(
            "How much would a full traditional wolf chest tattoo cost? "
            "I want this booked soon."
        ),
        new_image_urls=["https://example.com/traditional-reference.jpg"],
        existing_db_state={},
        recent_chat_history=[],
    )

    assert result.suggested_artist == "Hoss"
    assert result.risk_level == "high"
    assert "preferred date" in result.missing_information
    assert extraction.calls[0]["style_tags"] == ["traditional"]
    assert len(router.calls) == 1


def test_process_inquiry_request_with_no_images() -> None:
    """Validate no-image scenario still completes pipeline with unknown style."""
    vision = StubVisionAnalyzer(tags=["unknown"])
    extraction = StubTextExtractor(
        draft=TattooExtractionDraft(
            tattoo_idea="Client asks for script quote without references",
            style_tags=["unknown"],
            placement="",
            size_estimate_cm="",
            color_preference="",
            missing_information=[
                "size in cm",
                "placement",
                "reference images",
                "color preference",
                "preferred date",
            ],
        )
    )
    router = StubRouter(
        output=AIExtractionOutput(
            tattoo_idea="Client asks for script quote without references",
            style_tags=["unknown"],
            placement="",
            size_estimate_cm="",
            color_preference="",
            suggested_artist="Unclear",
            confidence_level="low",
            ai_reasoning="No visual references and incomplete intake details.",
            missing_information=[
                "size in cm",
                "placement",
                "reference images",
                "color preference",
                "preferred date",
            ],
            risk_level="low",
            draft_reply=(
                "Thanks for reaching out. Please share placement, size in cm, "
                "color preference, preferred date, and reference images."
            ),
        )
    )

    brain = StudioAIBrain(
        vision_analyzer=vision,
        text_extractor=extraction,
        router=router,
    )

    result = brain.process_inquiry(
        current_message="I want a script tattoo but have no images yet.",
        new_image_urls=[],
        existing_db_state={},
        recent_chat_history=[],
    )

    assert result.style_tags == ["unknown"]
    assert result.suggested_artist == "Unclear"
    assert result.risk_level == "low"
    assert vision.calls == [[]]
    assert extraction.calls[0]["new_image_urls"] == []
    assert len(router.calls) == 1
