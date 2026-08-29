"""Tests for reference-image style and color intake behavior."""

from __future__ import annotations

import json
from types import SimpleNamespace
from typing import cast

from langchain_openai import ChatOpenAI

from ai_brain.extraction import TattooTextExtractor
from ai_brain.reply import ConversationReplyComposer
from ai_brain.vision import TattooVisionAnalyzer


class BlankExtractionLLM:
    """Return blank style-adjacent fields for deterministic reconciliation."""

    def invoke(self, messages: object) -> SimpleNamespace:
        """Return valid extraction JSON with every intake item missing."""
        return SimpleNamespace(
            content=json.dumps(
                {
                    "tattoo_idea": "Reference tattoo",
                    "placement": "",
                    "size_estimate_cm": "",
                    "color_preference": "",
                    "missing_information": [
                        "size in cm",
                        "placement",
                        "reference images",
                        "tattoo style",
                        "color preference",
                        "preferred date",
                    ],
                }
            )
        )


class FailingExtractionLLM:
    """Force deterministic extraction fallback for generic requests."""

    def invoke(self, messages: object) -> object:
        """Simulate an unavailable extraction provider."""
        raise RuntimeError("Simulated provider failure")


def _extractor() -> TattooTextExtractor:
    """Return an extractor with no external model dependency."""
    return TattooTextExtractor(
        llm=cast(ChatOpenAI, BlankExtractionLLM()),
    )


def test_vision_parser_returns_approved_style_and_color() -> None:
    """Vision JSON is normalized into the strict image-analysis contract."""
    analyzer = TattooVisionAnalyzer(
        llm=cast(ChatOpenAI, BlankExtractionLLM()),
    )
    output = analyzer._parse_vision_output(
        json.dumps(
            {
                "style_tags": ["watercolor", "floral"],
                "color_preference": "full color",
            }
        )
    )

    assert output.style_tags == ["watercolor", "floral"]
    assert output.color_preference == "color"


def test_recognized_image_removes_style_color_and_reference_missing() -> None:
    """Known visual evidence is retained and prepared for confirmation."""
    result = _extractor().extract(
        current_message="",
        style_tags=["watercolor", "floral"],
        visual_color_preference="color",
        new_image_urls=["https://example.com/reference.jpg"],
    )

    assert result.style_tags == ["watercolor", "floral"]
    assert result.color_preference == "color"
    assert "reference images" not in result.missing_information
    assert "tattoo style" not in result.missing_information
    assert "color preference" not in result.missing_information

    reply = ConversationReplyComposer().compose_validation(
        extracted=result,
        current_message="",
        recent_chat_history=[],
        risk_level="low",
    )
    assert "full-colour watercolor and floral tattoo" in reply
    assert "Does that sound right" in reply


def test_unclear_image_asks_style_and_color_manually() -> None:
    """An unclear image produces two focused visual follow-up questions."""
    result = _extractor().extract(
        current_message="",
        style_tags=["unknown"],
        visual_color_preference="unknown",
        new_image_urls=["https://example.com/unclear.jpg"],
    )
    reply = ConversationReplyComposer().compose(
        extracted=result,
        current_message="",
        recent_chat_history=[],
        risk_level="low",
    )

    assert "tattoo style" in result.missing_information
    assert "color preference" in result.missing_information
    assert "reference images" not in result.missing_information
    assert "What tattoo style would you like?" in reply
    assert "Would you like black-and-grey or colour?" in reply
    assert reply.count("?") == 2


def test_no_image_requests_reference_before_manual_visual_questions() -> None:
    """A missing image is requested before asking style and color directly."""
    result = _extractor().extract(
        current_message="I want a tattoo.",
        style_tags=["unknown"],
        visual_color_preference="unknown",
        new_image_urls=[],
    )
    reply = ConversationReplyComposer().compose(
        extracted=result,
        current_message="I want a tattoo.",
        recent_chat_history=[],
        risk_level="low",
    )

    assert "Do you have a reference image you can send?" in reply
    assert reply.count("?") == 1


def test_generic_tattoo_request_asks_for_concept_before_reference() -> None:
    """A vague tattoo request cannot be complete without an actual idea."""
    extractor = TattooTextExtractor(
        llm=cast(ChatOpenAI, FailingExtractionLLM()),
    )

    result = extractor.extract(
        current_message="I want a tattoo.",
        style_tags=["unknown"],
        new_image_urls=[],
    )
    reply = ConversationReplyComposer().compose(
        extracted=result,
        current_message="I want a tattoo.",
        recent_chat_history=[],
        risk_level="low",
    )

    assert "tattoo idea" in result.missing_information
    assert reply == "Got it. What tattoo idea or design do you have in mind?"
    assert "reference image" not in reply


def test_reference_image_decline_is_not_requested_again() -> None:
    """An unavailable reference is resolved instead of causing a loop."""
    result = _extractor().extract(
        current_message="I don't have a reference image.",
        style_tags=["unknown"],
        new_image_urls=[],
    )
    reply = ConversationReplyComposer().compose(
        extracted=result,
        current_message="I don't have a reference image.",
        recent_chat_history=[],
        risk_level="low",
    )

    assert "reference images" not in result.missing_information
    assert "reference image" not in reply.casefold()


def test_style_stated_in_text_is_not_requested_manually() -> None:
    """Explicit approved style words supplement unknown vision output."""
    result = _extractor().extract(
        current_message="I want a fine-line tattoo.",
        style_tags=["unknown"],
        visual_color_preference="unknown",
        new_image_urls=[],
    )

    assert result.style_tags == ["fine-line"]
    assert "tattoo style" not in result.missing_information
