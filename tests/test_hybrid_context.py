"""Hybrid-context extraction tests."""

from __future__ import annotations

from types import SimpleNamespace
from typing import cast

from langchain_openai import ChatOpenAI

from ai_brain.extraction import TattooTextExtractor
from ai_brain.schemas import Message


class StaticExtractionLLM:
    """Return a deterministic structured extraction response."""

    def invoke(self, messages: object) -> SimpleNamespace:
        """Return current-message values while leaving DB fields blank."""
        content = (
            '{"tattoo_idea":"fine-line lotus",'
            '"placement":"",'
            '"size_estimate_cm":"10cm",'
            '"color_preference":"",'
            '"missing_information":['
            '"placement","reference images",'
            '"color preference","preferred date"]}'
        )
        return SimpleNamespace(content=content)


class StaleExtractionLLM:
    """Return stale values to verify deterministic latest-turn correction."""

    def invoke(self, messages: object) -> SimpleNamespace:
        """Return valid JSON that incorrectly repeats database values."""
        content = (
            '{"tattoo_idea":"traditional dragon",'
            '"placement":"inner wrist",'
            '"size_estimate_cm":"5cm",'
            '"color_preference":"black-and-grey",'
            '"missing_information":[]}'
        )
        return SimpleNamespace(content=content)


class FailingExtractionLLM:
    """Raise a provider failure to exercise deterministic context fallback."""

    def invoke(self, messages: object) -> SimpleNamespace:
        """Simulate an unavailable extraction provider."""
        raise RuntimeError("simulated extraction failure")


def test_current_message_overrides_database_and_state_fills_blanks() -> None:
    """Current extraction wins while non-conflicting DB values are retained."""
    extractor = TattooTextExtractor(
        llm=cast(ChatOpenAI, StaticExtractionLLM()),
    )
    history = [
        Message(role="user", content="I originally asked for a 5cm lotus."),
        Message(role="assistant", content="We noted the 5cm size."),
    ]

    result = extractor.extract(
        current_message="Actually make it 10cm instead.",
        style_tags=["fine-line"],
        new_image_urls=[],
        existing_db_state={
            "size": "5cm",
            "placement": "inner wrist",
            "color_preference": "black-and-grey",
            "reference_images": ["https://example.com/old-reference.jpg"],
            "preferred_date": "2026-08-15",
        },
        recent_chat_history=history,
    )

    assert result.size_estimate_cm == "10cm"
    assert result.placement == "inner wrist"
    assert result.color_preference == "black-and-grey"
    assert result.missing_information == []


def test_extractor_accepts_image_only_message() -> None:
    """Image-only input receives a neutral internal caption for extraction."""
    extractor = TattooTextExtractor(
        llm=cast(ChatOpenAI, StaticExtractionLLM()),
    )

    result = extractor.extract(
        current_message="",
        style_tags=["fine-line"],
        new_image_urls=["https://example.com/whatsapp-image.jpg"],
        existing_db_state={},
        recent_chat_history=[],
    )

    assert result.style_tags == ["fine-line"]
    assert "reference images" not in result.missing_information


def test_current_message_deterministically_overrides_stale_model_values() -> None:
    """Explicit current facts override stale model, history, and DB values."""
    extractor = TattooTextExtractor(
        llm=cast(ChatOpenAI, StaleExtractionLLM()),
    )
    history = [
        Message(
            role="user",
            content="I wanted a 5cm fine-line wrist tattoo in black ink.",
        ),
    ]

    result = extractor.extract(
        current_message=(
            "Actually make it a 12cm traditional piece on my forearm in "
            "full color instead of fine-line."
        ),
        style_tags=["fine-line"],
        existing_db_state={
            "size": "5cm",
            "placement": "inner wrist",
            "color_preference": "black-and-grey",
        },
        recent_chat_history=history,
    )

    assert result.size_estimate_cm == "12cm"
    assert result.placement == "forearm"
    assert result.color_preference == "color"
    assert result.style_tags == ["traditional"]


def test_provider_failure_uses_current_then_history_then_database() -> None:
    """Fallback synthesis preserves source order without raising an error."""
    extractor = TattooTextExtractor(
        llm=cast(ChatOpenAI, FailingExtractionLLM()),
    )
    history = [
        Message(role="user", content="I would like it on my shoulder."),
    ]

    result = extractor.extract(
        current_message="Actually make it 10cm and full color.",
        style_tags=["geometric"],
        existing_db_state={
            "size": "5cm",
            "placement": "inner wrist",
            "color_preference": "black-and-grey",
        },
        recent_chat_history=history,
    )

    assert result.size_estimate_cm == "10cm"
    assert result.placement == "shoulder"
    assert result.color_preference == "color"
