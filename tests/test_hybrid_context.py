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
            "preferred_time": "14:30",
            "preferred_artist": "Silva",
            "appointment_type": "studio_visit",
            "tattoo_project_type": "new tattoo",
            "lead": {"name": "Maruf Hossain"},
        },
        recent_chat_history=history,
    )

    assert result.size_estimate_cm == "10cm"
    assert result.placement == "inner wrist"
    assert result.color_preference == "black-and-grey"
    assert result.date == "2026-08-15"
    assert result.time == "14:30"
    assert result.client_name == "Maruf Hossain"
    assert result.preferred_artist == "Silva"
    assert result.appointment_type == "studio_visit"
    assert result.availability == "2026-08-15"
    assert result.tattoo_project_type == "new tattoo"
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


def test_preferred_date_and_time_are_normalized_in_fallback() -> None:
    """Scheduling details remain available when the model call fails."""
    extractor = TattooTextExtractor(
        llm=cast(ChatOpenAI, FailingExtractionLLM()),
    )

    result = extractor.extract(
        current_message=(
            "I want a fine-line lotus tattoo on 2026-09-04 at 14:30."
        ),
        style_tags=["fine-line"],
        existing_db_state={},
        recent_chat_history=[],
    )

    assert result.date == "2026-09-04"
    assert result.time == "14:30"
    assert "preferred dates or availability" not in result.missing_information


def test_nested_intake_supplies_preferred_date_and_time() -> None:
    """Existing backend appointment fields populate the response contract."""
    extractor = TattooTextExtractor(
        llm=cast(ChatOpenAI, FailingExtractionLLM()),
    )

    result = extractor.extract(
        current_message="I still want the lotus tattoo.",
        style_tags=["fine-line"],
        existing_db_state={
            "intake": {
                "appointment_date": "2026-09-04",
                "appointment_time": "14:30",
            }
        },
        recent_chat_history=[],
    )

    assert result.date == "2026-09-04"
    assert result.time == "14:30"


def test_full_intake_fields_are_extracted_from_one_client_message() -> None:
    """Every new required answer can be retained without another question."""
    extractor = TattooTextExtractor(
        llm=cast(ChatOpenAI, FailingExtractionLLM()),
    )

    result = extractor.extract(
        current_message=(
            "My full name is Alex Morgan. I want a new tattoo of a "
            "fine-line black and grey rose, 10 cm on my forearm. I prefer "
            "Silva and need an online appointment. I am available on "
            "2026-09-18 at 14:30."
        ),
        style_tags=["fine-line"],
        new_image_urls=["https://example.com/rose-reference.jpg"],
        existing_db_state={},
        recent_chat_history=[],
    )

    assert result.client_name == "Alex Morgan"
    assert result.size_estimate_cm == "10 cm"
    assert result.placement == "forearm"
    assert result.color_preference == "black-and-grey"
    assert result.preferred_artist == "Silva"
    assert result.appointment_type == "online"
    assert result.availability == "2026-09-18 at 14:30"
    assert result.tattoo_project_type == "new tattoo"
    assert result.date == "2026-09-18"
    assert result.time == "14:30"
    assert result.missing_information == []


def test_price_only_opening_still_requires_tattoo_idea() -> None:
    """A first-message cost question is not mistaken for a design concept."""
    extractor = TattooTextExtractor(
        llm=cast(ChatOpenAI, FailingExtractionLLM()),
    )

    result = extractor.extract(
        current_message="Hi, how much will a tattoo cost?",
        style_tags=["unknown"],
        existing_db_state={"lead": {"name": "Maruf Hossain"}},
        recent_chat_history=[],
    )

    assert result.client_name == "Maruf Hossain"
    assert "client full name" not in result.missing_information
    assert "tattoo idea" in result.missing_information


def test_short_name_answer_is_resolved_from_question_context() -> None:
    """A bare full-name reply is retained during deterministic fallback."""
    extractor = TattooTextExtractor(
        llm=cast(ChatOpenAI, FailingExtractionLLM()),
    )

    result = extractor.extract(
        current_message="Maruf Hossain",
        style_tags=["unknown"],
        existing_db_state={},
        recent_chat_history=[
            Message(role="assistant", content="What is your full name?"),
        ],
    )

    assert result.client_name == "Maruf Hossain"
    assert "client full name" not in result.missing_information
    assert "tattoo idea" in result.missing_information
