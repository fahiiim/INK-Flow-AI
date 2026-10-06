"""Tests that inferred intake data never becomes fabricated client choice."""

from __future__ import annotations

import json
from types import SimpleNamespace
from typing import cast

from langchain_openai import ChatOpenAI

from ai_brain.extraction import TattooTextExtractor


class HallucinatingChoiceLLM:
    """Return unsupported choices to exercise deterministic safeguards."""

    def invoke(self, messages: object) -> SimpleNamespace:
        """Invent artist and project type values the client never supplied."""
        return SimpleNamespace(
            content=json.dumps(
                {
                    "client_name": "Fahim Sarker",
                    "tattoo_idea": "Solid black tongue coverage",
                    "placement": "tongue",
                    "size_estimate_cm": "3 cm",
                    "color_preference": "black-and-grey",
                    "preferred_artist": "Hoss",
                    "appointment_type": "studio_visit",
                    "availability": "2026-10-15",
                    "tattoo_project_type": "new tattoo",
                    "party_size": 1,
                    "multi_entity_detected": True,
                    "complexity_notes": "Studio feasibility review needed.",
                    "artist_preference_mode": "specific",
                    "missing_information": [],
                }
            )
        )


def test_reference_led_style_and_unselected_choices_remain_truthful() -> None:
    """Reference style is recorded without inventing artist or project type."""
    extractor = TattooTextExtractor(
        llm=cast(ChatOpenAI, HallucinatingChoiceLLM()),
    )

    result = extractor.extract(
        current_message=(
            "The style should be similar to the reference image. Which "
            "artist would you recommend?"
        ),
        style_tags=["unknown"],
        existing_db_state={
            "lead": {"name": "Fahim Sarker"},
            "intake": {
                "tattoo_idea": "Solid black tongue coverage",
                "placement": "tongue",
                "size_estimate_cm": "3 cm",
                "color_preference": "black-and-grey",
                "appointment_type": "studio_visit",
                "availability": "2026-10-15",
                "previous_image_urls": [
                    "https://example.com/reference.png"
                ],
            },
        },
    )

    assert result.style_tags == ["unknown"]
    assert "reference-led design" in result.complexity_notes
    assert "tattoo style" not in result.missing_information
    assert result.preferred_artist == ""
    assert result.artist_preference_mode == "recommend"
    assert "preferred artist" in result.missing_information
    assert result.tattoo_project_type == ""
    assert "tattoo project type" in result.missing_information
    assert result.multi_entity_detected is False


def test_named_vision_style_resolves_reference_led_request() -> None:
    """A reliable vision tag is retained as the actual style evidence."""
    extractor = TattooTextExtractor(
        llm=cast(ChatOpenAI, HallucinatingChoiceLLM()),
    )

    result = extractor.extract(
        current_message="Please match the style in the reference image.",
        style_tags=["traditional"],
        new_image_urls=["https://example.com/reference.png"],
    )

    assert result.style_tags == ["traditional"]
    assert "reference-led design" not in result.complexity_notes
    assert "tattoo style" not in result.missing_information


def test_artist_information_question_does_not_select_that_artist() -> None:
    """Mentioning an artist for information is not a preference choice."""
    extractor = TattooTextExtractor(
        llm=cast(ChatOpenAI, HallucinatingChoiceLLM()),
    )

    result = extractor.extract(
        current_message="Can you tell me about Hoss and his specialties?",
        style_tags=["traditional"],
        existing_db_state={
            "lead": {"name": "Fahim Sarker"},
            "intake": {
                "tattoo_idea": "Eagle",
                "placement": "chest",
                "size_estimate_cm": "15 cm",
                "color_preference": "black-and-grey",
            },
        },
    )

    assert result.client_intent == "artist_guidance"
    assert result.preferred_artist == ""
    assert "preferred artist" in result.missing_information
