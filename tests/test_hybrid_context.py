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
