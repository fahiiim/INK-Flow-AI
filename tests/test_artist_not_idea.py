"""An artist name in the client's message is never saved as the idea."""

from __future__ import annotations

from typing import Any, cast

import pytest
from langchain_openai import ChatOpenAI

from ai_brain.extraction import TattooTextExtractor
from ai_brain.schemas import Message


class FailingLLM:
    """Force the deterministic extraction path."""

    def invoke(self, *_args: Any, **_kwargs: Any) -> Any:
        raise RuntimeError("LLM unavailable")


def _extractor() -> TattooTextExtractor:
    return TattooTextExtractor(llm=cast(ChatOpenAI, FailingLLM()))


@pytest.mark.parametrize(
    "message",
    [
        "I would like Lana for the tattoo.",
        "I need Lana for my tattoo.",
        "I want Lana to do my tattoo.",
        "I want Lana for this tattoo.",
        "I would like to have Lana as my tattoo artist.",
    ],
)
def test_artist_request_keeps_the_existing_idea(message: str) -> None:
    """Choosing an artist does not overwrite the saved design."""
    result = _extractor().extract(
        current_message=message,
        style_tags=["traditional", "floral"],
        existing_db_state={
            "lead": {"name": "Fahim Sarker", "source": "outlook"},
            "intake": {"tattoo_idea": "Rose with a dagger", "placement": "back"},
        },
    )

    assert result.tattoo_idea == "Rose with a dagger"
    assert result.preferred_artist == "Lana"


def test_stored_artist_name_idea_is_replaced_from_history() -> None:
    """A previously saved artist-name idea is ignored, not repeated."""
    result = _extractor().extract(
        current_message="That's a new tattoo.",
        style_tags=["traditional", "floral"],
        existing_db_state={
            "lead": {"name": "Fahim Sarker", "source": "outlook"},
            "intake": {"tattoo_idea": "Lana", "preferred_artist": "Lana"},
        },
        recent_chat_history=[
            Message(
                role="user",
                content="I want a traditional rose tattoo on my back.",
            ),
            Message(role="assistant", content="Lovely! Any preferred artist?"),
            Message(role="user", content="I would like Lana for the tattoo."),
        ],
    )

    assert result.tattoo_idea.casefold() != "lana"
    assert "rose" in result.tattoo_idea.casefold()
