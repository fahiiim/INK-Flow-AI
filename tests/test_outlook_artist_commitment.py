"""Regression tests for Outlook intakes that select an artist conversationally."""

from __future__ import annotations

import json
from types import SimpleNamespace
from typing import cast
from unittest.mock import Mock

import pytest
from langchain_openai import ChatOpenAI

from ai_brain.extraction import TattooTextExtractor
from ai_brain.routing import TattooRouter
from ai_brain.schemas import Message, TattooExtractionDraft
from ai_brain.vector_store import VectorStoreManager


class StaticExtractionLLM:
    """Return the fields a well-behaved model extracts for the final turn."""

    def invoke(self, messages: object) -> SimpleNamespace:
        """Return a complete extraction payload."""
        payload = {
            "client_name": "Fahim Sarker",
            "tattoo_idea": "black abstract symbol with curved strokes and dots",
            "placement": "nose",
            "size_estimate_cm": "3 cm",
            "color_preference": "black-and-grey",
            "date": "2026-10-12",
            "time": "",
            "preferred_artist": "Sandra",
            "appointment_type": "studio_visit",
            "availability": "next Monday",
            "tattoo_project_type": "new tattoo",
            "artist_preference_mode": "specific",
            "missing_information": [],
        }
        return SimpleNamespace(content=json.dumps(payload))


class FailingLLM:
    """LLM stub that always forces deterministic fallback logic."""

    def invoke(self, messages: object) -> object:
        """Raise a predictable provider failure."""
        raise RuntimeError("Simulated LLM failure")


_NOSE_THREAD = [
    Message(
        role="user",
        content=(
            "Hello, I want a tattoo on my nose just like this picture. "
            "how much it will cost"
        ),
    ),
    Message(
        role="assistant",
        content=(
            "Dear Fahim,\n\nDo you have a preferred artist, or are you open "
            "to a recommendation?\n\nKind regards,\nTattoo Hysteria"
        ),
    ),
    Message(
        role="user",
        content=(
            "Yes, it's about 3cm. And for the artists. I don't know anyone.\n"
            "Could you please tell me who the artists are and brief details "
            "about them"
        ),
    ),
    Message(
        role="assistant",
        content=(
            "Dear Fahim,\n\nWould you like to request Sandra, or would you "
            "prefer to explore the other artists first? Would you prefer an "
            "online consultation or a studio visit?\n\nKind regards,\n"
            "Tattoo Hysteria"
        ),
    ),
    Message(
        role="user",
        content=(
            "I think I'll move forward with Sandra then.\n"
            "And I'd visit the studio next Monday."
        ),
    ),
    Message(
        role="assistant",
        content=(
            "Dear Fahim,\n\nJust one last detail: will this be a new tattoo "
            "on untattooed skin, or a cover-up or rework of an existing "
            "tattoo?\n\nKind regards,\nTattoo Hysteria"
        ),
    ),
]
_NOSE_STATE = {
    "lead": {
        "name": "Fahim Sarker",
        "email": "fahimsarker0805@gmail.com",
        "source": "outlook",
    },
    "intake": {
        "style_tags": ["abstract"],
        "reference_images": ["https://example.com/image.png"],
    },
}


@pytest.mark.parametrize(
    "message",
    [
        "I think I'll move forward with Sandra then.",
        "I think I’ll go ahead with Sandra.",
        "Let's proceed with Sandra.",
        "I'd like to request Sandra please.",
        "I'll stick with Sandra.",
        "I'd like Sandra for this one.",
        "Sandra sounds good.",
    ],
)
def test_conversational_artist_commitment_is_a_selection(message: str) -> None:
    """Natural commitment phrases select the named artist."""
    extractor = TattooTextExtractor(llm=cast(ChatOpenAI, FailingLLM()))

    assert extractor._extract_preferred_artist_from_text(message) == "Sandra"


@pytest.mark.parametrize(
    "message",
    [
        "Could you tell me more about Sandra?",
        "Is Sandra available next week?",
        "I don't know Sandra's work yet.",
    ],
)
def test_artist_mentions_without_commitment_are_not_selections(
    message: str,
) -> None:
    """Questions about an artist do not silently select that artist."""
    extractor = TattooTextExtractor(llm=cast(ChatOpenAI, FailingLLM()))

    assert extractor._extract_preferred_artist_from_text(message) == ""


def test_completed_outlook_nose_thread_has_no_missing_artist() -> None:
    """The final project-type answer completes an artist-selected intake."""
    extractor = TattooTextExtractor(
        llm=cast(ChatOpenAI, StaticExtractionLLM()),
    )

    draft = extractor.extract(
        current_message="thats a new tattoo",
        style_tags=[],
        existing_db_state=_NOSE_STATE,
        recent_chat_history=_NOSE_THREAD,
    )

    assert draft.preferred_artist == "Sandra"
    assert draft.artist_preference_mode == "specific"
    assert draft.tattoo_project_type == "new tattoo"
    assert draft.missing_information == []


def test_completed_outlook_nose_thread_is_high_risk() -> None:
    """A complete Outlook intake reaches staff review instead of auto-reply."""
    extractor = TattooTextExtractor(
        llm=cast(ChatOpenAI, StaticExtractionLLM()),
    )
    draft = extractor.extract(
        current_message="thats a new tattoo",
        style_tags=[],
        existing_db_state=_NOSE_STATE,
        recent_chat_history=_NOSE_THREAD,
    )
    vector_store = Mock(spec=VectorStoreManager)
    vector_store.records = tuple(range(10))
    vector_store.search_similar_cases.return_value = []
    router = TattooRouter(
        llm=cast(ChatOpenAI, FailingLLM()),
        vector_store=cast(VectorStoreManager, vector_store),
    )

    result = router.route(
        draft,
        current_message="thats a new tattoo",
        recent_chat_history=_NOSE_THREAD,
        existing_db_state=_NOSE_STATE,
        message_source="outlook",
    )

    assert result.risk_level == "high"
    assert result.staff_review_required is True
    assert result.telegram_review_required is True
    assert result.auto_reply_allowed is False
    assert result.auto_reply is False


def test_llm_draft_cannot_claim_completion_while_details_are_missing() -> None:
    """A question-free draft is rejected when intake details remain."""
    router = TattooRouter(llm=cast(ChatOpenAI, FailingLLM()))
    draft = TattooExtractionDraft(
        tattoo_idea="black abstract symbol",
        style_tags=["abstract"],
        placement="nose",
        size_estimate_cm="3 cm",
        color_preference="black-and-grey",
        appointment_type="studio_visit",
        availability="next Monday",
        tattoo_project_type="new tattoo",
        missing_information=["preferred artist"],
    )

    with pytest.raises(ValueError, match="omits a question"):
        router._validate_draft_reply(
            draft_reply=(
                "Dear Fahim,\n\nWe have the details needed for the studio to "
                "review your request.\n\nKind regards,\nTattoo Hysteria"
            ),
            extracted=draft,
            message_source="outlook",
            current_message="thats a new tattoo",
        )
