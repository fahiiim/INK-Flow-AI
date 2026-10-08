"""WhatsApp reply style: first-message welcome and message length."""

from __future__ import annotations

import json
from types import SimpleNamespace
from typing import cast
from unittest.mock import Mock

import pytest
from langchain_openai import ChatOpenAI

from ai_brain.routing import TattooRouter
from ai_brain.schemas import Message, TattooExtractionDraft
from ai_brain.vector_store import VectorStoreManager


class DraftLLM:
    """Return routing JSON, then a fixed draft reply."""

    def __init__(self, draft_reply: str) -> None:
        self._responses = [
            json.dumps(
                {"confidence_level": "low", "ai_reasoning": "Test routing."}
            ),
            json.dumps({"draft_reply": draft_reply}),
        ]

    def invoke(self, messages: object) -> SimpleNamespace:
        """Return the next scripted response."""
        return SimpleNamespace(content=self._responses.pop(0))


def _router(draft_reply: str) -> TattooRouter:
    vector_store = Mock(spec=VectorStoreManager)
    vector_store.records = tuple(range(10))
    vector_store.search_similar_cases.return_value = []
    return TattooRouter(
        llm=cast(ChatOpenAI, DraftLLM(draft_reply)),
        vector_store=cast(VectorStoreManager, vector_store),
    )


def _palm_draft() -> TattooExtractionDraft:
    return TattooExtractionDraft(
        client_name="Fahim Sarker",
        tattoo_idea="abstract watercolor ribbons",
        style_tags=["abstract", "watercolor"],
        placement="palm",
        size_estimate_cm="",
        color_preference="color",
        missing_information=["size in cm", "preferred artist"],
    )


_STATE = {"lead": {"name": "Fahim Sarker", "source": "whatsapp"}}
_FIRST_MESSAGE = (
    "Okay, I need a colorful tattoo on the palm of my hand that matches "
    "this reference."
)


def test_first_whatsapp_reply_welcomes_the_client() -> None:
    """The first bot message opens with a Tattoo Hysteria welcome."""
    result = _router(
        "A colourful watercolor palm piece sounds lovely! "
        "Would around 6–10 cm suit you?"
    ).route(
        _palm_draft(),
        current_message=_FIRST_MESSAGE,
        recent_chat_history=[Message(role="user", content=_FIRST_MESSAGE)],
        existing_db_state=_STATE,
        message_source="whatsapp",
    )

    assert result.draft_reply == (
        "Hi Fahim, welcome to Tattoo Hysteria! A colourful watercolor palm "
        "piece sounds lovely! Would around 6–10 cm suit you?"
    )


def test_existing_greeting_is_replaced_not_duplicated() -> None:
    """A model greeting becomes the welcome instead of a second hello."""
    result = _router(
        "Hi Fahim! A colourful palm piece sounds lovely. "
        "Would around 6–10 cm suit you?"
    ).route(
        _palm_draft(),
        current_message=_FIRST_MESSAGE,
        recent_chat_history=[],
        existing_db_state=_STATE,
        message_source="whatsapp",
    )

    assert result.draft_reply.startswith(
        "Hi Fahim, welcome to Tattoo Hysteria! A colourful palm piece"
    )
    assert result.draft_reply.count("Fahim") == 1


def test_model_welcome_is_kept_as_is() -> None:
    """A draft that already welcomes the client is not prefixed again."""
    draft = (
        "Hi Fahim, welcome to Tattoo Hysteria! A colourful palm piece "
        "sounds lovely. Would around 6–10 cm suit you?"
    )
    result = _router(draft).route(
        _palm_draft(),
        current_message=_FIRST_MESSAGE,
        recent_chat_history=[],
        existing_db_state=_STATE,
        message_source="whatsapp",
    )

    assert result.draft_reply == draft


def test_later_whatsapp_replies_have_no_welcome() -> None:
    """Only the first bot message carries the welcome."""
    history = [
        Message(role="user", content=_FIRST_MESSAGE),
        Message(role="assistant", content="Would around 6–10 cm suit you?"),
    ]
    result = _router(
        "Great, 10 cm noted. Do you have a preferred artist?"
    ).route(
        _palm_draft(),
        current_message="yes i also think its about 10cm long.",
        recent_chat_history=history,
        existing_db_state=_STATE,
        message_source="whatsapp",
    )

    assert "welcome" not in result.draft_reply.casefold()


def test_outlook_replies_have_no_whatsapp_welcome() -> None:
    """The WhatsApp welcome never enters email drafts."""
    result = _router(
        "Dear Fahim,\n\nA colourful palm piece sounds lovely. Would around "
        "6–10 cm suit you?\n\nKind regards,\nTattoo Hysteria"
    ).route(
        _palm_draft(),
        current_message=_FIRST_MESSAGE,
        recent_chat_history=[],
        existing_db_state=_STATE,
        message_source="outlook",
    )

    assert result.draft_reply.startswith("Dear Fahim,")


def test_overlong_whatsapp_draft_uses_short_fallback() -> None:
    """Email-length model prose is rejected for WhatsApp."""
    long_draft = (
        "A flowing abstract watercolor tattoo in purple, magenta, and blue, "
        "kept to your palm and inspired by the reference, sounds lovely! "
        "Palm placement needs a closer studio review, particularly for how "
        "well the colors will hold over time on such a high-friction area. "
        "For the palm-only design, would approximately 6–10 cm long suit "
        "you, or would you prefer a different size?"
    )
    assert len(long_draft) > 350

    result = _router(long_draft).route(
        _palm_draft(),
        current_message=_FIRST_MESSAGE,
        recent_chat_history=[Message(role="assistant", content="Hello!")],
        existing_db_state=_STATE,
        message_source="whatsapp",
    )

    assert result.draft_reply != long_draft
    assert "purple, magenta, and blue" not in result.draft_reply


def test_whatsapp_draft_rejects_email_formatting() -> None:
    """WhatsApp drafts cannot use an email salutation or sign-off."""
    router = _router("unused")

    with pytest.raises(ValueError, match="email formatting"):
        router._validate_draft_reply(
            draft_reply="Dear Fahim, what size would you like?",
            extracted=_palm_draft(),
            message_source="whatsapp",
            current_message="hello",
        )


@pytest.mark.parametrize(
    "message",
    [
        "So could you please let me know about the artists descriptively?",
        "could you please describe about the artists",
        "Can you tell me about your artists?",
        "Who are the artists?",
    ],
)
def test_artist_roster_requests_get_the_longer_limit(message: str) -> None:
    """Artist-directory answers may exceed the normal WhatsApp limit."""
    router = _router("unused")

    assert router._is_artist_roster_question(message) is True
