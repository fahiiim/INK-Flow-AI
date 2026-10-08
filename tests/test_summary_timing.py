"""Collected details are recapped only when one requirement is left."""

from __future__ import annotations

import json
from types import SimpleNamespace
from typing import cast
from unittest.mock import Mock

from langchain_openai import ChatOpenAI

from ai_brain.reply import ConversationReplyComposer
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


def _locket(missing: list[str]) -> TattooExtractionDraft:
    return TattooExtractionDraft(
        client_name="Fahim Sarker",
        tattoo_idea="Locket with cat portrait",
        style_tags=["fine-line"],
        placement="chest",
        size_estimate_cm="12-16 cm",
        color_preference="black-and-grey",
        preferred_artist="Nina",
        missing_information=missing,
    )


_HISTORY = [
    Message(role="user", content="I want a locket tattoo on my chest."),
    Message(role="assistant", content="Lovely idea! What size would you like?"),
]
_STATE = {"lead": {"name": "Fahim Sarker", "source": "outlook"}}
_RECAP = (
    "Thanks! So that's a black and grey fine-line locket with cat portrait "
    "on your chest at 12-16 cm with Nina. "
    "Would you prefer an online consultation or a studio visit?"
)


def test_llm_recap_is_rejected_while_several_items_are_missing() -> None:
    """A draft restating earlier details falls back when 2+ items remain."""
    result = _router(_RECAP).route(
        _locket(["appointment type", "preferred date"]),
        current_message="Nina sounds great.",
        recent_chat_history=_HISTORY,
        existing_db_state=_STATE,
        message_source="whatsapp",
    )

    assert result.draft_reply != _RECAP
    assert "cat portrait" not in result.draft_reply


def test_llm_recap_is_allowed_when_one_item_is_left() -> None:
    """The final-item turn may summarise every collected detail."""
    result = _router(_RECAP).route(
        _locket(["appointment type"]),
        current_message="Nina sounds great.",
        recent_chat_history=_HISTORY,
        existing_db_state=_STATE,
        message_source="whatsapp",
    )

    assert "cat portrait" in result.draft_reply


def test_llm_may_echo_details_from_the_current_message() -> None:
    """Acknowledging what the client just said is not a recap."""
    reply = (
        "A black and grey fine-line locket with cat portrait on your chest "
        "sounds lovely! What size would you like?"
    )
    result = _router(reply).route(
        _locket(["size in cm", "appointment type"]),
        current_message=(
            "I want a black and grey fine-line locket with cat portrait on my "
            "chest."
        ),
        recent_chat_history=[],
        existing_db_state=_STATE,
        message_source="whatsapp",
    )

    assert "cat portrait" in result.draft_reply


def test_outlook_template_skips_recap_until_last_item() -> None:
    """Outlook follow-ups summarise only on the final-item turn."""
    composer = ConversationReplyComposer()
    several_missing = composer.compose_outlook_email(
        _locket(["appointment type", "preferred date"]),
        existing_db_state=_STATE,
        current_message="Nina sounds great.",
        recent_chat_history=_HISTORY,
    )
    one_missing = composer.compose_outlook_email(
        _locket(["appointment type"]),
        existing_db_state=_STATE,
        current_message="Nina sounds great.",
        recent_chat_history=_HISTORY,
    )

    assert "cat portrait" not in several_missing.casefold()
    assert "cat portrait" in one_missing.casefold()


def test_whatsapp_template_skips_recap_until_last_item() -> None:
    """WhatsApp follow-ups summarise only on the final-item turn."""
    composer = ConversationReplyComposer()
    several_missing = composer.compose_validation(
        _locket(["appointment type", "preferred date"]),
        current_message="Nina sounds great.",
        recent_chat_history=_HISTORY,
        risk_level="low",
    )
    one_missing = composer.compose_validation(
        _locket(["appointment type"]),
        current_message="Nina sounds great.",
        recent_chat_history=_HISTORY,
        risk_level="low",
    )

    assert "Does that sound right" not in several_missing
    assert "cat portrait" not in several_missing.casefold()
    assert "cat portrait" in one_missing.casefold()
