"""Tests for strict Quick M1 draft-reply generation in routing."""

from __future__ import annotations

import json
from types import SimpleNamespace
from typing import cast

from langchain_core.messages import BaseMessage, HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI

from ai_brain.prompts import DRAFT_REPLY_SYSTEM_PROMPT
from ai_brain.routing import TattooRouter
from ai_brain.schemas import Message, TattooExtractionDraft


class SequentialLLM:
    """Return deterministic responses for reasoning and draft calls."""

    def __init__(self, responses: list[str]) -> None:
        self._responses = iter(responses)
        self.calls: list[list[BaseMessage]] = []

    def invoke(self, messages: list[BaseMessage]) -> SimpleNamespace:
        """Capture messages and return the next configured JSON response."""
        self.calls.append(messages)
        return SimpleNamespace(content=next(self._responses))


def _complete_draft() -> TattooExtractionDraft:
    """Return extracted details ready for client validation."""
    return TattooExtractionDraft(
        tattoo_idea="Fine-line lotus",
        style_tags=["fine-line"],
        placement="inner wrist",
        size_estimate_cm="5cm",
        color_preference="black-and-grey",
        missing_information=["preferred date"],
    )


def _reasoning_response() -> str:
    """Return valid internal routing JSON."""
    return json.dumps(
        {
            "confidence_level": "high",
            "ai_reasoning": "Fine-line work matches Nina.",
        }
    )


def test_router_accepts_strict_validation_draft_reply() -> None:
    """A valid draft string is returned unchanged to the caller."""
    draft_reply = (
        "Got it, a 5cm black-and-grey fine-line tattoo on your inner wrist. "
        "Does that sound right, or would you like to change anything? "
        "What date or time works best for you?"
    )
    fake_llm = SequentialLLM(
        [
            _reasoning_response(),
            json.dumps({"draft_reply": draft_reply}),
        ]
    )

    result = TattooRouter(
        llm=cast(ChatOpenAI, fake_llm),
    ).route(
        extracted=_complete_draft(),
        current_message="I want a 5cm fine-line lotus on my wrist.",
        recent_chat_history=[],
        existing_db_state={"lead_name": "Samim"},
    )

    assert result.draft_reply == draft_reply
    assert len(fake_llm.calls) == 2
    assert isinstance(fake_llm.calls[1][0], SystemMessage)
    assert fake_llm.calls[1][0].content == DRAFT_REPLY_SYSTEM_PROMPT


def test_confirmed_history_is_available_to_draft_prompt() -> None:
    """Confirmed details produce a next-step reply without revalidation."""
    confirmed_reply = (
        "Perfect, thanks for confirming. What date would work best for you?"
    )
    fake_llm = SequentialLLM(
        [
            _reasoning_response(),
            json.dumps({"draft_reply": confirmed_reply}),
        ]
    )
    history = [
        Message(role="user", content="Yes, those details are correct."),
    ]

    result = TattooRouter(
        llm=cast(ChatOpenAI, fake_llm),
    ).route(
        extracted=_complete_draft(),
        current_message="Yes, those details are correct.",
        recent_chat_history=history,
        existing_db_state={"preferred_date": "not provided"},
    )

    assert result.draft_reply == confirmed_reply
    assert "Please confirm" not in result.draft_reply
    human_message = fake_llm.calls[1][1]
    assert isinstance(human_message, HumanMessage)
    assert "Yes, those details are correct." in str(human_message.content)
    assert '"preferred_date": "not provided"' in str(human_message.content)


def test_non_string_draft_reply_uses_validated_fallback() -> None:
    """Non-string LLM output cannot enter AIExtractionOutput."""
    fake_llm = SequentialLLM(
        [
            _reasoning_response(),
            json.dumps({"draft_reply": 123}),
        ]
    )

    result = TattooRouter(
        llm=cast(ChatOpenAI, fake_llm),
    ).route(
        extracted=_complete_draft(),
        current_message="I want this tattoo.",
        recent_chat_history=[],
    )

    assert result.draft_reply != "123"
    assert "Got it, a 5cm black-and-grey fine-line tattoo" in result.draft_reply
    assert "Does that sound right" in result.draft_reply
    assert "- Style:" not in result.draft_reply
    assert "Unknown" not in result.draft_reply
    assert result.draft_reply.count("?") <= 2