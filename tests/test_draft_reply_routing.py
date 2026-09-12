"""Tests for strict Quick M1 draft-reply generation in routing."""

from __future__ import annotations

import json
from types import SimpleNamespace
from typing import cast
from unittest.mock import Mock

from langchain_core.messages import BaseMessage
from langchain_openai import ChatOpenAI

from ai_brain.routing import TattooRouter
from ai_brain.schemas import Message, TattooExtractionDraft
from ai_brain.vector_store import VectorStoreManager


class SequentialLLM:
    """Return deterministic responses for reasoning and draft calls."""

    def __init__(self, responses: list[str]) -> None:
        self._responses = iter(responses)
        self.calls: list[list[BaseMessage]] = []

    def invoke(self, messages: list[BaseMessage]) -> SimpleNamespace:
        """Capture messages and return the next configured JSON response."""
        self.calls.append(messages)
        return SimpleNamespace(content=next(self._responses))


def _warm_vector_store() -> VectorStoreManager:
    """Return a ten-record vector-store mock with no search matches."""
    vector_store = Mock(spec=VectorStoreManager)
    vector_store.records = tuple(range(10))
    vector_store.search_similar_cases.return_value = []
    return cast(VectorStoreManager, vector_store)


def _availability_missing_draft() -> TattooExtractionDraft:
    """Return extracted details with only availability outstanding."""
    return TattooExtractionDraft(
        tattoo_idea="Fine-line lotus",
        style_tags=["fine-line"],
        placement="inner wrist",
        size_estimate_cm="5cm",
        color_preference="black-and-grey",
        missing_information=["preferred dates or availability"],
    )


def _fully_complete_draft() -> TattooExtractionDraft:
    """Return a complete intake ready for staff review."""
    return TattooExtractionDraft(
        client_name="Samim Osman",
        tattoo_idea="Fine-line lotus",
        style_tags=["fine-line"],
        placement="inner wrist",
        size_estimate_cm="5cm",
        color_preference="black-and-grey",
        preferred_artist="Nina",
        appointment_type="studio_visit",
        availability="Weekends",
        tattoo_project_type="new tattoo",
        missing_information=[],
    )


def _reasoning_response() -> str:
    """Return valid internal routing JSON."""
    return json.dumps(
        {
            "confidence_level": "high",
            "ai_reasoning": "Fine-line work matches Nina.",
        }
    )


def test_router_uses_exact_question_for_incomplete_whatsapp_intake() -> None:
    """Incomplete WhatsApp replies use the controlled intake wording."""
    expected_reply = (
        "Got it, a 5cm black-and-grey fine-line tattoo on your inner wrist. "
        "Does that sound right, or would you like to change anything? "
        "What are your preferred dates or general availability?"
    )
    fake_llm = SequentialLLM([_reasoning_response()])

    result = TattooRouter(
        llm=cast(ChatOpenAI, fake_llm),
        vector_store=_warm_vector_store(),
    ).route(
        extracted=_availability_missing_draft(),
        current_message="I want a 5cm fine-line lotus on my wrist.",
        recent_chat_history=[],
        existing_db_state={"lead_name": "Samim"},
    )

    assert result.draft_reply == expected_reply
    assert len(fake_llm.calls) == 1


def test_confirmed_history_continues_with_exact_next_question() -> None:
    """Confirmed details continue directly to the next required question."""
    fake_llm = SequentialLLM([_reasoning_response()])
    history = [
        Message(role="user", content="Yes, those details are correct."),
    ]

    result = TattooRouter(
        llm=cast(ChatOpenAI, fake_llm),
        vector_store=_warm_vector_store(),
    ).route(
        extracted=_availability_missing_draft(),
        current_message="Yes, those details are correct.",
        recent_chat_history=history,
        existing_db_state={"preferred_date": "not provided"},
    )

    assert result.draft_reply == (
        "Got it. What are your preferred dates or general availability?"
    )
    assert "Please confirm" not in result.draft_reply
    assert len(fake_llm.calls) == 1


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
        vector_store=_warm_vector_store(),
    ).route(
        extracted=_fully_complete_draft(),
        current_message="I want this tattoo.",
        recent_chat_history=[],
    )

    assert result.draft_reply != "123"
    assert "studio team review" in result.draft_reply
    assert "- Style:" not in result.draft_reply
    assert "Unknown" not in result.draft_reply
    assert result.draft_reply.count("?") <= 2


def test_outlook_route_uses_email_composer_and_one_routing_llm_call() -> None:
    """Outlook bypasses chat drafting and requests all missing facts."""
    fake_llm = SequentialLLM([_reasoning_response()])
    extracted = TattooExtractionDraft(
        tattoo_idea="Floral tattoo",
        style_tags=["unknown"],
        placement="",
        size_estimate_cm="",
        color_preference="",
        missing_information=[
            "size in cm",
            "placement",
            "color preference",
            "tattoo style",
            "reference images",
            "preferred artist",
            "appointment type",
            "preferred dates or availability",
            "tattoo project type",
        ],
    )

    result = TattooRouter(
        llm=cast(ChatOpenAI, fake_llm),
        vector_store=_warm_vector_store(),
    ).route(
        extracted=extracted,
        current_message="I am interested in a floral tattoo.",
        existing_db_state={"lead": {"name": "Maruf Hossain"}},
        message_source="outlook",
    )

    assert result.draft_reply.startswith("Dear Maruf,\n\n")
    assert "Subject:" not in result.draft_reply
    assert result.draft_reply.count("\n- ") == 10
    assert "Dear Maruf," in result.draft_reply
    assert result.risk_level == "low"
    assert result.auto_reply_allowed is True
    assert len(fake_llm.calls) == 1


def test_first_outlook_price_question_collects_missing_information() -> None:
    """An initial cost question stays low risk until intake is complete."""
    fake_llm = SequentialLLM([_reasoning_response()])
    extracted = TattooExtractionDraft(
        tattoo_idea="",
        style_tags=["unknown"],
        placement="",
        size_estimate_cm="",
        color_preference="",
        missing_information=[
            "tattoo idea",
            "size in cm",
            "placement",
            "color preference",
            "tattoo style",
            "reference images",
            "preferred artist",
            "appointment type",
            "preferred dates or availability",
            "tattoo project type",
        ],
    )

    result = TattooRouter(
        llm=cast(ChatOpenAI, fake_llm),
        vector_store=_warm_vector_store(),
    ).route(
        extracted=extracted,
        current_message="Hi, how much will a tattoo cost?",
        existing_db_state={"lead": {"name": "Maruf Hossain"}},
        message_source="outlook",
    )

    assert result.risk_level == "low"
    assert result.auto_reply_allowed is True
    assert result.telegram_review_required is False
    assert "review the design details before confirming the price" in (
        result.draft_reply
    )
    assert "please reply to this email with all of the following" in (
        result.draft_reply
    )
    assert result.draft_reply.count("\n- ") == 10
