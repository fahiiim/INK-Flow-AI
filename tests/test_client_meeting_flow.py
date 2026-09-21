"""Regression coverage for the conversation demonstrated by the client."""

from __future__ import annotations

from typing import cast

from langchain_openai import ChatOpenAI

from ai_brain.email_cleaning import strip_quoted_email_content
from ai_brain.extraction import TattooTextExtractor
from ai_brain.reply import ConversationReplyComposer
from ai_brain.routing import TattooRouter
from ai_brain.schemas import Message


class FailingLLM:
    """Force deterministic local extraction and routing in these tests."""

    def invoke(self, messages: object) -> object:
        """Simulate an unavailable external model provider."""
        raise RuntimeError("External LLM calls are disabled in tests.")


def _extractor() -> TattooTextExtractor:
    return TattooTextExtractor(llm=cast(ChatOpenAI, FailingLLM()))


def test_group_watercolor_request_keeps_separate_person_details() -> None:
    """Two clients' colours remain separate and natural sizing is accepted."""
    first_message = (
        "Hi, I would like to have a tattoo for me and my girlfriend. "
        "It's in watercolor style. Is it possible?"
    )
    extractor = _extractor()
    first = extractor.extract(
        current_message=first_message,
        style_tags=["unknown"],
        existing_db_state={"lead": {"name": "Kosa Martin"}},
    )

    assert first.party_size == 2
    assert [project.person_label for project in first.projects] == [
        "Client",
        "Girlfriend",
    ]
    assert first.style_tags == ["watercolor"]
    assert first.color_preference == "color"
    assert first.placement == ""
    assert "color preference" not in first.missing_information

    first_reply = ConversationReplyComposer().compose_outlook_email(
        first,
        existing_db_state={"lead": {"name": "Kosa Martin"}},
        current_message=first_message,
    )
    assert "recorded the following" not in first_reply.casefold()
    assert "still need" not in first_reply.casefold()
    assert "Yes, we can help with that" in first_reply
    assert "colour watercolor" not in first_reply
    assert "2 watercolor tattoos" in first_reply
    assert first_reply.count("\n- ") <= 2

    second_message = (
        "I want a tulip. Mine in red, my girlfriend's in blue, both "
        "hand-sized. I don't know which artist; please recommend the best "
        "fit. How much will it cost?"
    )
    history = [
        Message(role="user", content=first_message),
        Message(role="assistant", content=first_reply),
    ]
    second = extractor.extract(
        current_message=second_message,
        style_tags=first.style_tags,
        existing_db_state={
            "lead": {"name": "Kosa Martin"},
            "intake": first.model_dump(),
        },
        recent_chat_history=history,
    )

    assert second.tattoo_idea == "Tulip"
    assert second.party_size == 2
    assert second.placement == ""
    assert second.size_estimate_cm == ""
    assert second.size_description == "hand-sized"
    assert second.size_status == "approximate"
    assert second.preferred_artist == "No preference"
    assert second.artist_preference_mode == "recommend"
    assert second.pricing_requested is True
    assert "size in cm" not in second.missing_information
    assert "placement" in second.missing_information
    assert "preferred artist" not in second.missing_information
    assert [project.color_preference for project in second.projects] == [
        "red",
        "blue",
    ]

    routed = TattooRouter(
        llm=cast(ChatOpenAI, FailingLLM()),
    ).route(
        extracted=second,
        current_message=second_message,
        recent_chat_history=history,
        existing_db_state={"lead": {"name": "Kosa Martin"}},
        message_source="outlook",
    )

    assert routed.risk_level == "low"
    assert routed.suggested_artist == "Hoss"
    assert routed.staff_review_required is True
    assert routed.telegram_review_required is True
    assert routed.intake_status == "needs_staff_review"
    assert "multiple_tattoo_projects" in routed.review_reasons
    assert "strongest match" in routed.draft_reply
    assert "confirm the price" in routed.draft_reply
    assert "recorded the following" not in routed.draft_reply.casefold()

    whatsapp = TattooRouter(
        llm=cast(ChatOpenAI, FailingLLM()),
    ).route(
        extracted=first,
        current_message=first_message,
        existing_db_state={"lead": {"name": "Kosa Martin"}},
        message_source="whatsapp",
    )
    assert whatsapp.draft_reply.startswith("Yes, we can help with that.")
    assert "2 watercolor tattoos" in whatsapp.draft_reply


def test_not_sure_is_a_valid_size_answer_and_triggers_staff_help() -> None:
    """An uncertain client answer is recorded instead of asked repeatedly."""
    history = [
        Message(
            role="assistant",
            content="What size would you prefer in centimetres?",
        )
    ]
    extracted = _extractor().extract(
        current_message="I'm not sure.",
        style_tags=["watercolor"],
        existing_db_state={
            "lead": {"name": "Kosa Martin"},
            "intake": {
                "tattoo_idea": "Tulip",
                "placement": "forearm",
                "color_preference": "color",
                "preferred_artist": "No preference",
                "artist_preference_mode": "recommend",
            },
        },
        recent_chat_history=history,
    )

    assert extracted.size_description == "not sure"
    assert "size in cm" not in extracted.missing_information

    routed = TattooRouter(
        llm=cast(ChatOpenAI, FailingLLM()),
    ).route(
        extracted=extracted,
        current_message="I'm not sure.",
        recent_chat_history=history,
    )
    assert "client_unsure_about_size" in routed.review_reasons
    assert "What size would you prefer" not in routed.draft_reply


def test_email_signature_is_not_treated_as_tattoo_content() -> None:
    """A normal sign-off never leaks into extraction or staff summaries."""
    message = "I want a tulip.\n\nKind regards,\nKosa"

    assert strip_quoted_email_content(message) == "I want a tulip."
    extracted = _extractor().extract(
        current_message=message,
        style_tags=["watercolor"],
        existing_db_state={"lead": {"name": "Kosa Martin"}},
    )
    assert extracted.tattoo_idea == "Tulip"
    assert "Kind regards" not in extracted.tattoo_idea
