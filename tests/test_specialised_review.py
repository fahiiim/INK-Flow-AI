"""Regression tests for specialised-placement and status-review policy."""

from __future__ import annotations

from typing import cast

from langchain_openai import ChatOpenAI

from ai_brain.routing import TattooRouter
from ai_brain.schemas import ClientIntent, Message, TattooExtractionDraft


class FailingLLM:
    """Fail if a review-sensitive reply incorrectly reaches the LLM."""

    def invoke(self, messages: object) -> object:
        """Reject non-deterministic reply generation in safety paths."""
        raise AssertionError("Specialised review replies must be deterministic")


def _tongue_draft(
    *,
    client_intent: ClientIntent = "continue_intake",
) -> TattooExtractionDraft:
    """Return an incomplete tongue request suitable for routing tests."""
    return TattooExtractionDraft(
        client_name="Fahim Sarker",
        tattoo_idea="Solid black coverage from a reference image",
        style_tags=["unknown"],
        placement="tongue",
        size_estimate_cm="3 cm",
        color_preference="black-and-grey",
        date="2026-10-15",
        appointment_type="studio_visit",
        artist_preference_mode="recommend",
        pricing_requested=True,
        client_intent=client_intent,
        missing_information=[
            "preferred artist",
            "tattoo project type",
        ],
    )


def test_incomplete_tongue_request_keeps_collecting_details() -> None:
    """Specialised placement no longer stops intake before it is complete."""
    router = TattooRouter(llm=cast(ChatOpenAI, FailingLLM()))

    result = router.route(
        _tongue_draft(),
        current_message="Yes, I also want it on my tongue, how much will it cost",
        recent_chat_history=[Message(role="assistant", content="Hi there!")],
        existing_db_state={"lead": {"name": "Fahim Sarker"}},
        message_source="whatsapp",
    )

    assert result.risk_level == "low"
    assert result.staff_review_required is False
    assert result.intake_status == "collecting_info"
    assert result.telegram_review_required is False
    assert result.review_reasons == []
    assert result.suggested_artist == "Unclear"
    assert "tongue tattoos need studio approval" in result.draft_reply
    assert "?" in result.draft_reply
    assert "flagged your request" not in result.draft_reply


def test_tongue_notice_is_only_given_once() -> None:
    router = TattooRouter(llm=cast(ChatOpenAI, FailingLLM()))
    history = [
        Message(
            role="assistant",
            content=(
                "Just so you know, tongue tattoos need studio approval before "
                "an artist, price, or booking can be confirmed. Do you have a "
                "preferred artist?"
            ),
        ),
    ]

    result = router.route(
        _tongue_draft(),
        current_message="no preference",
        recent_chat_history=history,
        existing_db_state={"lead": {"name": "Fahim Sarker"}},
        message_source="whatsapp",
    )

    assert "need studio approval" not in result.draft_reply


def test_complete_tongue_request_escalates_with_specialised_reason() -> None:
    """Once every detail is collected, staff review the full intake."""
    router = TattooRouter(llm=cast(ChatOpenAI, FailingLLM()))
    complete = _tongue_draft().model_copy(
        update={
            "preferred_artist": "No preference",
            "artist_preference_mode": "no_preference",
            "tattoo_project_type": "new tattoo",
            "missing_information": [],
        }
    )

    result = router.route(
        complete,
        current_message="It's a new tattoo",
        recent_chat_history=[Message(role="assistant", content="Hi there!")],
        existing_db_state={"lead": {"name": "Fahim Sarker"}},
        message_source="whatsapp",
    )

    assert result.risk_level == "high"
    assert result.staff_review_required is True
    assert result.auto_reply is False
    assert result.telegram_review_required is True
    assert "specialised_placement_requires_approval" in result.review_reasons
    assert "Nothing has been approved or booked" in result.draft_reply


def test_status_question_never_invents_a_workflow_update() -> None:
    """Unknown status is escalated without claiming approval or booking."""
    router = TattooRouter(llm=cast(ChatOpenAI, FailingLLM()))

    result = router.route(
        _tongue_draft(client_intent="status_update"),
        current_message="Is there any update?",
        existing_db_state={"lead": {"name": "Fahim Sarker"}},
        message_source="outlook",
    )

    assert result.auto_reply is False
    assert result.staff_review_required is True
    assert "client_status_update_required" in result.review_reasons
    assert "can’t confirm a live update" in result.draft_reply
    assert "Nothing has been approved or booked" in result.draft_reply
    assert "?" not in result.draft_reply
