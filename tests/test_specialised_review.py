"""Regression tests for specialised-placement and status-review policy."""

from __future__ import annotations

from typing import cast

from langchain_openai import ChatOpenAI

from ai_brain.routing import TattooRouter
from ai_brain.schemas import ClientIntent, TattooExtractionDraft


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


def test_tongue_request_pauses_intake_for_immediate_staff_review() -> None:
    """Specialised placement is reviewed before artist or booking flow."""
    router = TattooRouter(llm=cast(ChatOpenAI, FailingLLM()))

    result = router.route(
        _tongue_draft(),
        current_message="Okay then proceed",
        existing_db_state={"lead": {"name": "Fahim Sarker"}},
        message_source="outlook",
    )

    assert result.risk_level == "low"
    assert result.staff_review_required is True
    assert result.intake_status == "needs_staff_review"
    assert result.auto_reply_allowed is False
    assert result.auto_reply is False
    assert result.model_dump(mode="json", by_alias=True)["Auto-reply"] is False
    assert result.telegram_review_required is True
    assert "specialised_placement_requires_approval" in result.review_reasons
    assert result.suggested_artist == "Unclear"
    assert "flagged your request for review" in result.draft_reply
    assert "Nothing has been approved or booked" in result.draft_reply
    assert "?" not in result.draft_reply


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
