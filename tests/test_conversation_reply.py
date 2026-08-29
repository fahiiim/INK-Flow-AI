"""Regression tests for natural multi-turn client replies."""

from __future__ import annotations

from typing import cast

from langchain_openai import ChatOpenAI

from ai_brain.extraction import TattooTextExtractor
from ai_brain.reply import ConversationReplyComposer
from ai_brain.routing import TattooRouter
from ai_brain.schemas import Message, TattooExtractionDraft


class FailingLLM:
    """Force deterministic extraction and routing fallbacks."""

    def invoke(self, messages: object) -> object:
        """Simulate a provider failure."""
        raise RuntimeError("Simulated provider failure")


def _incomplete_draft() -> TattooExtractionDraft:
    """Build a draft with the full intake checklist missing."""
    return TattooExtractionDraft(
        tattoo_idea="",
        style_tags=["unknown"],
        placement="",
        size_estimate_cm="",
        color_preference="",
        missing_information=[
            "size in cm",
            "placement",
            "reference images",
            "color preference",
            "preferred date",
        ],
    )


def test_greeting_receives_a_human_opening_not_full_questionnaire() -> None:
    """A greeting should start a conversation rather than demand all fields."""
    reply = ConversationReplyComposer().compose(
        extracted=_incomplete_draft(),
        current_message="hello team inkflow",
        recent_chat_history=[],
        risk_level="high",
    )

    assert reply.startswith(("Hey!", "Hi!"))
    assert "size in centimeters" not in reply
    assert reply.count("?") == 1
    assert len(reply) < 100


def test_date_correction_is_remembered_and_not_requested_again() -> None:
    """Today/time details in recent history remove preferred-date missing state."""
    llm = cast(ChatOpenAI, FailingLLM())
    extractor = TattooTextExtractor(llm=llm)
    history = [
        Message(role="user", content="I want the tattoo on my back."),
        Message(
            role="user",
            content="My preferred date is today at 2 PM.",
        ),
        Message(
            role="assistant",
            content="What date would you prefer?",
        ),
    ]
    current_message = "I said that my preferred date is todaye."

    extracted = extractor.extract(
        current_message=current_message,
        style_tags=["unknown"],
        new_image_urls=[],
        existing_db_state={"placement": "back"},
        recent_chat_history=history,
    )
    result = TattooRouter(llm=llm).route(
        extracted=extracted,
        current_message=current_message,
        recent_chat_history=history,
    )

    assert "preferred date" not in extracted.missing_information
    assert "date" not in result.draft_reply.casefold()
    assert result.draft_reply.startswith("Got it, a tattoo on your back.")
    assert "Does that sound right" in result.draft_reply
    assert "reference image" in result.draft_reply
    assert "- Placement:" not in result.draft_reply
    assert len(result.draft_reply) < 300


def test_normal_date_message_gets_a_simple_acknowledgement() -> None:
    """A newly supplied date is acknowledged without correction language."""
    extracted = TattooExtractionDraft(
        tattoo_idea="Back tattoo",
        style_tags=["unknown"],
        placement="back",
        size_estimate_cm="",
        color_preference="",
        missing_information=[
            "size in cm",
            "reference images",
            "color preference",
        ],
    )
    history = [
        Message(
            role="assistant",
            content="What rough size in cm are you thinking?",
        ),
    ]

    reply = ConversationReplyComposer().compose(
        extracted=extracted,
        current_message="My preferred date is today at 2 PM.",
        recent_chat_history=history,
        risk_level="high",
    )

    assert reply.startswith("Got it - I've noted the timing.")
    assert "You're right" not in reply
    assert "reference image" in reply
    assert "date" not in reply.casefold()


def test_incomplete_request_asks_for_reference_image_first() -> None:
    """A reference image is requested before other unknown visual details."""
    extracted = TattooExtractionDraft(
        tattoo_idea="Back tattoo",
        style_tags=["unknown"],
        placement="back",
        size_estimate_cm="",
        color_preference="",
        missing_information=[
            "size in cm",
            "reference images",
            "color preference",
            "preferred date",
        ],
    )
    reply = ConversationReplyComposer().compose(
        extracted=extracted,
        current_message="I want a tattoo on my back.",
        recent_chat_history=[],
        risk_level="high",
    )

    assert reply.count("?") == 1
    assert "reference image" in reply
    assert "rough size" not in reply
    assert "Where on the body" not in reply
    assert "preferred date" not in reply.casefold()


def test_outlook_email_requests_every_missing_item_at_once() -> None:
    """Outlook receives one professional email with the full missing list."""
    extracted = TattooExtractionDraft(
        tattoo_idea="Floral tattoo",
        style_tags=["unknown"],
        placement="",
        size_estimate_cm="",
        color_preference="",
        missing_information=[
            "size in cm",
            "placement",
            "reference images",
            "tattoo style",
            "color preference",
            "preferred date",
        ],
    )

    reply = ConversationReplyComposer().compose_outlook_email(
        extracted,
        existing_db_state={
            "lead": {
                "name": "Maruf Hossain",
                "source": "outlook",
            }
        },
    )

    assert reply.startswith("Dear Maruf,\n\n")
    assert "Subject:" not in reply
    assert "please reply to this email with all of the following" in reply
    assert "- Approximate tattoo size in centimeters" in reply
    assert "- Intended body placement" in reply
    assert "- Reference images" in reply
    assert "- Preferred tattoo style" in reply
    assert "- Color preference" in reply
    assert "- Preferred appointment date" in reply
    assert "Thank you for contacting Tattoo Hysteria." in reply
    assert reply.endswith("Kind regards,\nTattoo Hysteria")


def test_outlook_complete_inquiry_confirms_review_without_questions() -> None:
    """A complete Outlook inquiry becomes a polished receipt email."""
    extracted = TattooExtractionDraft(
        tattoo_idea="Fine-line lotus",
        style_tags=["fine-line"],
        placement="inner wrist",
        size_estimate_cm="5cm",
        color_preference="black-and-grey",
        missing_information=[],
    )

    reply = ConversationReplyComposer().compose_outlook_email(extracted)

    assert reply.startswith("Hello,\n\n")
    assert "Subject:" not in reply
    assert "- Tattoo concept: Fine-line lotus" in reply
    assert "- Style: fine-line" in reply
    assert "- Placement: inner wrist" in reply
    assert "all of the following information" not in reply
    assert "contact you with the next steps" in reply
