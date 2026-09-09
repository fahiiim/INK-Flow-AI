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
            "client full name",
            "tattoo idea",
            "size in cm",
            "placement",
            "color preference",
            "tattoo style",
            "reference images",
            "preferred artist",
            "service type",
            "preferred dates or availability",
            "tattoo project type",
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

    assert reply == "Hi! What is your full name?"
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
    assert "What is your full name?" in result.draft_reply
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
    assert "Would you like colour or black and grey?" in reply
    assert "date" not in reply.casefold()


def test_incomplete_request_follows_the_required_question_order() -> None:
    """WhatsApp asks the earliest outstanding intake questions first."""
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

    assert reply.count("?") == 2
    assert "What size would you prefer in centimetres?" in reply
    assert "Would you like colour or black and grey?" in reply
    assert "reference image" not in reply
    assert "preferred date" not in reply.casefold()


def test_whatsapp_preferred_artist_question_lists_five_artists() -> None:
    """The controlled WhatsApp artist question exposes only studio artists."""
    extracted = TattooExtractionDraft(
        tattoo_idea="Fine-line flower",
        style_tags=["fine-line", "floral"],
        placement="wrist",
        size_estimate_cm="5cm",
        color_preference="black-and-grey",
        missing_information=["preferred artist"],
    )

    reply = ConversationReplyComposer().compose(
        extracted=extracted,
        current_message="Those details are correct.",
        recent_chat_history=[],
        risk_level="low",
    )

    assert reply.endswith(
        "Do you have a preferred artist? Please choose Hoss, Nina, Lana, "
        "Sandra, Silva, or say no preference."
    )


def test_whatsapp_service_question_lists_every_service_code() -> None:
    """The visit-or-online question includes every configured service option."""
    extracted = TattooExtractionDraft(
        tattoo_idea="Fine-line flower",
        style_tags=["fine-line", "floral"],
        placement="wrist",
        size_estimate_cm="5cm",
        color_preference="black-and-grey",
        missing_information=["service type"],
    )

    reply = ConversationReplyComposer().compose(
        extracted=extracted,
        current_message="I have selected the other details.",
        recent_chat_history=[],
        risk_level="low",
    )

    assert "Can you visit the studio, or do you need an online appointment?" in reply
    for code in ("CH", "CN", "OCH", "OCN", "RH", "RN", "ORH", "ORN", "TH", "TN"):
        assert f"\n{code} - " in reply


def test_outlook_email_requests_every_missing_item_at_once() -> None:
    """Outlook receives one professional email with the full missing list."""
    extracted = TattooExtractionDraft(
        tattoo_idea="",
        style_tags=["unknown"],
        placement="",
        size_estimate_cm="",
        color_preference="",
        missing_information=[
            "client full name",
            "tattoo idea",
            "size in cm",
            "placement",
            "color preference",
            "tattoo style",
            "reference images",
            "preferred artist",
            "service type",
            "preferred dates or availability",
            "tattoo project type",
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
    assert "- What is your full name?" in reply
    assert "- What is your tattoo idea or background story?" in reply
    assert "- What size would you prefer in centimetres?" in reply
    assert "- Where on your body would you like the tattoo?" in reply
    assert "- Would you like colour or black and grey?" in reply
    assert "- What tattoo style would you prefer?" in reply
    assert "- Could you share any reference or inspiration images?" in reply
    assert "Please choose Hoss, Nina, Lana, Sandra, Silva" in reply
    assert "- Can you visit the studio" in reply
    assert "  - CH - In-person consultation with Hoss" in reply
    assert "  - TN - Tattoo session with Nina" in reply
    assert "- What are your preferred dates or general availability?" in reply
    assert "new tattoo, cover-up, continuation, or touch-up" in reply
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
        date="2026-09-04",
        time="14:30",
        client_name="Maruf Hossain",
        preferred_artist="Silva",
        service_code="CH",
        availability="Weekends",
        tattoo_project_type="new tattoo",
        missing_information=[],
    )

    reply = ConversationReplyComposer().compose_outlook_email(extracted)

    assert reply.startswith("Dear Maruf,\n\n")
    assert "Subject:" not in reply
    assert "- Tattoo concept: Fine-line lotus" in reply
    assert "- Style: fine-line" in reply
    assert "- Placement: inner wrist" in reply
    assert "- Preferred date: 2026-09-04" in reply
    assert "- Preferred time: 14:30" in reply
    assert "- Preferred artist: Silva" in reply
    assert "- Selected service: CH - In-person consultation with Hoss" in reply
    assert "- Availability: Weekends" in reply
    assert "- Tattoo project type: new tattoo" in reply
    assert "all of the following information" not in reply
    assert "contact you with the next steps" in reply
