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
            "appointment type",
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


def test_whatsapp_appointment_type_question_lists_two_options() -> None:
    """The appointment question exposes only the two supported values."""
    extracted = TattooExtractionDraft(
        tattoo_idea="Fine-line flower",
        style_tags=["fine-line", "floral"],
        placement="wrist",
        size_estimate_cm="5cm",
        color_preference="black-and-grey",
        missing_information=["appointment type"],
    )

    reply = ConversationReplyComposer().compose(
        extracted=extracted,
        current_message="I have selected the other details.",
        recent_chat_history=[],
        risk_level="low",
    )

    assert reply.endswith(
        "Would you prefer an online appointment or a studio visit? "
        "Please reply with online or studio visit."
    )
    assert "service code" not in reply.casefold()
    assert "CH -" not in reply


def test_whatsapp_style_question_lists_every_supported_style() -> None:
    """WhatsApp presents the complete controlled style vocabulary."""
    extracted = TattooExtractionDraft(
        tattoo_idea="A flower and moon design",
        style_tags=["unknown"],
        placement="forearm",
        size_estimate_cm="8 cm",
        color_preference="black-and-grey",
        missing_information=["tattoo style"],
    )

    reply = ConversationReplyComposer().compose(
        extracted=extracted,
        current_message="The other details are correct.",
        recent_chat_history=[],
        risk_level="low",
    )

    assert reply.endswith(
        "What tattoo style would you prefer? Please choose one or more from: "
        "fine-line, watercolor, minimal, floral, micro-realism, "
        "black-and-grey, calligraphy, traditional, geometric."
    )


def test_outlook_email_asks_only_the_next_two_missing_items() -> None:
    """Outlook starts a natural conversation with two useful questions."""
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
            "appointment type",
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
    assert "we'd be happy to help" in reply
    assert "recorded the following" not in reply.casefold()
    assert "please reply to this email with all of the following" not in reply
    assert "What is your full name?" in reply
    assert "What is your tattoo idea or background story?" in reply
    assert "What size would you prefer in centimetres?" not in reply
    assert "\n- " not in reply
    assert "- Where on your body would you like the tattoo?" not in reply
    assert "studio_visit" not in reply
    assert "service code" not in reply.casefold()
    assert "CH -" not in reply
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
        appointment_type="studio_visit",
        availability="Weekends",
        tattoo_project_type="new tattoo",
        missing_information=[],
    )

    reply = ConversationReplyComposer().compose_outlook_email(extracted)

    assert reply.startswith("Dear Maruf,\n\n")
    assert "Subject:" not in reply
    assert "5cm black-and-grey fine-line lotus tattoo" in reply
    assert "To confirm what I have so far" in reply
    assert "on your inner wrist" in reply
    assert "recorded the following" not in reply.casefold()
    assert reply.count("\n- ") == 0
    assert "studio_visit" not in reply
    assert "all of the following information" not in reply
    assert "contact you with the next steps" in reply


def test_outlook_hides_internal_enum_and_duplicate_availability() -> None:
    """Client email uses natural labels and shows an exact schedule once."""
    extracted = TattooExtractionDraft(
        client_name="Fahim Sarker",
        tattoo_idea="Colorful flower",
        style_tags=["fine-line"],
        placement="hand",
        size_estimate_cm="3 cm",
        color_preference="color",
        date="2026-09-12",
        time="14:30",
        preferred_artist="Hoss",
        appointment_type="studio_visit",
        availability="2026-09-12 at 14:30",
        tattoo_project_type="new tattoo",
        missing_information=[],
    )

    reply = ConversationReplyComposer().compose_outlook_email(extracted)

    assert "3 cm colour fine-line colorful flower tattoo" in reply
    assert "studio_visit" not in reply
    assert "- Availability:" not in reply
