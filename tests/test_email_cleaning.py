"""Regression tests for defensive Outlook and Gmail reply cleanup."""

from __future__ import annotations

from types import SimpleNamespace
from typing import cast

from langchain_openai import ChatOpenAI

from ai_brain.email_cleaning import strip_quoted_email_content
from ai_brain.extraction import TattooTextExtractor
from ai_brain.reply import ConversationReplyComposer
from ai_brain.schemas import TattooInquiryInput


class FailingExtractionLLM:
    """Force field-specific deterministic extraction."""

    def invoke(self, messages: object) -> SimpleNamespace:
        """Simulate an unavailable extraction provider."""
        raise RuntimeError("simulated extraction failure")


def test_plain_gmail_quote_is_removed_from_outlook_input() -> None:
    """Only the newest reply reaches AI extraction and routing."""
    raw_message = (
        "3 cm, colorful flower, I prefer Hoss, a studio visit, and a new "
        "tattoo.\n\n"
        "On Wed, Sep 9, 2026 at 3:23 PM Tattoo Hysteria wrote:\n"
        "> Would you like colour or black and grey?\n"
        "> Choose an artist or say no preference.\n"
        "> Is this a new tattoo, cover-up, continuation, or touch-up?"
    )

    inquiry = TattooInquiryInput(
        current_message=raw_message,
        message_source="outlook",
    )

    assert inquiry.current_message == (
        "3 cm, colorful flower, I prefer Hoss, a studio visit, and a new "
        "tattoo."
    )


def test_outlook_headers_html_quotes_and_angle_quotes_are_removed() -> None:
    """Common Outlook, Gmail HTML, and plain quote markers are supported."""
    assert strip_quoted_email_content(
        "I prefer Hoss.\n\nFrom: Tattoo Hysteria\nSent: Wednesday"
    ) == "I prefer Hoss."
    assert strip_quoted_email_content(
        '<div>Studio visit<br>please</div><div class="gmail_quote">old</div>'
    ) == "Studio visit\nplease"
    assert strip_quoted_email_content(
        "A colorful flower.\n> Previous message"
    ) == "A colorful flower."
    assert strip_quoted_email_content(
        "A colorful flower.\n\nOn Wed, Sep 9, 2026 at 3:23 PM\n"
        "Tattoo Hysteria <studio@example.com> wrote:\nOld message"
    ) == "A colorful flower."


def test_first_email_with_only_logistics_keeps_tattoo_idea_missing() -> None:
    """Placement and scheduling details are not promoted into a concept."""
    extractor = TattooTextExtractor(
        llm=cast(ChatOpenAI, FailingExtractionLLM()),
    )

    result = extractor.extract(
        current_message=(
            "Hello Tattoo Hysteria, I'm very excited to get a tattoo on my "
            "hand at your shop this Saturday."
        ),
        style_tags=["unknown"],
        existing_db_state={"lead": {"name": "Fahim Sarker"}},
    )

    assert result.tattoo_idea == ""
    assert "tattoo idea" in result.missing_information


def test_realistic_second_email_extracts_only_client_answers() -> None:
    """Quoted questionnaire choices cannot override explicit client answers."""
    inquiry = TattooInquiryInput(
        current_message=(
            "3 cm, colorful flower, I prefer Hoss and will visit the studio. "
            "This is a new tattoo.\n\n"
            "On Wed, Sep 9, 2026 at 3:23 PM Tattoo Hysteria wrote:\n"
            "> Would you like colour or black and grey?\n"
            "> Choose Hoss, Nina, Lana, Sandra, Silva, or no preference.\n"
            "> Is this a new tattoo, cover-up, continuation, or touch-up?"
        ),
        message_source="outlook",
        new_image_urls=["https://example.com/reference.jpg"],
        existing_db_state={
            "lead": {"name": "Fahim Sarker", "source": "outlook"},
            "intake": {
                "placement": "hand",
                "date": "2026-09-12",
                "availability": "2026-09-12",
            },
        },
    )
    extractor = TattooTextExtractor(
        llm=cast(ChatOpenAI, FailingExtractionLLM()),
    )

    result = extractor.extract(
        current_message=inquiry.current_message,
        style_tags=["floral"],
        visual_color_preference="unknown",
        new_image_urls=inquiry.new_image_urls,
        existing_db_state=inquiry.existing_db_state,
        recent_chat_history=inquiry.recent_chat_history,
    )

    assert result.tattoo_idea == "Colorful flower"
    assert result.placement == "hand"
    assert result.size_estimate_cm == "3 cm"
    assert result.color_preference == "color"
    assert result.date == "2026-09-12"
    assert result.time == ""
    assert result.preferred_artist == "Hoss"
    assert result.appointment_type == "studio_visit"
    assert result.tattoo_project_type == "new tattoo"
    assert result.missing_information == ["tattoo style"]

    reply = ConversationReplyComposer().compose_outlook_email(
        result,
        existing_db_state=inquiry.existing_db_state,
    )
    assert "- Tattoo concept: Colorful flower" in reply
    assert "- Color preference: color" in reply
    assert "- Preferred time:" not in reply
    assert "- Preferred artist: Hoss" in reply
    assert "- Appointment type: Studio visit" in reply
    assert "- Availability:" not in reply
    assert "- What tattoo style would you prefer?" in reply
    assert "On Wed" not in reply


def test_artist_only_followup_preserves_every_other_stored_value() -> None:
    """A narrow correction updates only the field the client mentions."""
    inquiry = TattooInquiryInput(
        current_message=(
            "I'd prefer Hoss as my artist.\n\n"
            "On Wed, Sep 9, 2026 at 3:24 PM Tattoo Hysteria wrote:\n"
            "> Please choose an artist or say no preference."
        ),
        message_source="outlook",
        new_image_urls=["https://example.com/reference.jpg"],
        existing_db_state={
            "lead": {"name": "Fahim Sarker"},
            "intake": {
                "tattoo_idea": "Colorful flower",
                "placement": "hand",
                "size_estimate_cm": "3 cm",
                "color_preference": "color",
                "date": "2026-09-12",
                "availability": "2026-09-12",
                "appointment_type": "studio_visit",
                "tattoo_project_type": "new tattoo",
            },
        },
    )
    extractor = TattooTextExtractor(
        llm=cast(ChatOpenAI, FailingExtractionLLM()),
    )

    result = extractor.extract(
        current_message=inquiry.current_message,
        style_tags=["fine-line"],
        new_image_urls=inquiry.new_image_urls,
        existing_db_state=inquiry.existing_db_state,
    )

    assert result.tattoo_idea == "Colorful flower"
    assert result.color_preference == "color"
    assert result.time == ""
    assert result.preferred_artist == "Hoss"
    assert result.appointment_type == "studio_visit"
    assert result.tattoo_project_type == "new tattoo"
