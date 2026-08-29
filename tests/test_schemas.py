"""Contract tests for strict AI Brain input and output schemas."""

from __future__ import annotations

from typing import Any

import pytest
from pydantic import ValidationError

from ai_brain.schemas import AIExtractionOutput, TattooInquiryInput


def _valid_output_payload() -> dict[str, Any]:
    """Return a valid final analysis payload for mutation tests."""
    return {
        "tattoo_idea": "Fine-line lotus",
        "style_tags": ["fine-line"],
        "placement": "inner wrist",
        "size_estimate_cm": "5cm",
        "color_preference": "black-and-grey",
        "suggested_artist": "Nina",
        "confidence_level": "high",
        "ai_reasoning": "Fine-line work matches Nina.",
        "missing_information": [],
        "risk_level": "low",
        "draft_reply": "Got it. What date works best for you?",
    }


def test_hybrid_input_accepts_only_canonical_fields() -> None:
    """Canonical hybrid fields validate and legacy aliases are forbidden."""
    inquiry = TattooInquiryInput(
        current_message="Actually make it 10cm.",
        new_image_urls=[" https://example.com/reference.jpg "],
        existing_db_state={"size": "5cm"},
        recent_chat_history=[],
    )

    assert inquiry.current_message == "Actually make it 10cm."
    assert inquiry.new_image_urls == ["https://example.com/reference.jpg"]
    assert inquiry.message_source == "whatsapp"

    outlook_inquiry = TattooInquiryInput(
        current_message="I would like to discuss a tattoo.",
        message_source=" Outlook ",
    )
    assert outlook_inquiry.message_source == "outlook"

    nested_outlook_inquiry = TattooInquiryInput(
        current_message="I would like to discuss a tattoo.",
        existing_db_state={
            "lead": {"source": "outlook"},
            "intake": {"source": "outlook"},
        },
    )
    assert nested_outlook_inquiry.message_source == "outlook"

    explicit_whatsapp_inquiry = TattooInquiryInput(
        current_message="I would like to discuss a tattoo.",
        existing_db_state={"intake": {"source": "outlook"}},
        message_source="whatsapp",
    )
    assert explicit_whatsapp_inquiry.message_source == "whatsapp"

    with pytest.raises(ValidationError):
        TattooInquiryInput.model_validate(
            {
                "client_text": "Legacy message",
                "image_urls": [],
            }
        )

    with pytest.raises(ValidationError):
        TattooInquiryInput(
            current_message="Valid message",
            message_source="instagram",
        )


def test_input_drops_empty_history_and_accepts_image_only_message() -> None:
    """WhatsApp media entries with no text do not cause validation errors."""
    inquiry = TattooInquiryInput(
        current_message="",
        new_image_urls=["https://example.com/whatsapp-image.jpg"],
        recent_chat_history=[
            {"role": "user", "content": ""},
            {"role": "user", "content": "hi"},
        ],
    )

    assert inquiry.current_message == ""
    assert len(inquiry.recent_chat_history) == 1
    assert inquiry.recent_chat_history[0].content == "hi"


def test_input_rejects_request_without_text_or_image() -> None:
    """A request with no usable current content remains invalid."""
    with pytest.raises(ValidationError):
        TattooInquiryInput(
            current_message="",
            new_image_urls=[],
            recent_chat_history=[],
        )


@pytest.mark.parametrize(
    ("field_name", "invalid_value"),
    [
        ("suggested_artist", ""),
        ("confidence_level", "certain"),
        ("risk_level", "medium"),
        ("style_tags", ["neo-traditional"]),
        ("style_tags", []),
    ],
)
def test_output_rejects_values_outside_contract(
    field_name: str,
    invalid_value: object,
) -> None:
    """Final output accepts only declared literals and non-empty style tags."""
    payload = _valid_output_payload()
    payload[field_name] = invalid_value

    with pytest.raises(ValidationError):
        AIExtractionOutput.model_validate(payload)


def test_output_rejects_extra_fields() -> None:
    """Unknown response fields cannot silently enter the backend contract."""
    payload = _valid_output_payload()
    payload["unexpected"] = True

    with pytest.raises(ValidationError):
        AIExtractionOutput.model_validate(payload)


def test_high_risk_output_rejects_auto_reply_delivery() -> None:
    """The output contract cannot mark a high-risk draft as sendable."""
    payload = _valid_output_payload()
    payload.update(
        {
            "risk_level": "high",
            "auto_reply_allowed": True,
            "telegram_review_required": True,
        }
    )

    with pytest.raises(ValidationError, match="cannot allow auto-replies"):
        AIExtractionOutput.model_validate(payload)
