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

    with pytest.raises(ValidationError):
        TattooInquiryInput.model_validate(
            {
                "client_text": "Legacy message",
                "image_urls": [],
            }
        )


@pytest.mark.parametrize(
    ("field_name", "invalid_value"),
    [
        ("suggested_artist", "Lana"),
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