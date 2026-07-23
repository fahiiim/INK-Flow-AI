"""Tests for the Quick M1 client-validation prompt contract."""

from __future__ import annotations

from ai_brain.prompts import (
    DRAFT_REPLY_SYSTEM_PROMPT,
    build_draft_reply_human_prompt,
)
from ai_brain.schemas import Message


def test_system_prompt_locks_validation_and_confirmation_behavior() -> None:
    """The prompt requires validation once and forbids repeated requests."""
    required_sentence = (
        "Please confirm if these details are correct, or let me know if you "
        "would like to change anything."
    )

    assert required_sentence in DRAFT_REPLY_SYSTEM_PROMPT
    assert "Style, Placement, Size, and Color" in DRAFT_REPLY_SYSTEM_PROMPT
    assert "DO NOT repeat the summary" in DRAFT_REPLY_SYSTEM_PROMPT
    assert "DO NOT ask for validation again" in DRAFT_REPLY_SYSTEM_PROMPT
    assert "premium tattoo studio" in DRAFT_REPLY_SYSTEM_PROMPT


def test_human_prompt_includes_details_and_recent_confirmation() -> None:
    """Draft generation receives extracted facts and confirmation history."""
    prompt = build_draft_reply_human_prompt(
        current_message="Yes, those details are correct.",
        extracted_details={
            "style_tags": ["fine-line"],
            "placement": "inner wrist",
            "size_estimate_cm": "5cm",
            "color_preference": "black-and-grey",
        },
        missing_information=["preferred date"],
        recent_chat_history=[
            Message(
                role="user",
                content="Yes, those details are correct.",
            ),
        ],
        suggested_artist="Nina",
        risk_level="low",
        format_instructions="Return draft_reply as a JSON string field.",
    )

    assert '"placement": "inner wrist"' in prompt
    assert '"size_estimate_cm": "5cm"' in prompt
    assert '"content": "Yes, those details are correct."' in prompt
    assert '"missing_information": [' in prompt
    assert "Return draft_reply as a JSON string field." in prompt