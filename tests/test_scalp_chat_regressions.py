"""Regressions from the WhatsApp scalp-tattoo conversation."""

from __future__ import annotations

import pytest

from ai_brain.extraction import TattooTextExtractor
from ai_brain.reply import ConversationReplyComposer
from ai_brain.schemas import Message, TattooExtractionDraft
from tests.test_whatsapp_style import _router

_ARTISTS = ("Lana", "Nina", "Hossam", "Sliva", "Sandra", "Mila")


def _extractor() -> TattooTextExtractor:
    return TattooTextExtractor.__new__(TattooTextExtractor)


def _scalp_draft(**overrides: object) -> TattooExtractionDraft:
    values: dict[str, object] = {
        "client_name": "Fahim Sarker",
        "tattoo_idea": "",
        "style_tags": ["blackwork", "geometric", "abstract"],
        "placement": "head",
        "size_estimate_cm": "30 cm",
        "size_status": "exact",
        "color_preference": "black-and-grey",
        "artist_preference_mode": "recommend",
        "missing_information": [
            "preferred artist",
            "appointment type",
            "preferred dates or availability",
            "tattoo project type",
        ],
    }
    values.update(overrides)
    return TattooExtractionDraft.model_validate(values)


@pytest.mark.parametrize(
    "message",
    [
        "I mean, yeah, from front to back and right to left, both maybe 30cm.",
        "It should go back to front, about 30cm.",
        "Thanks, I'll get back to you tomorrow.",
    ],
)
def test_directional_back_is_not_a_placement(message: str) -> None:
    assert _extractor()._extract_placement_from_text(message) == ""


@pytest.mark.parametrize(
    "message",
    [
        "I want a tattoo on my back.",
        "A small piece on the back of my shoulder",
    ],
)
def test_body_back_is_still_a_placement(message: str) -> None:
    assert _extractor()._extract_placement_from_text(message) in {
        "back",
        "shoulder",
    }


def test_size_message_keeps_head_placement_from_history() -> None:
    extractor = _extractor()
    history = [
        Message(
            role="user",
            content="It has to cover my whole head, possibly the whole hair area",
        ),
    ]
    placement = extractor._resolve_context_field(
        llm_value="",
        current_message=(
            "I mean, yeah, from front to back and right to left, both "
            "maybe 30cm around."
        ),
        recent_chat_history=history,
        existing_db_state={},
        state_keys=("placement",),
        value_extractor=extractor._extract_placement_from_text,
        field_terms=(),
    )
    assert placement == "head"


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        ("Black-and-grey, inferred from the reference", "black-and-grey"),
        ("Black and grey (from the reference image)", "black-and-grey"),
        ("Full colour", "color"),
        ("", ""),
    ],
)
def test_colour_notes_are_normalized(raw: str, expected: str) -> None:
    assert _extractor()._normalize_color_value(raw) == expected


def test_wanting_to_make_a_tattoo_is_not_an_idea() -> None:
    extractor = _extractor()
    assert extractor._extract_tattoo_idea_from_text(
        "hello, I want to make a tattoo.\nHow much do you usually charge?"
    ) == ""
    assert extractor._is_missing_tattoo_idea("to make a tattoo") is True


def test_artist_directory_puts_each_artist_on_its_own_line() -> None:
    directory = _router("unused")._artist_directory_summary()
    lines = directory.splitlines()

    assert lines[0] == "Here are our resident artists:"
    for artist in _ARTISTS:
        assert any(line.startswith(f"• {artist}: ") for line in lines)


def test_help_choosing_lists_artists_recommends_and_asks_once() -> None:
    directory = _router("unused")._artist_directory_summary()
    reply = ConversationReplyComposer().compose_validation(
        extracted=_scalp_draft(),
        current_message=(
            "I don't know about the artists. Could you please help me to "
            "choose a suitable artist for that?"
        ),
        recent_chat_history=[
            Message(role="assistant", content="Hi, welcome!"),
        ],
        risk_level="low",
        suggested_artist="Sandra",
        artist_directory=directory,
    )

    assert directory in reply
    assert "Sandra would be the best match" in reply
    assert reply.endswith(
        "Would you like to go ahead with Sandra, or would you prefer "
        "another artist?"
    )
    assert reply.count("?") == 1
    assert "Does that sound right" not in reply
    assert "on your back" not in reply


def test_summary_is_not_repeated_after_it_was_shown() -> None:
    history = [
        Message(
            role="assistant",
            content=(
                "Got it, a 30 cm tattoo on your head. Does that sound right, "
                "or would you like to change anything?"
            ),
        ),
    ]
    reply = ConversationReplyComposer().compose_validation(
        extracted=_scalp_draft(
            missing_information=["preferred artist", "tattoo project type"],
        ),
        current_message="today 5 pm",
        recent_chat_history=history,
        risk_level="low",
        suggested_artist="Sandra",
    )

    assert reply == (
        "Got it, I've noted today at 5 pm. Would you like to go ahead with "
        "Sandra, or would you prefer another artist?"
    )


def test_yes_to_offered_artist_sets_preferred_artist() -> None:
    history = [
        Message(role="user", content="today 5 pm"),
        Message(
            role="assistant",
            content=(
                "Got it, I've noted today at 5 pm. Would you like to go ahead "
                "with Sandra, or would you prefer another artist?"
            ),
        ),
    ]
    extractor = _extractor()

    assert extractor._resolve_preferred_artist(
        current_message="yes",
        recent_chat_history=history,
        existing_db_state={},
    ) == "Sandra"
    assert extractor._resolve_preferred_artist(
        current_message="ok but maybe someone else",
        recent_chat_history=history,
        existing_db_state={},
    ) == ""


def test_yes_to_a_non_artist_question_does_not_pick_an_artist() -> None:
    history = [
        Message(
            role="assistant",
            content=(
                "Based on the portfolio match, I'd recommend Sandra. Would you "
                "prefer an online appointment or a studio visit?"
            ),
        ),
    ]
    assert _extractor()._resolve_preferred_artist(
        current_message="ok",
        recent_chat_history=history,
        existing_db_state={},
    ) == ""


def test_single_line_artist_list_from_model_is_rejected() -> None:
    router = _router("unused")
    single_line = (
        "Our artists are Lana (fine-line), Nina (abstract), Hossam "
        "(watercolor), Sliva (realism), Sandra (blackwork), and Mila "
        "(new-school). Who would you like?"
    )

    with pytest.raises(ValueError, match="own line"):
        router._validate_draft_reply(
            draft_reply=single_line,
            extracted=_scalp_draft(),
            message_source="whatsapp",
            current_message="Who are the artists?",
        )
