"""Regression coverage for natural Outlook conversation behavior."""

from __future__ import annotations

import json
from types import SimpleNamespace
from typing import cast

import pytest
from langchain_openai import ChatOpenAI

from ai_brain.extraction import (
    _PLACEMENT_ALIASES,
    _PLACEMENT_SIZE_RANGES,
    TattooTextExtractor,
)
from ai_brain.routing import TattooRouter
from ai_brain.schemas import Message


class FailingLLM:
    """Force deterministic extraction and reply fallbacks."""

    def invoke(self, messages: object) -> object:
        """Simulate an unavailable provider."""
        raise RuntimeError("Simulated provider failure")


class StaticExtractionLLM:
    """Return one deterministic extraction payload."""

    def __init__(self, payload: dict[str, object]) -> None:
        self._payload = payload

    def invoke(self, messages: object) -> SimpleNamespace:
        """Return the configured extraction as model JSON."""
        return SimpleNamespace(content=json.dumps(self._payload))


def _extractor() -> TattooTextExtractor:
    """Build an extractor without external API calls."""
    return TattooTextExtractor(llm=cast(ChatOpenAI, FailingLLM()))


def _first_assistant_reply() -> Message:
    """Return the size-and-idea question from the reported email flow."""
    return Message(
        role="assistant",
        content=(
            "What is your tattoo idea or background story? What size would "
            "you prefer in centimetres?"
        ),
    )


def _design_message() -> str:
    """Return the client's star design and sizing question."""
    return (
        "I want to have a tattoo on my neck. It is like multiple stars, just "
        "like the attached picture. I have no idea about the size; can you "
        "suggest it to me and also share the estimated cost?"
    )


def test_multiple_stars_and_size_uncertainty_are_separate_fields() -> None:
    """Uncertain sizing cannot replace an explicitly stated tattoo concept."""
    result = _extractor().extract(
        current_message=_design_message(),
        style_tags=["unknown"],
        new_image_urls=["https://example.com/stars.jpg"],
        existing_db_state={"lead": {"name": "Maruf Hossain"}},
        recent_chat_history=[_first_assistant_reply()],
    )

    assert result.tattoo_idea == "Multiple stars"
    assert result.placement == "neck"
    assert result.size_description == "neck-sized"
    assert result.size_estimate_cm == "5-15 cm"
    assert result.client_intent == "size_guidance"
    assert result.pricing_requested is True
    assert "tattoo idea" not in result.missing_information
    assert "size in cm" not in result.missing_information


def test_standalone_black_is_normalized_as_black_and_grey() -> None:
    """A plain black answer must not be converted into full colour."""
    history = [
        _first_assistant_reply(),
        Message(role="user", content=_design_message()),
        Message(
            role="assistant",
            content="Would you like colour or black and grey?",
        ),
    ]
    result = _extractor().extract(
        current_message=(
            "Regarding the color, I will need it to be black, just like the "
            "attached one."
        ),
        style_tags=["unknown"],
        new_image_urls=[],
        existing_db_state={"lead": {"name": "Maruf Hossain"}},
        recent_chat_history=history,
    )

    assert result.color_preference == "black-and-grey"
    assert result.tattoo_idea == "Multiple stars"
    assert result.placement == "neck"


def test_withdrawal_closes_inquiry_and_stops_all_questions() -> None:
    """A withdrawn client receives one closure acknowledgement only."""
    extracted = _extractor().extract(
        current_message="I am no longer interested.",
        style_tags=["unknown"],
        existing_db_state={"lead": {"name": "Maruf Hossain"}},
        recent_chat_history=[
            _first_assistant_reply(),
            Message(role="user", content=_design_message()),
        ],
    )
    result = TattooRouter(
        llm=cast(ChatOpenAI, FailingLLM())
    ).route(
        extracted=extracted,
        current_message="I am no longer interested.",
        existing_db_state={"lead": {"name": "Maruf Hossain"}},
        message_source="outlook",
    )

    assert extracted.client_intent == "withdrawal"
    assert result.conversation_status == "closed"
    assert result.intake_status == "closed"
    assert result.risk_level == "low"
    assert result.auto_reply_allowed is True
    assert result.telegram_review_required is False
    assert "closed your tattoo inquiry" in result.draft_reply
    assert "preferred timing" not in result.draft_reply
    assert "?" not in result.draft_reply


def test_complaint_repeating_withdrawal_remains_closed() -> None:
    """Frustration around an earlier withdrawal cannot restart collection."""
    result = _extractor().extract(
        current_message=(
            "I said I am no longer interested, so why are you asking me "
            "these questions?"
        ),
        style_tags=["unknown"],
        existing_db_state={"conversation_status": "closed"},
    )

    assert result.client_intent == "withdrawal"
    assert result.conversation_status == "closed"


def test_visual_subject_can_recover_an_image_only_concept() -> None:
    """Vision subjects can supply a concrete concept when text is absent."""
    result = _extractor().extract(
        current_message="",
        style_tags=["minimal"],
        visual_color_preference="black-and-grey",
        visual_subjects=["three outline stars"],
        visual_description="Three outlined stars arranged vertically.",
        new_image_urls=["https://example.com/stars.jpg"],
    )

    assert result.tattoo_idea == "Three outline stars"
    assert "tattoo idea" not in result.missing_information


def test_full_chest_reference_request_resolves_concept_and_size() -> None:
    """A named reference and chest-spanning size are useful intake details."""
    result = _extractor().extract(
        current_message=(
            "I wanna copy a tattoo on my chest that will look like Conor "
            "McGregor's one. I don't really know the size. It will cover my "
            "both chest, and the tattoo will be colorful."
        ),
        style_tags=["unknown"],
        new_image_urls=["https://example.com/chest-reference.jpg"],
        existing_db_state={"lead": {"name": "Fahim Sarker"}},
    )

    assert result.tattoo_idea == "Conor McGregor-inspired design"
    assert result.placement == "chest"
    assert result.size_description == "full-chest"
    assert result.size_estimate_cm == "30-40 cm"
    assert result.size_status == "approximate"
    assert result.color_preference == "color"
    assert "tattoo idea" not in result.missing_information
    assert "size in cm" not in result.missing_information
    assert "reference images" not in result.missing_information


def test_reference_image_gets_client_confirmable_size_suggestion() -> None:
    """A reference and placement produce guidance without artist deferral."""
    message = "I want this rose tattoo on my hand. I've attached the image."
    result = _extractor().extract(
        current_message=message,
        style_tags=["fine-line"],
        new_image_urls=["https://example.com/rose.jpg"],
        existing_db_state={"lead": {"name": "Fahim Sarker"}},
    )

    assert result.size_estimate_cm == "10-15 cm"
    assert result.size_description == "hand-sized"
    assert result.size_status == "approximate"
    assert "size in cm" not in result.missing_information
    assert "reference-informed planning estimate" in result.complexity_notes

    routed = TattooRouter(
        llm=cast(ChatOpenAI, FailingLLM()),
    ).route(
        extracted=result,
        current_message=message,
        existing_db_state={"lead": {"name": "Fahim Sarker"}},
        message_source="outlook",
    )

    assert "Based on your reference image" in routed.draft_reply
    assert "around 10 to 15 cm" in routed.draft_reply
    assert "confirm the size" not in routed.draft_reply.casefold()
    assert "recommend suitable dimensions" not in routed.draft_reply.casefold()


def test_visual_design_can_refine_size_within_placement_limits() -> None:
    """A grounded visual recommendation may narrow the placement range."""
    llm = StaticExtractionLLM(
        {
            "client_name": "Fahim Sarker",
            "tattoo_idea": "Detailed rose",
            "placement": "wrist",
            "size_estimate_cm": "6-8 cm",
            "color_preference": "black-and-grey",
        }
    )
    extractor = TattooTextExtractor(llm=cast(ChatOpenAI, llm))
    result = extractor.extract(
        current_message=(
            "I want this detailed rose on my wrist. The reference image is "
            "attached."
        ),
        style_tags=["micro-realism"],
        visual_description="A detailed rose with fine internal shading.",
        new_image_urls=["https://example.com/detailed-rose.jpg"],
        existing_db_state={"lead": {"name": "Fahim Sarker"}},
    )

    assert result.size_estimate_cm == "6-8 cm"
    assert result.size_description == "wrist-sized"
    assert result.size_status == "approximate"


def test_no_reference_image_asks_client_for_size_directly() -> None:
    """Placement alone must not manufacture a size recommendation."""
    message = "I want a fine-line rose tattoo on my hand."
    result = _extractor().extract(
        current_message=message,
        style_tags=["fine-line"],
        existing_db_state={"lead": {"name": "Fahim Sarker"}},
    )

    assert result.size_estimate_cm == ""
    assert result.size_description == ""
    assert "size in cm" in result.missing_information

    routed = TattooRouter(
        llm=cast(ChatOpenAI, FailingLLM()),
    ).route(
        extracted=result,
        current_message=message,
        existing_db_state={"lead": {"name": "Fahim Sarker"}},
        message_source="outlook",
    )

    assert "What size would you prefer in centimetres?" in routed.draft_reply
    assert "artist can recommend" not in routed.draft_reply.casefold()


def test_explicit_eagle_idea_drops_fillers_and_corrects_obvious_typo() -> None:
    """Conversational filler and an obvious feather typo stay out of state."""
    result = _extractor().extract(
        current_message=(
            "The tattoo idea is basically the eagles fins/feathres."
        ),
        style_tags=["traditional"],
        existing_db_state={
            "lead": {"name": "Fahim Sarker"},
            "intake": {
                "tattoo_idea": "Conor McGregor-inspired design",
                "placement": "chest",
                "size_description": "full-chest",
            },
        },
    )

    assert result.tattoo_idea == "Eagle feathers"
    assert result.placement == "chest"
    assert result.size_estimate_cm == "30-40 cm"


def test_every_supported_body_placement_has_a_planning_range() -> None:
    """Every placement alias must resolve to a configured centimetre range."""
    supported_placements = {
        canonical for _, canonical in _PLACEMENT_ALIASES
    }

    assert supported_placements <= set(_PLACEMENT_SIZE_RANGES)


@pytest.mark.parametrize(
    ("message", "placement", "description", "size_range"),
    [
        (
            "I don't know the size; it will be on my scalp.",
            "scalp",
            "scalp-sized",
            "5-20 cm",
        ),
        (
            "I don't know the size; it will be behind the ear.",
            "behind the ear",
            "behind the ear-sized",
            "2-6 cm",
        ),
        (
            "I don't know the size; it will be on my upper back.",
            "upper back",
            "upper back-sized",
            "15-35 cm",
        ),
        (
            "I don't know the size; it will be on my forearm.",
            "forearm",
            "forearm-sized",
            "8-20 cm",
        ),
        (
            "I don't know the size; it will be on my finger.",
            "finger",
            "finger-sized",
            "1-5 cm",
        ),
        (
            "I don't know the size; it will be over my ribs.",
            "rib cage",
            "rib cage-sized",
            "10-30 cm",
        ),
        (
            "I don't know the size; it will be on my thigh.",
            "thigh",
            "thigh-sized",
            "12-30 cm",
        ),
        (
            "I don't know the size; it will be on my foot.",
            "foot",
            "foot-sized",
            "5-15 cm",
        ),
    ],
)
def test_unknown_size_uses_body_placement_planning_range(
    message: str,
    placement: str,
    description: str,
    size_range: str,
) -> None:
    """Unknown sizes receive conservative placement-aware ranges."""
    result = _extractor().extract(
        current_message=message,
        style_tags=["unknown"],
        new_image_urls=["https://example.com/reference.jpg"],
        existing_db_state={"lead": {"name": "Fahim Sarker"}},
    )

    assert result.placement == placement
    assert result.size_description == description
    assert result.size_estimate_cm == size_range
    assert result.size_status == "approximate"
    assert "size in cm" not in result.missing_information


def test_unlisted_body_placement_receives_safe_fallback_range() -> None:
    """A model-resolved placement outside the catalog still gets guidance."""
    result = _extractor().extract(
        current_message="I don't know the exact size.",
        style_tags=["unknown"],
        new_image_urls=["https://example.com/reference.jpg"],
        existing_db_state={
            "lead": {"name": "Fahim Sarker"},
            "intake": {"placement": "Achilles tendon"},
        },
    )

    assert result.placement == "Achilles tendon"
    assert result.size_description == "achilles tendon-sized"
    assert result.size_estimate_cm == "5-20 cm"
    assert result.size_status == "approximate"
    assert "size in cm" not in result.missing_information


def test_three_finger_tattoos_remain_one_client_with_three_designs() -> None:
    """Separate tattoos stay distinct without becoming three people."""
    extractor = _extractor()
    first_message = (
        "I want to get tattoos on my three fingers. I wanna visit to your "
        "studio soon."
    )
    first = extractor.extract(
        current_message=first_message,
        style_tags=["unknown"],
        existing_db_state={"lead": {"name": "Fahim Sarker"}},
    )

    assert first.placement == "three fingers"
    assert first.appointment_type == "studio_visit"
    assert first.party_size == 1

    second_message = (
        "The idea is: on one finger it will be a heart shape, another one "
        "will be a sword, and another will be a diamond shaped tattoo. "
        "Maybe it will be 1.5 cm on each finger."
    )
    second = extractor.extract(
        current_message=second_message,
        style_tags=first.style_tags,
        existing_db_state={
            "lead": {"name": "Fahim Sarker"},
            "intake": first.model_dump(mode="json"),
        },
    )

    assert second.tattoo_idea == (
        "Three finger tattoos: heart, sword, and diamond"
    )
    assert second.placement == "three fingers"
    assert second.size_estimate_cm == "1.5 cm"
    assert second.party_size == 1
    assert second.multi_entity_detected is True
    assert len(second.projects) == 3
    assert {project.person_label for project in second.projects} == {"Client"}
    assert [project.tattoo_idea for project in second.projects] == [
        "Heart",
        "Sword",
        "Diamond",
    ]


def test_artist_roster_question_is_answered_and_projects_are_not_public() -> None:
    """Artist information is answered while the API remains flat."""
    extractor = _extractor()
    stored_intake = {
        "client_name": "Fahim Sarker",
        "tattoo_idea": "Three finger tattoos: heart, sword, and diamond",
        "placement": "three fingers",
        "size_estimate_cm": "1.5 cm",
        "color_preference": "black-and-grey",
        "appointment_type": "studio_visit",
        "party_size": 1,
        "multi_entity_detected": True,
        "complexity_notes": (
            "Complex routing required: 3 tattoo projects for one client"
        ),
    }
    message = (
        "It's a minimal style. Who are the artists in your shop? Could you "
        "please guide me about them?"
    )
    extracted = extractor.extract(
        current_message=message,
        style_tags=["unknown"],
        existing_db_state={
            "lead": {"name": "Fahim Sarker"},
            "intake": stored_intake,
        },
    )
    result = TattooRouter(
        llm=cast(ChatOpenAI, FailingLLM())
    ).route(
        extracted=extracted,
        current_message=message,
        existing_db_state={
            "lead": {"name": "Fahim Sarker"},
            "intake": stored_intake,
        },
        message_source="outlook",
    )

    assert extracted.client_intent == "artist_guidance"
    assert extracted.tattoo_idea == stored_intake["tattoo_idea"]
    assert "minimal like:" not in result.draft_reply.casefold()
    for artist in ("Hoss", "Nina", "Lana", "Sandra", "Silva"):
        assert artist in result.draft_reply
    assert "projects" not in result.model_dump(mode="json")
