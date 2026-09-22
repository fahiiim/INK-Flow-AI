"""Regression coverage for the conversation demonstrated by the client."""

from __future__ import annotations

from typing import Any, cast

from langchain_openai import ChatOpenAI

from ai_brain.email_cleaning import strip_quoted_email_content
from ai_brain.extraction import TattooTextExtractor
from ai_brain.reply import ConversationReplyComposer
from ai_brain.routing import TattooRouter
from ai_brain.schemas import Message, TattooExtractionDraft


class FailingLLM:
    """Force deterministic local extraction and routing in these tests."""

    def invoke(self, messages: object) -> object:
        """Simulate an unavailable external model provider."""
        raise RuntimeError("External LLM calls are disabled in tests.")


def _extractor() -> TattooTextExtractor:
    return TattooTextExtractor(llm=cast(ChatOpenAI, FailingLLM()))


def test_group_watercolor_request_keeps_separate_person_details() -> None:
    """Two clients' colours remain separate and natural sizing is accepted."""
    first_message = (
        "Hi, I would like to have a tattoo for me and my girlfriend. "
        "It's in watercolor style. Is it possible?"
    )
    extractor = _extractor()
    first = extractor.extract(
        current_message=first_message,
        style_tags=["unknown"],
        existing_db_state={"lead": {"name": "Kosa Martin"}},
    )

    assert first.party_size == 2
    assert first.multi_entity_detected is True
    assert "complex routing required" in first.complexity_notes.casefold()
    assert [project.person_label for project in first.projects] == [
        "Client",
        "Girlfriend",
    ]
    assert first.style_tags == ["watercolor"]
    assert first.color_preference == "color"
    assert first.placement == ""
    assert "color preference" not in first.missing_information

    first_reply = ConversationReplyComposer().compose_outlook_email(
        first,
        existing_db_state={"lead": {"name": "Kosa Martin"}},
        current_message=first_message,
    )
    assert "recorded the following" not in first_reply.casefold()
    assert "still need" not in first_reply.casefold()
    assert "Yes, we can help with that" in first_reply
    assert "colour watercolor" not in first_reply
    assert "2 watercolor tattoos" in first_reply
    assert "\n- " not in first_reply

    second_message = (
        "I want a tulip. Mine in red, my girlfriend's in blue, both "
        "hand-sized. I don't know which artist; please recommend the best "
        "fit. How much will it cost?"
    )
    history = [
        Message(role="user", content=first_message),
        Message(role="assistant", content=first_reply),
    ]
    second = extractor.extract(
        current_message=second_message,
        style_tags=first.style_tags,
        existing_db_state={
            "lead": {"name": "Kosa Martin"},
            "intake": first.model_dump(),
        },
        recent_chat_history=history,
    )

    assert second.tattoo_idea == "Tulip"
    assert second.party_size == 2
    assert second.placement == ""
    assert second.size_estimate_cm == "10-15 cm"
    assert second.size_description == "hand-sized"
    assert second.size_status == "approximate"
    assert second.preferred_artist == "No preference"
    assert second.artist_preference_mode == "recommend"
    assert second.pricing_requested is True
    assert "size in cm" not in second.missing_information
    assert "placement" in second.missing_information
    assert "preferred artist" not in second.missing_information
    assert [project.color_preference for project in second.projects] == [
        "red",
        "blue",
    ]

    routed = TattooRouter(
        llm=cast(ChatOpenAI, FailingLLM()),
    ).route(
        extracted=second,
        current_message=second_message,
        recent_chat_history=history,
        existing_db_state={"lead": {"name": "Kosa Martin"}},
        message_source="outlook",
    )

    assert routed.risk_level == "low"
    assert routed.multi_entity_detected is True
    assert routed.complexity_notes == second.complexity_notes
    assert routed.suggested_artist == "Hoss"
    assert routed.staff_review_required is True
    assert routed.telegram_review_required is True
    assert routed.intake_status == "needs_staff_review"
    assert "multiple_tattoo_projects" in routed.review_reasons
    assert "complex_routing_required" in routed.review_reasons
    assert "strongest match" in routed.draft_reply
    assert "custom estimate" in routed.draft_reply
    assert "recorded the following" not in routed.draft_reply.casefold()
    assert "flagged it for a personal review" in routed.draft_reply

    whatsapp = TattooRouter(
        llm=cast(ChatOpenAI, FailingLLM()),
    ).route(
        extracted=first,
        current_message=first_message,
        existing_db_state={"lead": {"name": "Kosa Martin"}},
        message_source="whatsapp",
    )
    assert whatsapp.draft_reply.startswith("Yes, we can help with that.")
    assert "2 watercolor tattoos" in whatsapp.draft_reply


def test_not_sure_is_a_valid_size_answer_and_triggers_staff_help() -> None:
    """An uncertain client answer is recorded instead of asked repeatedly."""
    history = [
        Message(
            role="assistant",
            content="What size would you prefer in centimetres?",
        )
    ]
    extracted = _extractor().extract(
        current_message="I'm not sure.",
        style_tags=["watercolor"],
        existing_db_state={
            "lead": {"name": "Kosa Martin"},
            "intake": {
                "tattoo_idea": "Tulip",
                "placement": "forearm",
                "size_estimate_cm": "12 cm",
                "color_preference": "color",
                "preferred_artist": "No preference",
                "artist_preference_mode": "recommend",
            },
        },
        recent_chat_history=history,
    )

    assert extracted.size_description == "not sure"
    assert extracted.size_estimate_cm == ""
    assert "size in cm" not in extracted.missing_information

    routed = TattooRouter(
        llm=cast(ChatOpenAI, FailingLLM()),
    ).route(
        extracted=extracted,
        current_message="I'm not sure.",
        recent_chat_history=history,
    )
    assert "client_unsure_about_size" in routed.review_reasons
    assert "What size would you prefer" not in routed.draft_reply


def test_coin_sized_request_gets_an_approximate_range() -> None:
    """A common qualitative size becomes a safe range without re-questioning."""
    extracted = _extractor().extract(
        current_message=(
            "I want a coin-sized red heart tattoo on my wrist."
        ),
        style_tags=["fine-line"],
        existing_db_state={"lead": {"name": "Kosa Martin"}},
    )

    assert extracted.size_description == "coin-sized"
    assert extracted.size_estimate_cm == "2-3 cm"
    assert extracted.size_status == "approximate"
    assert "size in cm" not in extracted.missing_information


def test_email_signature_is_not_treated_as_tattoo_content() -> None:
    """A normal sign-off never leaks into extraction or staff summaries."""
    message = "I want a tulip.\n\nKind regards,\nKosa"

    assert strip_quoted_email_content(message) == "I want a tulip."
    extracted = _extractor().extract(
        current_message=message,
        style_tags=["watercolor"],
        existing_db_state={"lead": {"name": "Kosa Martin"}},
    )
    assert extracted.tattoo_idea == "Tulip"
    assert "Kind regards" not in extracted.tattoo_idea


def test_real_email_thread_preserves_concept_size_color_and_placement() -> None:
    """The reported production thread never forgets previously supplied data."""
    extractor = _extractor()
    router = TattooRouter(llm=cast(ChatOpenAI, FailingLLM()))
    state: dict[str, Any] = {
        "lead": {"name": "Fahim Sarker", "source": "outlook"},
    }
    history: list[Message] = []

    def process(
        message: str,
        image_urls: list[str] | None = None,
    ) -> tuple[TattooExtractionDraft, str]:
        extracted = extractor.extract(
            current_message=message,
            style_tags=["unknown"],
            new_image_urls=image_urls or [],
            existing_db_state=state,
            recent_chat_history=history,
        )
        routed = router.route(
            extracted=extracted,
            current_message=message,
            recent_chat_history=history,
            existing_db_state=state,
            message_source="outlook",
        )
        history.extend(
            [
                Message(role="user", content=message),
                Message(role="assistant", content=routed.draft_reply),
            ]
        )
        state["intake"] = routed.model_dump()
        return extracted, routed.draft_reply

    first, first_reply = process(
        "Hello, are you there? I want to get a tattoo on my hand, sized "
        "(I've attached the reference image here). How much will it cost?",
        ["https://example.com/first-reference.png"],
    )
    assert first.placement == "hand"
    assert "reference images" not in first.missing_information
    assert "custom estimate" in first_reply

    second, second_reply = process(
        "Okay, I've attached the photo of my tattoo. I don't know the exact "
        "size; it would be about my hand size. The background is that I'm "
        "matching this with my girlfriend, and it is skeleton art on my hand.",
        ["https://example.com/second-reference.png"],
    )
    assert second.tattoo_idea == "Skeleton art"
    assert second.placement == "hand"
    assert second.size_estimate_cm == "10-15 cm"
    assert second.party_size == 2
    assert second.multi_entity_detected is True
    assert "tattoo idea" not in second.missing_information
    assert "size in cm" not in second.missing_information
    assert "tattoo idea or background story" not in second_reply.casefold()
    assert "updated design details" in second_reply

    third, third_reply = process("It would be in black and gray.")
    assert third.tattoo_idea == "Skeleton art"
    assert third.placement == "hand"
    assert third.size_estimate_cm == "10-15 cm"
    assert third.color_preference == "black-and-grey"
    assert "tattoo idea or background story" not in third_reply.casefold()

    fourth, fourth_reply = process(
        "I said my background story earlier, and here is the reference image "
        "again.",
        ["https://example.com/third-reference.png"],
    )
    assert fourth.tattoo_idea == "Skeleton art"
    assert fourth.placement == "hand"
    assert fourth.color_preference == "black-and-grey"
    assert "where on your body" not in fourth_reply.casefold()
    assert "tattoo idea or background story" not in fourth_reply.casefold()

    fifth, fifth_reply = process("I already said it is on my hand.")
    assert fifth.tattoo_idea == "Skeleton art"
    assert fifth.placement == "hand"
    assert fifth.size_estimate_cm == "10-15 cm"
    assert fifth.color_preference == "black-and-grey"
    assert "where on your body" not in fifth_reply.casefold()
    assert "tattoo idea or background story" not in fifth_reply.casefold()
    assert "sorry for asking again" in fifth_reply.casefold()


def test_saved_project_reference_images_remain_fulfilled() -> None:
    """A stored multi-person project keeps its uploaded references fulfilled."""
    extracted = _extractor().extract(
        current_message="It should be black and grey.",
        style_tags=["unknown"],
        existing_db_state={
            "intake": {
                "projects": [
                    {
                        "reference_image_urls": [
                            "https://example.com/skeleton-reference.png",
                        ],
                    }
                ]
            }
        },
    )

    assert "reference images" not in extracted.missing_information


def test_matching_existing_tattoo_email_gets_a_specific_reply() -> None:
    """A partner's existing tattoo and hand-fit sizing are understood."""
    extractor = _extractor()
    router = TattooRouter(llm=cast(ChatOpenAI, FailingLLM()))
    state: dict[str, Any] = {
        "lead": {"name": "Fahim Sarker", "source": "outlook"},
    }
    first_message = (
        "Hello, I want a tattoo on my hand. I've attached the reference "
        "image. How much will it cost?"
    )
    first = extractor.extract(
        current_message=first_message,
        style_tags=["unknown"],
        new_image_urls=["https://example.com/reference.png"],
        existing_db_state=state,
    )
    first_routed = router.route(
        extracted=first,
        current_message=first_message,
        existing_db_state=state,
        message_source="outlook",
    )
    history = [
        Message(role="user", content=first_message),
        Message(role="assistant", content=first_routed.draft_reply),
    ]
    state["intake"] = first_routed.model_dump()
    second_message = (
        "I want it in a watercolor style. I don't know the exact size; it "
        "will fit on my hand. My girlfriend has a tattoo on her hand and I "
        "want to match it in a gray black color combination. It will be a "
        "skeleton on my hand."
    )

    second = extractor.extract(
        current_message=second_message,
        style_tags=["unknown"],
        existing_db_state=state,
        recent_chat_history=history,
    )
    routed = router.route(
        extracted=second,
        current_message=second_message,
        recent_chat_history=history,
        existing_db_state=state,
        message_source="outlook",
    )

    assert second.tattoo_idea == "Skeleton"
    assert second.style_tags == ["watercolor"]
    assert second.placement == "hand"
    assert second.size_estimate_cm == "10-15 cm"
    assert second.size_description == "hand-sized"
    assert second.size_status == "approximate"
    assert second.color_preference == "black-and-grey"
    assert second.party_size == 1
    assert second.multi_entity_detected is True
    assert "girlfriend's existing tattoo" in second.complexity_notes
    assert "size in cm" not in second.missing_information
    assert "reference images" not in second.missing_information
    assert routed.risk_level == "low"
    assert routed.review_reasons == ["complex_routing_required"]
    assert routed.auto_reply_allowed is True
    assert routed.telegram_review_required is True
    assert "10-15 cm black-and-grey watercolor Skeleton" in (
        routed.draft_reply
    )
    assert "match an existing tattoo" in routed.draft_reply
    assert "10 to 15 cm sound right?" in routed.draft_reply
    assert "updated design details" in routed.draft_reply
