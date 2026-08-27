"""Acceptance tests for pause, delivery, and client-draft safeguards."""

from __future__ import annotations

import json
from types import SimpleNamespace
from typing import cast
from unittest.mock import Mock

from langchain_core.messages import BaseMessage
from langchain_openai import ChatOpenAI

from ai_brain.extraction import TattooTextExtractor
from ai_brain.processor import StudioAIBrain
from ai_brain.routing import TattooRouter
from ai_brain.schemas import TattooExtractionDraft
from ai_brain.vector_store import VectorStoreManager
from ai_brain.vision import TattooVisionAnalyzer


class NaturalToneLLM:
    """Return deterministic mocked routing and client-draft responses."""

    def __init__(self) -> None:
        self.calls: list[list[BaseMessage]] = []
        self._responses = iter(
            [
                json.dumps(
                    {
                        "confidence_level": "high",
                        "ai_reasoning": (
                            "Configured fine-line routing is applicable."
                        ),
                    }
                ),
                json.dumps(
                    {
                        "draft_reply": (
                            "Got it, a 5cm fine-line lotus on your inner "
                            "wrist! Would you like black-and-grey or colour, "
                            "and what date works best for you?"
                        )
                    }
                ),
            ]
        )

    def invoke(
        self,
        messages: list[BaseMessage],
    ) -> SimpleNamespace:
        """Capture one mocked OpenAI call and return its next response."""
        self.calls.append(messages)
        return SimpleNamespace(content=next(self._responses))


def _warm_vector_store() -> VectorStoreManager:
    """Return an injected warm store without embedding-provider calls."""
    vector_store = Mock(spec=VectorStoreManager)
    vector_store.record_count = 10
    vector_store.search_similar_cases.return_value = []
    return cast(VectorStoreManager, vector_store)


def test_automation_pause() -> None:
    """Paused automation exits before any vision, extraction, or LLM work."""
    vision = Mock(spec=TattooVisionAnalyzer)
    extractor = Mock(spec=TattooTextExtractor)
    router = Mock(spec=TattooRouter)
    brain = StudioAIBrain(
        vision_analyzer=vision,
        text_extractor=extractor,
        router=router,
    )

    result = brain.process_inquiry(
        current_message="Please change my existing booking.",
        new_image_urls=["https://example.com/reference.jpg"],
        existing_db_state={"automation_paused": True},
        recent_chat_history=[],
    )

    assert result.risk_level == "high"
    assert result.suggested_artist == "Unclear"
    assert result.draft_reply == "A staff member will reply to you shortly."
    assert result.auto_reply_allowed is False
    assert result.telegram_review_required is True
    vision.analyze_images.assert_not_called()
    extractor.extract.assert_not_called()
    router.route.assert_not_called()


def test_draft_reply_natural_tone() -> None:
    """A mocked draft contains natural known facts and no blank field list."""
    mocked_llm = NaturalToneLLM()
    router = TattooRouter(
        llm=cast(ChatOpenAI, mocked_llm),
        vector_store=_warm_vector_store(),
    )
    extracted = TattooExtractionDraft(
        tattoo_idea="Fine-line lotus",
        style_tags=["fine-line"],
        placement="inner wrist",
        size_estimate_cm="5cm",
        color_preference="",
        missing_information=["color preference", "preferred date"],
    )

    result = router.route(
        extracted=extracted,
        current_message="I want a 5cm fine-line lotus on my inner wrist.",
        recent_chat_history=[],
    )

    normalized_reply = result.draft_reply.casefold()
    assert result.draft_reply.startswith(
        "Got it, a 5cm fine-line lotus on your inner wrist!"
    )
    assert "unknown" not in normalized_reply
    assert "none" not in normalized_reply
    assert "n/a" not in normalized_reply
    assert "\n-" not in result.draft_reply
    assert "style:" not in normalized_reply
    assert "placement:" not in normalized_reply
    assert result.draft_reply.count("?") <= 2
    assert len(mocked_llm.calls) == 2
