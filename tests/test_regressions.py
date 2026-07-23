"""Regression tests for routing, LLM, and vision safeguards."""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any, cast

import pytest
from langchain_core.messages import BaseMessage, HumanMessage
from langchain_openai import ChatOpenAI

import ai_brain.llm as llm_module
from ai_brain.config import LLMSettings
from ai_brain.routing import TattooRouter
from ai_brain.schemas import TattooExtractionDraft
from ai_brain.vision import TattooVisionAnalyzer


class FailingLLM:
    """LLM stub that always forces deterministic fallback logic."""

    def invoke(self, messages: object) -> object:
        """Raise a predictable provider failure."""
        raise RuntimeError("Simulated LLM failure")


class CapturingVisionLLM:
    """LLM stub that records messages sent by the vision analyzer."""

    def __init__(self) -> None:
        self.messages: list[BaseMessage] = []

    def invoke(self, messages: list[BaseMessage]) -> SimpleNamespace:
        """Capture messages and return a valid JSON style response."""
        self.messages = messages
        return SimpleNamespace(content='["fine-line"]')


def _router_with_failing_llm() -> TattooRouter:
    """Build a router whose LLM call always enters fallback logic."""
    llm = cast(ChatOpenAI, FailingLLM())
    return TattooRouter(llm=llm)


def test_router_flags_high_risk_keywords_before_missing_information() -> None:
    """Sensitive intent remains high risk despite standard missing fields."""
    draft = TattooExtractionDraft(
        tattoo_idea="Client requests a price quote and complex design advice",
        style_tags=["unknown"],
        placement="",
        size_estimate_cm="",
        color_preference="",
        missing_information=[
            "size in cm",
            "placement",
            "reference images",
            "color preference",
            "preferred date",
        ],
    )

    result = _router_with_failing_llm().route(
        draft,
        current_message="I need complex design advice and a price quote.",
    )

    assert result.risk_level == "high"
    assert result.suggested_artist == "Unclear"
    assert result.confidence_level == "low"
    assert "studio team review" in result.draft_reply


def test_router_keeps_all_standard_missing_information_low_risk() -> None:
    """The complete standard intake checklist is safe for auto-reply."""
    draft = TattooExtractionDraft(
        tattoo_idea="A tattoo idea without intake details",
        style_tags=["unknown"],
        placement="",
        size_estimate_cm="",
        color_preference="",
        missing_information=[
            "size in cm",
            "placement",
            "reference images",
            "color preference",
            "preferred date",
        ],
    )

    result = _router_with_failing_llm().route(
        draft,
        current_message="I want to discuss a tattoo idea.",
    )

    assert result.risk_level == "low"
    assert "studio team review" not in result.draft_reply
    assert result.draft_reply.count("?") == 2


def test_router_keeps_basic_missing_information_low_risk() -> None:
    """Only missing size and placement remains a low-risk follow-up."""
    draft = TattooExtractionDraft(
        tattoo_idea="Minimal floral tattoo",
        style_tags=["minimal", "floral"],
        placement="",
        size_estimate_cm="",
        color_preference="black-and-grey",
        missing_information=["size in cm", "placement"],
    )

    result = _router_with_failing_llm().route(draft)

    assert result.risk_level == "low"
    assert result.suggested_artist == "Nina"
    assert result.confidence_level == "high"
    assert "black-and-grey minimal and floral tattoo" in result.draft_reply
    assert "Does that sound right" in result.draft_reply
    assert "- Style:" not in result.draft_reply
    assert "Unknown" not in result.draft_reply


def test_chat_model_forces_zero_temperature(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Chat model remains deterministic regardless of environment setting."""
    captured: dict[str, Any] = {}
    settings = LLMSettings(
        api_key="test-key",
        temperature=1.0,
        timeout_seconds=30,
        max_retries=2,
    )

    class FakeChatOpenAI:
        """Capture constructor options without creating an OpenAI client."""

        def __init__(self, **kwargs: Any) -> None:
            captured.update(kwargs)

    llm_module.get_chat_model.cache_clear()
    monkeypatch.setattr(llm_module, "get_settings", lambda: settings)
    monkeypatch.setattr(llm_module, "ChatOpenAI", FakeChatOpenAI)

    llm_module.get_chat_model(model_name="determinism-test")

    assert captured["temperature"] == 0.0
    llm_module.get_chat_model.cache_clear()


def test_vision_user_message_contains_only_images() -> None:
    """Vision user payload avoids redundant text instructions."""
    captured_llm = CapturingVisionLLM()
    analyzer = TattooVisionAnalyzer(
        llm=cast(ChatOpenAI, captured_llm),
    )

    response = analyzer._invoke_vision_model(
        ["data:image/jpeg;base64,dGVzdA=="]
    )

    assert response == '["fine-line"]'
    assert len(captured_llm.messages) == 2

    human_message = captured_llm.messages[1]
    assert isinstance(human_message, HumanMessage)
    assert isinstance(human_message.content, list)
    assert human_message.content == [
        {
            "type": "image_url",
            "image_url": {
                "url": "data:image/jpeg;base64,dGVzdA==",
            },
        }
    ]
