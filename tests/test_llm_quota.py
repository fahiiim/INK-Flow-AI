"""No client reply is produced when the AI provider quota is exhausted."""

from __future__ import annotations

from typing import cast
from unittest.mock import Mock

import httpx
import openai
from langchain_openai import ChatOpenAI

from ai_brain.extraction import TattooTextExtractor
from ai_brain.llm_health import is_quota_exhausted_error
from ai_brain.processor import StudioAIBrain
from ai_brain.routing import TattooRouter
from ai_brain.summary import HighRiskSummaryBuilder
from ai_brain.vector_store import VectorStoreManager
from tests.test_processor import StubVisionAnalyzer


def _quota_error() -> openai.RateLimitError:
    request = httpx.Request("POST", "https://api.openai.com/v1/responses")
    return openai.RateLimitError(
        "Error code: 429 - You have no credits remaining.",
        response=httpx.Response(429, request=request),
        body={
            "error": {
                "message": "You have no credits remaining.",
                "type": "insufficient_quota",
                "code": "credit_balance_exhausted",
            }
        },
    )


class FailingLLM:
    """Raise the configured error on every model call."""

    def __init__(self, error: Exception) -> None:
        self._error = error

    def invoke(self, messages: object) -> object:
        raise self._error


def _brain(error: Exception) -> StudioAIBrain:
    llm = cast(ChatOpenAI, FailingLLM(error))
    vector_store = Mock(spec=VectorStoreManager)
    vector_store.records = tuple(range(10))
    vector_store.search_similar_cases.return_value = []
    return StudioAIBrain(
        vision_analyzer=StubVisionAnalyzer(tags=["unknown"]),
        text_extractor=TattooTextExtractor(llm=llm),
        router=TattooRouter(
            llm=llm,
            vector_store=cast(VectorStoreManager, vector_store),
        ),
    )


def _ask(brain: StudioAIBrain, source: str = "whatsapp"):
    return brain.process_inquiry(
        current_message="I wanna do a tattoo in my abs/side-abs area",
        existing_db_state={"lead": {"name": "Fahim Sarker"}},
        recent_chat_history=[],
        message_source=source,
    )


def test_quota_error_is_detected_directly_and_when_wrapped() -> None:
    error = _quota_error()
    wrapped = RuntimeError("Tattoo detail extraction failed.")
    wrapped.__cause__ = error

    assert is_quota_exhausted_error(error) is True
    assert is_quota_exhausted_error(wrapped) is True
    assert is_quota_exhausted_error(TimeoutError("Request timed out.")) is False


def test_exhausted_quota_sends_no_whatsapp_reply() -> None:
    result = _ask(_brain(_quota_error()))

    assert result.draft_reply == ""
    assert result.auto_reply is False
    assert result.auto_reply_allowed is False
    assert result.staff_review_required is True
    assert result.telegram_review_required is True
    assert any("quota exhausted" in reason for reason in result.review_reasons)


def test_exhausted_quota_sends_no_outlook_reply() -> None:
    result = _ask(_brain(_quota_error()), source="outlook")

    assert result.draft_reply == ""
    assert result.auto_reply is False


def test_quota_state_does_not_leak_into_the_next_request() -> None:
    _ask(_brain(_quota_error()))
    result = _ask(_brain(TimeoutError("Request timed out.")))

    assert result.draft_reply != ""


def test_other_model_failures_keep_the_template_fallback() -> None:
    result = _ask(_brain(TimeoutError("Request timed out.")))

    assert result.draft_reply != ""
    assert "quota" not in result.ai_reasoning.casefold()


def test_telegram_message_explains_missing_draft() -> None:
    message = HighRiskSummaryBuilder().combine_with_draft(
        summary="Fahim wants an abs tattoo.",
        draft_reply="",
    )

    assert "reply to the client manually" in message
