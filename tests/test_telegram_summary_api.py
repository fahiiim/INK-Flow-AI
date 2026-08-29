"""Tests for the high-risk Telegram summary API endpoint."""

from __future__ import annotations

from typing import Any

from fastapi.testclient import TestClient

from ai_brain.schemas import AIExtractionOutput, Message
from api.dependencies import get_ai_brain
from api.main import create_app


class StubAIBrain:
    """Return one configured analysis and track endpoint invocations."""

    def __init__(self, result: AIExtractionOutput) -> None:
        self._result = result
        self.calls: list[dict[str, object]] = []

    def process_inquiry(
        self,
        current_message: str | None = None,
        new_image_urls: list[str] | None = None,
        existing_db_state: dict[str, Any] | None = None,
        recent_chat_history: list[Message] | None = None,
        message_source: str | None = None,
    ) -> AIExtractionOutput:
        """Record validated context and return the configured analysis."""
        self.calls.append(
            {
                "current_message": current_message,
                "new_image_urls": new_image_urls or [],
                "existing_db_state": existing_db_state or {},
                "recent_chat_history": recent_chat_history or [],
                "message_source": message_source,
            }
        )
        return self._result


def _analysis(risk_level: str) -> AIExtractionOutput:
    """Build a valid high- or low-risk analysis fixture."""
    return AIExtractionOutput.model_validate(
        {
            "tattoo_idea": "Large traditional wolf with pricing request",
            "style_tags": ["traditional"],
            "placement": "back",
            "size_estimate_cm": "30cm",
            "color_preference": "black-and-grey",
            "date": "2026-09-04",
            "time": "14:30",
            "suggested_artist": "Hoss",
            "confidence_level": "high",
            "ai_reasoning": "Large traditional work and pricing need review.",
            "missing_information": ["reference images"],
            "risk_level": risk_level,
            "draft_reply": (
                "Got it. I'll have the studio team review this and get back "
                "to you."
            ),
        }
    )


def _client(brain: StubAIBrain) -> TestClient:
    """Create an isolated app with a deterministic AI dependency."""
    application = create_app()
    application.dependency_overrides[get_ai_brain] = lambda: brain
    return TestClient(application)


def _payload() -> dict[str, object]:
    """Return the shared hybrid input payload."""
    return {
        "current_message": "How much is a 30cm wolf tattoo?",
        "new_image_urls": [],
        "existing_db_state": {
            "lead_id": 3,
            "lead_name": "Samim Osman",
            "lead_phone": "8801775155760",
        },
        "recent_chat_history": [
            {
                "role": "user",
                "content": "I want it on my back.",
            }
        ],
    }


def test_high_risk_endpoint_returns_summary_then_draft_reply() -> None:
    """High-risk analysis returns staff summary and Telegram-ready text."""
    brain = StubAIBrain(_analysis("high"))

    with _client(brain) as client:
        response = client.post(
            "/api/v1/inquiries/telegram-summary",
            json=_payload(),
        )

    assert response.status_code == 200
    body = response.json()
    assert body["risk_level"] == "high"
    assert "\n" not in body["summary"]
    assert body["summary"].startswith(
        "Samim Osman (lead ID 3, phone 8801775155760) submitted a "
        "high-risk tattoo inquiry"
    )
    assert "uses a traditional style" in body["summary"]
    assert "measures approximately 30cm" in body["summary"]
    assert "is intended for the back" in body["summary"]
    assert "preferred appointment is 2026-09-04 at 14:30" in body["summary"]
    assert "No new reference images were provided" in body["summary"]
    assert "remaining information needed is reference images" in body["summary"]
    assert "Hoss is the suggested artist with high confidence" in body["summary"]
    assert body["draft_reply"] == _analysis("high").draft_reply
    assert body["telegram_message"].startswith(body["summary"])
    assert "\n\nDRAFT REPLY\n" in body["telegram_message"]
    assert body["telegram_message"].endswith(body["draft_reply"])
    assert len(brain.calls) == 1


def test_summary_endpoint_accepts_legacy_history_and_lead_payload() -> None:
    """Legacy summary callers can send chat_history and lead directly."""
    brain = StubAIBrain(_analysis("high"))

    with _client(brain) as client:
        response = client.post(
            "/api/v1/inquiries/telegram-summary",
            json={
                "chat_history": [
                    {
                        "role": "user",
                        "content": "I want a large floral tattoo on my hand.",
                    },
                    {
                        "role": "assistant",
                        "content": "Please confirm the appointment date.",
                    },
                    {
                        "role": "user",
                        "content": "What will be the price?",
                    },
                ],
                "lead": {
                    "lead_id": 3,
                    "lead_name": "Samim Osman",
                },
            },
        )

    assert response.status_code == 200
    assert brain.calls[0]["current_message"] == "What will be the price?"
    assert brain.calls[0]["existing_db_state"] == {
        "lead_id": 3,
        "lead_name": "Samim Osman",
    }


def test_low_risk_endpoint_returns_conflict_without_summary() -> None:
    """Low-risk inquiries are not formatted for Telegram escalation."""
    brain = StubAIBrain(_analysis("low"))

    with _client(brain) as client:
        response = client.post(
            "/api/v1/inquiries/telegram-summary",
            json=_payload(),
        )

    assert response.status_code == 409
    assert response.json() == {
        "detail": "Telegram summary is available only for high-risk inquiries."
    }
    assert len(brain.calls) == 1
