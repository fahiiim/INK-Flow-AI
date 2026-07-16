"""HTTP contract tests for the FastAPI adapter."""

from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

from ai_brain.errors import AnalysisPipelineError, ConfigurationError
from ai_brain.schemas import AIExtractionOutput
from api.dependencies import get_ai_brain
from api.main import create_app


class StubAIBrain:
    """Configurable AI Brain stub for API dependency overrides."""

    def __init__(
        self,
        result: AIExtractionOutput | None = None,
        error: Exception | None = None,
    ) -> None:
        self._result = result
        self._error = error
        self.calls: list[tuple[str, list[str]]] = []

    def process_inquiry(
        self,
        text: str,
        image_urls: list[str],
    ) -> AIExtractionOutput:
        """Record request data and return or raise the configured outcome."""
        self.calls.append((text, image_urls))
        if self._error:
            raise self._error
        if self._result is None:
            raise RuntimeError("Stub result was not configured.")
        return self._result


def _successful_output() -> AIExtractionOutput:
    """Build a valid final response for endpoint tests."""
    return AIExtractionOutput(
        tattoo_idea="Fine-line lotus",
        style_tags=["fine-line"],
        placement="inner wrist",
        size_estimate_cm="4",
        color_preference="black-and-grey",
        suggested_artist="Nina",
        confidence_level="high",
        ai_reasoning="Fine-line work routes to Nina.",
        missing_information=[],
        risk_level="low",
        draft_reply="Thanks for the details. Nina is a strong match.",
    )


def _client_with_brain(brain: StubAIBrain) -> TestClient:
    """Create an isolated app with a deterministic AI dependency."""
    application = create_app()
    application.dependency_overrides[get_ai_brain] = lambda: brain
    return TestClient(application)


def test_liveness_endpoint() -> None:
    """Liveness remains available without initializing OpenAI clients."""
    with TestClient(create_app()) as client:
        response = client.get("/health/live")

    assert response.status_code == 200
    assert response.json() == {
        "status": "ok",
        "service": "ink-flow-ai",
        "version": "1.0.0",
    }


def test_readiness_returns_503_for_missing_configuration(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Readiness reports unavailable when required AI config is invalid."""

    def raise_configuration_error() -> None:
        raise ConfigurationError("OPENAI_API_KEY is missing.")

    monkeypatch.setattr(
        "api.routes.get_settings",
        raise_configuration_error,
    )

    with TestClient(create_app()) as client:
        response = client.get("/health/ready")

    assert response.status_code == 503
    assert response.json() == {"detail": "AI service is not configured."}


def test_analyze_endpoint_returns_strict_output() -> None:
    """Successful analysis returns the complete validated output contract."""
    output = _successful_output()
    brain = StubAIBrain(result=output)

    with _client_with_brain(brain) as client:
        response = client.post(
            "/api/v1/inquiries/analyze",
            json={
                "client_text": "I want a 4 cm fine-line lotus on my wrist.",
                "image_urls": ["https://example.com/lotus.jpg"],
            },
        )

    assert response.status_code == 200
    assert response.json() == output.model_dump(mode="json")
    assert brain.calls == [
        (
            "I want a 4 cm fine-line lotus on my wrist.",
            ["https://example.com/lotus.jpg"],
        )
    ]


def test_analyze_endpoint_rejects_invalid_payload() -> None:
    """FastAPI rejects empty client text before invoking the AI Brain."""
    brain = StubAIBrain(result=_successful_output())

    with _client_with_brain(brain) as client:
        response = client.post(
            "/api/v1/inquiries/analyze",
            json={"client_text": "", "image_urls": []},
        )

    assert response.status_code == 422
    assert brain.calls == []


def test_analyze_endpoint_maps_pipeline_error() -> None:
    """Known domain failures become safe client errors."""
    brain = StubAIBrain(
        error=AnalysisPipelineError("Inquiry could not be processed."),
    )

    with _client_with_brain(brain) as client:
        response = client.post(
            "/api/v1/inquiries/analyze",
            json={"client_text": "Valid inquiry", "image_urls": []},
        )

    assert response.status_code == 400
    assert response.json() == {
        "detail": "Inquiry could not be processed.",
    }


def test_analyze_endpoint_hides_unexpected_errors() -> None:
    """Unexpected failures return a generic upstream error response."""
    brain = StubAIBrain(error=RuntimeError("Sensitive internal failure"))

    with _client_with_brain(brain) as client:
        response = client.post(
            "/api/v1/inquiries/analyze",
            json={"client_text": "Valid inquiry", "image_urls": []},
        )

    assert response.status_code == 502
    assert response.json() == {
        "detail": "AI analysis failed. Please retry later.",
    }
