"""Versioned FastAPI route definitions."""

from __future__ import annotations

import logging
from typing import Annotated

from fastapi import APIRouter, Depends, HTTPException, status

from ai_brain.config import get_settings
from ai_brain.errors import AnalysisPipelineError, ConfigurationError
from ai_brain.processor import StudioAIBrain
from ai_brain.schemas import AIExtractionOutput, TattooInquiryInput

from .constants import SERVICE_NAME, SERVICE_VERSION
from .dependencies import get_ai_brain
from .schemas import ErrorResponse, HealthResponse

LOGGER = logging.getLogger(__name__)

health_router = APIRouter(prefix="/health", tags=["Health"])
inquiry_router = APIRouter(
    prefix="/api/v1/inquiries",
    tags=["Tattoo inquiries"],
)

BrainDependency = Annotated[StudioAIBrain, Depends(get_ai_brain)]


@health_router.get(
    "/live",
    response_model=HealthResponse,
    summary="Check service liveness",
)
def check_liveness() -> HealthResponse:
    """Confirm that the HTTP process is running."""
    return HealthResponse(
        status="ok",
        service=SERVICE_NAME,
        version=SERVICE_VERSION,
    )


@health_router.get(
    "/ready",
    response_model=HealthResponse,
    responses={
        status.HTTP_503_SERVICE_UNAVAILABLE: {
            "model": ErrorResponse,
            "description": "OpenAI configuration is missing or invalid.",
        }
    },
    summary="Check AI service readiness",
)
def check_readiness() -> HealthResponse:
    """Confirm that required AI configuration is valid."""
    try:
        get_settings()
    except ConfigurationError as exc:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="AI service is not configured.",
        ) from exc

    return HealthResponse(
        status="ready",
        service=SERVICE_NAME,
        version=SERVICE_VERSION,
    )


@inquiry_router.post(
    "/analyze",
    response_model=AIExtractionOutput,
    responses={
        status.HTTP_400_BAD_REQUEST: {
            "model": ErrorResponse,
            "description": "The inquiry cannot be processed.",
        },
        status.HTTP_502_BAD_GATEWAY: {
            "model": ErrorResponse,
            "description": "The upstream AI pipeline failed.",
        },
        status.HTTP_503_SERVICE_UNAVAILABLE: {
            "model": ErrorResponse,
            "description": "The AI service is not configured.",
        },
    },
    summary="Analyze a tattoo inquiry",
)
def analyze_inquiry(
    payload: TattooInquiryInput,
    brain: BrainDependency,
) -> AIExtractionOutput:
    """Analyze text and images through the unified AI Brain pipeline."""
    try:
        return brain.process_inquiry(
            text=payload.client_text,
            image_urls=payload.image_urls,
        )
    except AnalysisPipelineError as exc:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(exc),
        ) from exc
    except ConfigurationError as exc:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="AI service is not configured.",
        ) from exc
    except Exception as exc:
        LOGGER.exception("Unhandled AI pipeline failure")
        raise HTTPException(
            status_code=status.HTTP_502_BAD_GATEWAY,
            detail="AI analysis failed. Please retry later.",
        ) from exc
