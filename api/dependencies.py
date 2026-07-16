"""FastAPI dependency providers."""

from __future__ import annotations

from functools import lru_cache

from fastapi import HTTPException, status

from ai_brain.errors import ConfigurationError
from ai_brain.processor import StudioAIBrain


@lru_cache(maxsize=1)
def get_ai_brain() -> StudioAIBrain:
    """Create and cache the AI Brain for reuse across requests."""
    try:
        return StudioAIBrain()
    except ConfigurationError as exc:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="AI service is not configured.",
        ) from exc
