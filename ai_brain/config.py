"""Runtime configuration for AI Brain modules."""

from __future__ import annotations

import os
from functools import lru_cache

from dotenv import load_dotenv
from pydantic import BaseModel, Field, SecretStr, ValidationError


class LLMSettings(BaseModel):
    """Validated settings for OpenAI-backed LangChain clients."""

    api_key: SecretStr = Field(description="OpenAI API key.")
    temperature: float = Field(default=0.0, ge=0.0, le=1.0)
    timeout_seconds: int = Field(default=30, ge=1)
    max_retries: int = Field(default=2, ge=0)


@lru_cache(maxsize=1)
def get_settings() -> LLMSettings:
    """Load and cache environment-driven settings."""
    load_dotenv()

    api_key = os.getenv("OPENAI_API_KEY", "").strip()
    if not api_key:
        msg = "OPENAI_API_KEY is required in environment variables."
        raise ValueError(msg)

    payload = {
        "api_key": api_key,
        "temperature": os.getenv("OPENAI_TEMPERATURE", "0.0"),
        "timeout_seconds": os.getenv("OPENAI_TIMEOUT_SECONDS", "30"),
        "max_retries": os.getenv("OPENAI_MAX_RETRIES", "2"),
    }

    try:
        return LLMSettings.model_validate(payload)
    except ValidationError as exc:
        raise ValueError("Invalid AI Brain environment configuration.") from exc
