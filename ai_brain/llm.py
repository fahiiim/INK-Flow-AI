"""Centralized LangChain model factories."""

from __future__ import annotations

from functools import lru_cache

from langchain_openai import ChatOpenAI

from .config import get_settings

_DETERMINISTIC_TEMPERATURE = 0.0


@lru_cache(maxsize=4)
def get_chat_model(model_name: str = "gpt-4o") -> ChatOpenAI:
    """Return a cached ChatOpenAI client for the requested model."""
    settings = get_settings()
    selected_model = model_name.strip() or "gpt-4o"

    return ChatOpenAI(
        api_key=settings.api_key.get_secret_value(),
        model=selected_model,
        temperature=_DETERMINISTIC_TEMPERATURE,
        timeout=settings.timeout_seconds,
        max_retries=settings.max_retries,
    )
