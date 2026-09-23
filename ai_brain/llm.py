"""Centralized LangChain model factories."""

from __future__ import annotations

from functools import lru_cache
from typing import Any

from langchain_openai import ChatOpenAI

from .config import get_settings

_DETERMINISTIC_TEMPERATURE = 0.0
_REASONING_MODEL_PREFIXES = ("gpt-5", "gpt-6", "o1", "o3", "o4")


@lru_cache(maxsize=4)
def get_chat_model(model_name: str | None = None) -> ChatOpenAI:
    """Return a cached ChatOpenAI client for the requested model."""
    settings = get_settings()
    selected_model = (model_name or settings.model_name).strip()

    client_options: dict[str, Any] = {
        "api_key": settings.api_key.get_secret_value(),
        "model": selected_model,
        "timeout": settings.timeout_seconds,
        "max_retries": settings.max_retries,
    }
    if selected_model.casefold().startswith(_REASONING_MODEL_PREFIXES):
        client_options["reasoning_effort"] = settings.reasoning_effort
        client_options["use_responses_api"] = True
    else:
        client_options["temperature"] = _DETERMINISTIC_TEMPERATURE
    return ChatOpenAI(**client_options)
