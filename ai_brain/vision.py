"""Vision analysis module for tattoo style detection."""

from __future__ import annotations

import base64
import json
import logging
import re
from typing import Any, cast

import httpx
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI
from pydantic import AnyHttpUrl, TypeAdapter, ValidationError

from .llm import get_chat_model
from .prompts import VISION_SYSTEM_PROMPT
from .schemas import STYLE_TAG_OPTIONS, StyleTag

LOGGER = logging.getLogger(__name__)

_STYLE_TAG_SET = set(STYLE_TAG_OPTIONS)

class TattooVisionAnalyzer:
    """Analyze reference images and return normalized style tags."""

    def __init__(
        self,
        llm: ChatOpenAI | None = None,
        model_name: str = "gpt-4o",
        timeout_seconds: float = 20.0,
    ) -> None:
        self._llm = llm or get_chat_model(model_name=model_name)
        self._timeout_seconds = timeout_seconds
        self._url_adapter = TypeAdapter(AnyHttpUrl)

    def analyze_styles(self, image_urls: list[str]) -> list[StyleTag]:
        """Detect tattoo styles from image URLs.

        Invalid URLs, download issues, or API failures return ["unknown"].
        """
        if not image_urls:
            return ["unknown"]

        data_uris: list[str] = []
        for image_url in image_urls:
            data_uri = self._download_as_data_uri(image_url)
            if data_uri:
                data_uris.append(data_uri)

        if not data_uris:
            return ["unknown"]

        try:
            raw_text = self._invoke_vision_model(data_uris)
            tags = self._parse_style_tags(raw_text)
            if tags:
                return tags
        except Exception as exc:  # pragma: no cover - defensive branch
            LOGGER.warning("Vision analysis failed: %s", exc)

        return ["unknown"]

    def _download_as_data_uri(self, image_url: str) -> str | None:
        """Download an image URL and convert it into a data URI."""
        try:
            normalized_url = str(self._url_adapter.validate_python(image_url))
        except ValidationError:
            LOGGER.warning("Invalid image URL skipped: %s", image_url)
            return None

        try:
            response = httpx.get(
                normalized_url,
                timeout=self._timeout_seconds,
                follow_redirects=True,
            )
            response.raise_for_status()
        except httpx.HTTPError as exc:
            LOGGER.warning("Unable to fetch image %s: %s", normalized_url, exc)
            return None

        if not response.content:
            LOGGER.warning("Image URL returned empty data: %s", normalized_url)
            return None

        content_type = self._extract_content_type(
            response.headers.get("content-type", "")
        )
        encoded = base64.b64encode(response.content).decode("ascii")
        return f"data:{content_type};base64,{encoded}"

    def _extract_content_type(self, header_value: str) -> str:
        """Return a safe image MIME type from response headers."""
        normalized = header_value.split(";")[0].strip().lower()
        if normalized.startswith("image/"):
            return normalized
        return "image/jpeg"

    def _invoke_vision_model(self, data_uris: list[str]) -> str:
        """Call the vision-capable model and return text content."""
        content: list[dict[str, Any]] = []
        for data_uri in data_uris:
            content.append(
                {
                    "type": "image_url",
                    "image_url": {"url": data_uri},
                }
            )

        response = self._llm.invoke(
            [
                SystemMessage(content=VISION_SYSTEM_PROMPT),
                HumanMessage(content=content),
            ]
        )
        return self._coerce_content_to_text(response.content)

    def _coerce_content_to_text(self, content: Any) -> str:
        """Normalize LangChain response content to plain text."""
        if isinstance(content, str):
            return content
        if isinstance(content, list):
            text_parts: list[str] = []
            for item in content:
                if isinstance(item, dict):
                    text_value = item.get("text")
                    if isinstance(text_value, str):
                        text_parts.append(text_value)
            return "".join(text_parts)
        return str(content)

    def _parse_style_tags(self, raw_text: str) -> list[StyleTag]:
        """Parse and sanitize model output into approved style tags."""
        parsed: Any
        try:
            parsed = json.loads(raw_text)
        except json.JSONDecodeError:
            match = re.search(r"\[[\s\S]*\]", raw_text)
            if not match:
                return []
            try:
                parsed = json.loads(match.group(0))
            except json.JSONDecodeError:
                return []

        if not isinstance(parsed, list):
            return []

        cleaned: list[str] = []
        for item in parsed:
            if not isinstance(item, str):
                continue
            normalized = item.strip().lower()
            if normalized in _STYLE_TAG_SET and normalized not in cleaned:
                cleaned.append(normalized)

        if not cleaned:
            return []
        if "unknown" in cleaned and len(cleaned) > 1:
            cleaned = [tag for tag in cleaned if tag != "unknown"]

        return cast(list[StyleTag], cleaned)
