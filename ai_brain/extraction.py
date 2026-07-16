"""Text extraction and missing-information detection module."""

from __future__ import annotations

import logging
import re
from typing import Any, cast

from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.output_parsers import JsonOutputParser
from langchain_openai import ChatOpenAI
from pydantic import BaseModel, ConfigDict, Field

from .errors import AnalysisPipelineError
from .llm import get_chat_model
from .prompts import EXTRACTION_SYSTEM_PROMPT, build_extraction_human_prompt
from .schemas import (
    MISSING_INFORMATION_OPTIONS,
    STYLE_TAG_OPTIONS,
    MissingInformationItem,
    StyleTag,
    TattooExtractionDraft,
)

LOGGER = logging.getLogger(__name__)

_STYLE_TAG_SET = set(STYLE_TAG_OPTIONS)
_MISSING_SET = set(MISSING_INFORMATION_OPTIONS)

class _ExtractionSubset(BaseModel):
    """Subset schema used to parse extraction fields from the LLM."""

    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)

    tattoo_idea: str = Field(
        description="Core tattoo idea extracted from client text.",
    )
    placement: str = Field(
        description="Requested body placement for the tattoo.",
    )
    size_estimate_cm: str = Field(
        description="Tattoo size estimate in centimeters.",
    )
    color_preference: str = Field(
        description="Color preference or constraints.",
    )
    missing_information: list[MissingInformationItem] = Field(
        default_factory=list,
        description="Missing items from the required intake checklist.",
    )


class TattooTextExtractor:
    """Extract structured tattoo fields from inquiry text."""

    def __init__(
        self,
        llm: ChatOpenAI | None = None,
        model_name: str = "gpt-4o",
    ) -> None:
        self._llm = llm or get_chat_model(model_name=model_name)
        self._parser = JsonOutputParser(pydantic_object=_ExtractionSubset)

    def extract(
        self,
        client_text: str,
        style_tags: list[str],
        image_urls: list[str] | None = None,
    ) -> TattooExtractionDraft:
        """Extract tattoo details and identify missing intake information."""
        normalized_text = client_text.strip()
        if not normalized_text:
            raise AnalysisPipelineError("client_text must not be empty.")

        normalized_tags = self._normalize_style_tags(style_tags)
        safe_image_urls = image_urls or []

        try:
            llm_output = self._invoke_extraction_llm(
                client_text=normalized_text,
                style_tags=normalized_tags,
                image_urls=safe_image_urls,
            )
            missing_information = self._finalize_missing_information(
                llm_output=llm_output,
                client_text=normalized_text,
                image_urls=safe_image_urls,
            )
            return TattooExtractionDraft(
                tattoo_idea=llm_output.tattoo_idea,
                style_tags=normalized_tags,
                placement=llm_output.placement,
                size_estimate_cm=llm_output.size_estimate_cm,
                color_preference=llm_output.color_preference,
                missing_information=missing_information,
            )
        except Exception as exc:  # pragma: no cover - defensive branch
            LOGGER.warning("Text extraction fallback used: %s", exc)
            return self._build_fallback_draft(
                client_text=normalized_text,
                style_tags=normalized_tags,
                image_urls=safe_image_urls,
            )

    def _invoke_extraction_llm(
        self,
        client_text: str,
        style_tags: list[StyleTag],
        image_urls: list[str],
    ) -> _ExtractionSubset:
        """Call the model and parse strict JSON output for extraction fields."""
        format_instructions = self._parser.get_format_instructions()
        human_prompt = build_extraction_human_prompt(
            client_text=client_text,
            style_tags=style_tags,
            image_urls=image_urls,
            required_items=MISSING_INFORMATION_OPTIONS,
            format_instructions=format_instructions,
        )

        response = self._llm.invoke(
            [
                SystemMessage(content=EXTRACTION_SYSTEM_PROMPT),
                HumanMessage(content=human_prompt),
            ]
        )
        payload = self._parser.parse(self._coerce_content_to_text(response.content))
        return _ExtractionSubset.model_validate(payload)

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

    def _normalize_style_tags(self, style_tags: list[str]) -> list[StyleTag]:
        """Normalize incoming style tags to the approved taxonomy."""
        cleaned: list[str] = []
        for tag in style_tags:
            normalized = tag.strip().lower()
            if normalized in _STYLE_TAG_SET and normalized not in cleaned:
                cleaned.append(normalized)

        if not cleaned:
            return ["unknown"]
        if "unknown" in cleaned and len(cleaned) > 1:
            cleaned = [tag for tag in cleaned if tag != "unknown"]

        return cast(list[StyleTag], cleaned)

    def _finalize_missing_information(
        self,
        llm_output: _ExtractionSubset,
        client_text: str,
        image_urls: list[str],
    ) -> list[MissingInformationItem]:
        """Apply deterministic checks on top of LLM missing-info output."""
        missing: set[str] = {
            item for item in llm_output.missing_information if item in _MISSING_SET
        }

        if self._is_blank(llm_output.size_estimate_cm):
            missing.add("size in cm")
        if self._is_blank(llm_output.placement):
            missing.add("placement")
        if self._is_blank(llm_output.color_preference):
            missing.add("color preference")
        if not image_urls:
            missing.add("reference images")
        if not self._mentions_preferred_date(client_text):
            missing.add("preferred date")

        ordered = [
            item for item in MISSING_INFORMATION_OPTIONS if item in missing
        ]
        return cast(list[MissingInformationItem], ordered)

    def _build_fallback_draft(
        self,
        client_text: str,
        style_tags: list[StyleTag],
        image_urls: list[str],
    ) -> TattooExtractionDraft:
        """Return a safe draft when the extraction call fails."""
        fallback = _ExtractionSubset(
            tattoo_idea=client_text[:220],
            placement="",
            size_estimate_cm="",
            color_preference="",
            missing_information=[],
        )
        missing = self._finalize_missing_information(
            llm_output=fallback,
            client_text=client_text,
            image_urls=image_urls,
        )
        return TattooExtractionDraft(
            tattoo_idea=fallback.tattoo_idea,
            style_tags=style_tags,
            placement=fallback.placement,
            size_estimate_cm=fallback.size_estimate_cm,
            color_preference=fallback.color_preference,
            missing_information=missing,
        )

    def _is_blank(self, value: str) -> bool:
        """Return True when the extracted field is effectively empty."""
        normalized = value.strip().lower()
        return normalized in {"", "unknown", "n/a", "none", "not provided"}

    def _mentions_preferred_date(self, text: str) -> bool:
        """Heuristic date mention detector for intake completeness checks."""
        normalized = text.lower()
        pattern = (
            r"\b(\d{1,2}[/\-]\d{1,2}([/\-]\d{2,4})?|"
            r"monday|tuesday|wednesday|thursday|friday|saturday|sunday|"
            r"next week|next month|january|february|march|april|may|june|"
            r"july|august|september|october|november|december)\b"
        )
        return bool(re.search(pattern, normalized))
