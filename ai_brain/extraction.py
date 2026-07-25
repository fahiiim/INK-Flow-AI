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
    Message,
    MissingInformationItem,
    StyleTag,
    TattooExtractionDraft,
    VisualColorPreference,
)

LOGGER = logging.getLogger(__name__)

_STYLE_TAG_SET = set(STYLE_TAG_OPTIONS)
_MISSING_SET = set(MISSING_INFORMATION_OPTIONS)
_STYLE_TEXT_ALIASES: dict[StyleTag, tuple[str, ...]] = {
    "fine-line": ("fine-line", "fine line", "fineline"),
    "watercolor": ("watercolor", "watercolour"),
    "minimal": ("minimal", "minimalist"),
    "floral": ("floral", "flower"),
    "micro-realism": ("micro-realism", "micro realism"),
    "black-and-grey": (
        "black-and-grey",
        "black and grey",
        "black and gray",
    ),
    "calligraphy": ("calligraphy", "lettering"),
    "traditional": ("traditional",),
    "geometric": ("geometric", "geometry"),
}
_DATE_MENTION_PATTERN = re.compile(
    r"\b(?:"
    r"\d{4}-\d{1,2}-\d{1,2}|"
    r"\d{1,2}[/\-]\d{1,2}(?:[/\-]\d{2,4})?|"
    r"todaye?|tomorrow|tonight|this morning|this afternoon|this evening|"
    r"next week|next month|"
    r"monday|tuesday|wednesday|thursday|friday|saturday|sunday|"
    r"january|february|march|april|may|june|july|august|september|"
    r"october|november|december"
    r")\b",
    flags=re.IGNORECASE,
)
_TIME_MENTION_PATTERN = re.compile(
    r"\b(?:at\s+)?\d{1,2}(?::[0-5]\d)?\s*(?:am|pm)\b",
    flags=re.IGNORECASE,
)


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
        current_message: str,
        style_tags: list[str],
        visual_color_preference: VisualColorPreference = "unknown",
        new_image_urls: list[str] | None = None,
        existing_db_state: dict[str, Any] | None = None,
        recent_chat_history: list[Message] | None = None,
    ) -> TattooExtractionDraft:
        """Extract details from the latest message and supplied context."""
        normalized_message = current_message.strip()
        safe_image_urls = list(new_image_urls or [])
        if not normalized_message and not safe_image_urls:
            raise AnalysisPipelineError(
                "current_message or new_image_urls must be provided."
            )
        if not normalized_message:
            normalized_message = (
                "Client sent reference image(s) without a text caption."
            )

        safe_db_state = dict(existing_db_state or {})
        safe_chat_history = list(recent_chat_history or [])
        conversation_text = self._user_conversation_text(
            current_message=normalized_message,
            recent_chat_history=safe_chat_history,
        )
        text_style_tags = self._detect_style_tags_from_text(conversation_text)
        normalized_tags = self._normalize_style_tags(
            [*style_tags, *text_style_tags]
        )

        try:
            llm_output = self._invoke_extraction_llm(
                current_message=normalized_message,
                style_tags=normalized_tags,
                visual_color_preference=visual_color_preference,
                new_image_urls=safe_image_urls,
                existing_db_state=safe_db_state,
                recent_chat_history=safe_chat_history,
            )
            visual_output = self._apply_visual_color_default(
                llm_output=llm_output,
                visual_color_preference=visual_color_preference,
            )
            resolved_output = self._apply_existing_state_defaults(
                llm_output=visual_output,
                existing_db_state=safe_db_state,
            )
            missing_information = self._finalize_missing_information(
                llm_output=resolved_output,
                current_message=normalized_message,
                new_image_urls=safe_image_urls,
                existing_db_state=safe_db_state,
                recent_chat_history=safe_chat_history,
                style_tags=normalized_tags,
            )
            return TattooExtractionDraft(
                tattoo_idea=resolved_output.tattoo_idea,
                style_tags=normalized_tags,
                placement=resolved_output.placement,
                size_estimate_cm=resolved_output.size_estimate_cm,
                color_preference=resolved_output.color_preference,
                missing_information=missing_information,
            )
        except Exception as exc:  # pragma: no cover - defensive branch
            LOGGER.warning("Text extraction fallback used: %s", exc)
            return self._build_fallback_draft(
                current_message=normalized_message,
                style_tags=normalized_tags,
                visual_color_preference=visual_color_preference,
                new_image_urls=safe_image_urls,
                existing_db_state=safe_db_state,
                recent_chat_history=safe_chat_history,
            )

    def _invoke_extraction_llm(
        self,
        current_message: str,
        style_tags: list[StyleTag],
        visual_color_preference: VisualColorPreference,
        new_image_urls: list[str],
        existing_db_state: dict[str, Any],
        recent_chat_history: list[Message],
    ) -> _ExtractionSubset:
        """Call the model and parse strict JSON output for extraction fields."""
        format_instructions = self._parser.get_format_instructions()
        human_prompt = build_extraction_human_prompt(
            current_message=current_message,
            style_tags=style_tags,
            visual_color_preference=visual_color_preference,
            new_image_urls=new_image_urls,
            existing_db_state=existing_db_state,
            recent_chat_history=recent_chat_history,
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

    def _detect_style_tags_from_text(self, text: str) -> list[str]:
        """Detect approved style names explicitly stated in conversation text."""
        normalized = text.casefold()
        detected: list[str] = []
        for style_tag, aliases in _STYLE_TEXT_ALIASES.items():
            if any(alias in normalized for alias in aliases):
                detected.append(style_tag)
        return detected

    def _apply_visual_color_default(
        self,
        llm_output: _ExtractionSubset,
        visual_color_preference: VisualColorPreference,
    ) -> _ExtractionSubset:
        """Use image color only when conversation extraction has no color."""
        color_preference = llm_output.color_preference
        if self._is_blank(color_preference):
            if visual_color_preference == "black-and-grey":
                color_preference = "black-and-grey"
            elif visual_color_preference == "color":
                color_preference = "color"
        return _ExtractionSubset(
            tattoo_idea=llm_output.tattoo_idea,
            placement=llm_output.placement,
            size_estimate_cm=llm_output.size_estimate_cm,
            color_preference=color_preference,
            missing_information=llm_output.missing_information,
        )

    def _finalize_missing_information(
        self,
        llm_output: _ExtractionSubset,
        current_message: str,
        new_image_urls: list[str],
        existing_db_state: dict[str, Any],
        recent_chat_history: list[Message],
        style_tags: list[StyleTag],
    ) -> list[MissingInformationItem]:
        """Reconcile missing fields against every supplied context source."""
        missing: set[str] = {
            item for item in llm_output.missing_information if item in _MISSING_SET
        }

        conversation_text = self._user_conversation_text(
            current_message=current_message,
            recent_chat_history=recent_chat_history,
        )
        checks: dict[MissingInformationItem, bool] = {
            "size in cm": self._is_blank(llm_output.size_estimate_cm),
            "placement": self._is_blank(llm_output.placement),
            "tattoo style": not any(tag != "unknown" for tag in style_tags),
            "color preference": self._is_blank(llm_output.color_preference),
            "reference images": not (
                new_image_urls
                or self._has_state_value(
                    existing_db_state,
                    ("reference_images", "image_urls", "images", "references"),
                )
                or self._mentions_reference_image(conversation_text)
            ),
            "preferred date": not (
                self._has_state_value(
                    existing_db_state,
                    ("preferred_date", "appointment_date", "requested_date"),
                )
                or self._mentions_preferred_date(conversation_text)
            ),
        }
        for item, is_missing in checks.items():
            if is_missing:
                missing.add(item)
            else:
                missing.discard(item)

        ordered = [
            item for item in MISSING_INFORMATION_OPTIONS if item in missing
        ]
        return cast(list[MissingInformationItem], ordered)

    def _build_fallback_draft(
        self,
        current_message: str,
        style_tags: list[StyleTag],
        visual_color_preference: VisualColorPreference,
        new_image_urls: list[str],
        existing_db_state: dict[str, Any],
        recent_chat_history: list[Message],
    ) -> TattooExtractionDraft:
        """Return a safe draft when the extraction call fails."""
        fallback = _ExtractionSubset(
            tattoo_idea=current_message[:220],
            placement="",
            size_estimate_cm=self._extract_size_from_text(current_message),
            color_preference="",
            missing_information=[],
        )
        visual_fallback = self._apply_visual_color_default(
            llm_output=fallback,
            visual_color_preference=visual_color_preference,
        )
        resolved_fallback = self._apply_existing_state_defaults(
            llm_output=visual_fallback,
            existing_db_state=existing_db_state,
        )
        missing = self._finalize_missing_information(
            llm_output=resolved_fallback,
            current_message=current_message,
            new_image_urls=new_image_urls,
            existing_db_state=existing_db_state,
            recent_chat_history=recent_chat_history,
            style_tags=style_tags,
        )
        return TattooExtractionDraft(
            tattoo_idea=resolved_fallback.tattoo_idea,
            style_tags=style_tags,
            placement=resolved_fallback.placement,
            size_estimate_cm=resolved_fallback.size_estimate_cm,
            color_preference=resolved_fallback.color_preference,
            missing_information=missing,
        )

    def _apply_existing_state_defaults(
        self,
        llm_output: _ExtractionSubset,
        existing_db_state: dict[str, Any],
    ) -> _ExtractionSubset:
        """Fill only blank extraction fields from existing database state."""
        return _ExtractionSubset(
            tattoo_idea=self._prefer_extracted_value(
                llm_output.tattoo_idea,
                existing_db_state,
                ("tattoo_idea", "idea", "concept"),
            ),
            placement=self._prefer_extracted_value(
                llm_output.placement,
                existing_db_state,
                ("placement",),
            ),
            size_estimate_cm=self._prefer_extracted_value(
                llm_output.size_estimate_cm,
                existing_db_state,
                ("size_estimate_cm", "size_cm", "size"),
            ),
            color_preference=self._prefer_extracted_value(
                llm_output.color_preference,
                existing_db_state,
                ("color_preference", "colour_preference", "color"),
            ),
            missing_information=llm_output.missing_information,
        )

    def _prefer_extracted_value(
        self,
        extracted_value: str,
        existing_db_state: dict[str, Any],
        state_keys: tuple[str, ...],
    ) -> str:
        """Prefer synthesized current context over existing database state."""
        if not self._is_blank(extracted_value):
            return extracted_value
        return self._get_state_text(existing_db_state, state_keys)

    def _get_state_text(
        self,
        existing_db_state: dict[str, Any],
        state_keys: tuple[str, ...],
    ) -> str:
        """Return the first non-empty scalar value for known state keys."""
        for key in state_keys:
            value = existing_db_state.get(key)
            if isinstance(value, str) and not self._is_blank(value):
                return value.strip()
            if isinstance(value, (int, float)):
                return str(value)
        return ""

    def _has_state_value(
        self,
        existing_db_state: dict[str, Any],
        state_keys: tuple[str, ...],
    ) -> bool:
        """Return whether database state contains a meaningful value."""
        for key in state_keys:
            value = existing_db_state.get(key)
            if isinstance(value, str) and not self._is_blank(value):
                return True
            if isinstance(value, (list, tuple, set, dict)) and value:
                return True
            if value is not None and not isinstance(value, (str, list, tuple, set, dict)):
                return True
        return False

    def _user_conversation_text(
        self,
        current_message: str,
        recent_chat_history: list[Message],
    ) -> str:
        """Combine current and prior user messages for deterministic checks."""
        user_messages = [
            message.content
            for message in recent_chat_history
            if message.role == "user"
        ]
        return " ".join([*user_messages, current_message])

    def _mentions_reference_image(self, text: str) -> bool:
        """Detect previously supplied reference images in conversation text."""
        normalized = text.lower()
        if "http://" in normalized or "https://" in normalized:
            return True
        pattern = (
            r"\b(sent|shared|attached|uploaded)\b.{0,30}"
            r"\b(image|photo|picture|reference)\b"
        )
        return bool(re.search(pattern, normalized))

    def _extract_size_from_text(self, text: str) -> str:
        """Extract an explicit centimeter size for safe fallback overrides."""
        pattern = r"\b\d+(?:\.\d+)?\s*(?:cm|centimeters?|centimetres?)\b"
        match = re.search(pattern, text, flags=re.IGNORECASE)
        return match.group(0).strip() if match else ""

    def _is_blank(self, value: str) -> bool:
        """Return True when the extracted field is effectively empty."""
        normalized = value.strip().lower()
        return normalized in {"", "unknown", "n/a", "none", "not provided"}

    def _mentions_preferred_date(self, text: str) -> bool:
        """Heuristic date mention detector for intake completeness checks."""
        return bool(
            _DATE_MENTION_PATTERN.search(text)
            or _TIME_MENTION_PATTERN.search(text)
        )
