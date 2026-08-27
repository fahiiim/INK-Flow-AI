"""Text extraction and missing-information detection module."""

from __future__ import annotations

import logging
import re
from collections.abc import Callable
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

_IMAGE_ONLY_MESSAGE = "Client sent reference image(s) without a text caption."
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
_PLACEMENT_ALIASES: tuple[tuple[str, str], ...] = (
    ("behind the ear", "behind the ear"),
    ("behind my ear", "behind the ear"),
    ("inner forearm", "inner forearm"),
    ("outer forearm", "outer forearm"),
    ("inner wrist", "inner wrist"),
    ("outer wrist", "outer wrist"),
    ("upper arm", "upper arm"),
    ("lower arm", "lower arm"),
    ("shoulder blade", "shoulder blade"),
    ("rib cage", "rib cage"),
    ("collarbone", "collarbone"),
    ("forearm", "forearm"),
    ("sternum", "sternum"),
    ("shoulder", "shoulder"),
    ("wrist", "wrist"),
    ("ankle", "ankle"),
    ("chest", "chest"),
    ("thigh", "thigh"),
    ("calf", "calf"),
    ("back", "back"),
    ("neck", "neck"),
    ("hand", "hand"),
    ("foot", "foot"),
    ("arm", "arm"),
    ("leg", "leg"),
)
_SIZE_FIELD_TERMS = (
    "size",
    "centimeter",
    "centimetre",
    " cm",
    "bigger",
    "smaller",
)
_PLACEMENT_FIELD_TERMS = (
    "placement",
    "body part",
    "where on",
    *tuple(alias for alias, _ in _PLACEMENT_ALIASES),
)
_COLOR_FIELD_TERMS = (
    "color",
    "colour",
    "black ink",
    "black-and-grey",
    "black and grey",
    "black and gray",
    "black & grey",
    "black & gray",
)
_NEGATION_PREFIX_PATTERN = re.compile(
    r"(?:\bnot|\bno|instead\s+of|rather\s+than|\bfrom)\s+$",
    flags=re.IGNORECASE,
)
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
            normalized_message = _IMAGE_ONLY_MESSAGE

        safe_db_state = dict(existing_db_state or {})
        safe_chat_history = list(recent_chat_history or [])
        current_style_tags = self._detect_style_tags_from_text(
            normalized_message
        )
        history_text = self._user_history_text(safe_chat_history)
        history_style_tags = self._detect_style_tags_from_text(history_text)
        text_style_tags = current_style_tags or history_style_tags
        negated_style_tags = self._detect_negated_style_tags(
            normalized_message
        )
        normalized_tags = self._normalize_style_tags(
            [
                tag
                for tag in [*style_tags, *text_style_tags]
                if tag not in negated_style_tags
            ]
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
            resolved_output = self._apply_context_defaults(
                llm_output=visual_output,
                current_message=normalized_message,
                recent_chat_history=safe_chat_history,
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

        try:
            response = self._llm.invoke(
                [
                    SystemMessage(content=EXTRACTION_SYSTEM_PROMPT),
                    HumanMessage(content=human_prompt),
                ]
            )
            response_text = self._coerce_content_to_text(response.content)
            payload = self._parser.parse(response_text)
            return _ExtractionSubset.model_validate(payload)
        except Exception as exc:
            raise AnalysisPipelineError(
                "Tattoo detail extraction failed."
            ) from exc

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
            if any(
                self._has_positive_phrase(normalized, alias)
                for alias in aliases
            ):
                detected.append(style_tag)
        return detected

    def _detect_negated_style_tags(self, text: str) -> set[StyleTag]:
        """Return styles explicitly rejected by the latest client message."""
        normalized = text.casefold()
        rejected: set[StyleTag] = set()
        for style_tag, aliases in _STYLE_TEXT_ALIASES.items():
            if any(
                self._has_negated_phrase(normalized, alias)
                for alias in aliases
            ):
                rejected.add(style_tag)
        return rejected

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
            tattoo_idea=(
                ""
                if current_message == _IMAGE_ONLY_MESSAGE
                else current_message[:220]
            ),
            placement="",
            size_estimate_cm=self._extract_size_from_text(current_message),
            color_preference="",
            missing_information=[],
        )
        visual_fallback = self._apply_visual_color_default(
            llm_output=fallback,
            visual_color_preference=visual_color_preference,
        )
        resolved_fallback = self._apply_context_defaults(
            llm_output=visual_fallback,
            current_message=current_message,
            recent_chat_history=recent_chat_history,
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

    def _apply_context_defaults(
        self,
        llm_output: _ExtractionSubset,
        current_message: str,
        recent_chat_history: list[Message],
        existing_db_state: dict[str, Any],
    ) -> _ExtractionSubset:
        """Resolve fields using current, model, history, then database order."""
        return _ExtractionSubset(
            tattoo_idea=self._prefer_extracted_value(
                llm_output.tattoo_idea,
                existing_db_state,
                ("tattoo_idea", "idea", "concept"),
            ),
            placement=self._resolve_context_field(
                llm_value=llm_output.placement,
                current_message=current_message,
                recent_chat_history=recent_chat_history,
                existing_db_state=existing_db_state,
                state_keys=("placement",),
                value_extractor=self._extract_placement_from_text,
                field_terms=_PLACEMENT_FIELD_TERMS,
            ),
            size_estimate_cm=self._resolve_context_field(
                llm_value=llm_output.size_estimate_cm,
                current_message=current_message,
                recent_chat_history=recent_chat_history,
                existing_db_state=existing_db_state,
                state_keys=("size_estimate_cm", "size_cm", "size"),
                value_extractor=self._extract_size_from_text,
                field_terms=_SIZE_FIELD_TERMS,
            ),
            color_preference=self._resolve_context_field(
                llm_value=llm_output.color_preference,
                current_message=current_message,
                recent_chat_history=recent_chat_history,
                existing_db_state=existing_db_state,
                state_keys=(
                    "color_preference",
                    "colour_preference",
                    "color",
                ),
                value_extractor=self._extract_color_from_text,
                field_terms=_COLOR_FIELD_TERMS,
            ),
            missing_information=llm_output.missing_information,
        )

    def _resolve_context_field(
        self,
        llm_value: str,
        current_message: str,
        recent_chat_history: list[Message],
        existing_db_state: dict[str, Any],
        state_keys: tuple[str, ...],
        value_extractor: Callable[[str], str],
        field_terms: tuple[str, ...],
    ) -> str:
        """Resolve one field while enforcing latest-message precedence."""
        current_value = value_extractor(current_message)
        if current_value:
            return current_value
        if not self._is_blank(llm_value):
            return llm_value
        if self._mentions_field(current_message, field_terms):
            return ""

        history_value = self._latest_history_value(
            recent_chat_history=recent_chat_history,
            value_extractor=value_extractor,
        )
        if history_value:
            return history_value
        return self._get_state_text(existing_db_state, state_keys)

    def _latest_history_value(
        self,
        recent_chat_history: list[Message],
        value_extractor: Callable[[str], str],
    ) -> str:
        """Extract the latest known value from prior client messages."""
        for message in reversed(recent_chat_history):
            if message.role != "user":
                continue
            value = value_extractor(message.content)
            if value:
                return value
        return ""

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

    def _user_history_text(
        self,
        recent_chat_history: list[Message],
    ) -> str:
        """Combine only prior client messages for style fallback checks."""
        return " ".join(
            message.content
            for message in recent_chat_history
            if message.role == "user"
        )

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
        matches = list(re.finditer(pattern, text, flags=re.IGNORECASE))
        return matches[-1].group(0).strip() if matches else ""

    def _extract_placement_from_text(self, text: str) -> str:
        """Extract the latest positively stated common body placement."""
        normalized = text.casefold()
        matches: list[tuple[int, str]] = []
        for alias, canonical in _PLACEMENT_ALIASES:
            pattern = rf"\b{re.escape(alias)}\b"
            for match in re.finditer(pattern, normalized):
                if not self._phrase_is_negated(normalized, match.start()):
                    matches.append((match.start(), canonical))
        if not matches:
            return ""
        return max(matches, key=lambda item: item[0])[1]

    def _extract_color_from_text(self, text: str) -> str:
        """Normalize an explicit latest-message color preference."""
        normalized = text.casefold()
        no_color_pattern = r"\b(?:no|without)\s+colou?r\b"
        if re.search(no_color_pattern, normalized):
            return "black-and-grey"

        patterns = (
            (
                r"\b(?:black[- ]and[- ]gr[ae]y|black\s*&\s*gr[ae]y|"
                r"black\s+ink(?:\s+only)?|black\s+only)\b",
                "black-and-grey",
            ),
            (
                r"\b(?:full\s+colou?r|colou?r(?:ed|ful)?|"
                r"red|blue|green|yellow|purple|orange|pink)\b",
                "color",
            ),
        )
        matches: list[tuple[int, str]] = []
        for pattern, value in patterns:
            for match in re.finditer(pattern, normalized):
                if not self._phrase_is_negated(normalized, match.start()):
                    matches.append((match.start(), value))
        if not matches:
            return ""
        return max(matches, key=lambda item: item[0])[1]

    def _mentions_field(
        self,
        text: str,
        field_terms: tuple[str, ...],
    ) -> bool:
        """Return whether the latest message intentionally addresses a field."""
        normalized = text.casefold()
        return any(term in normalized for term in field_terms)

    def _has_positive_phrase(self, text: str, phrase: str) -> bool:
        """Return whether a phrase appears outside a rejection expression."""
        pattern = rf"\b{re.escape(phrase)}\b"
        return any(
            not self._phrase_is_negated(text, match.start())
            for match in re.finditer(pattern, text)
        )

    def _has_negated_phrase(self, text: str, phrase: str) -> bool:
        """Return whether a phrase is explicitly rejected in nearby text."""
        pattern = rf"\b{re.escape(phrase)}\b"
        return any(
            self._phrase_is_negated(text, match.start())
            for match in re.finditer(pattern, text)
        )

    def _phrase_is_negated(self, text: str, phrase_start: int) -> bool:
        """Detect a rejection marker immediately before a matched phrase."""
        prefix = text[max(0, phrase_start - 24):phrase_start]
        return bool(_NEGATION_PREFIX_PATTERN.search(prefix))

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
