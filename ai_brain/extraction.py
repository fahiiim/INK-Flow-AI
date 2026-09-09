"""Text extraction and missing-information detection module."""

from __future__ import annotations

import logging
import re
from collections.abc import Callable
from datetime import date as calendar_date
from datetime import timedelta
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
    PreferredArtist,
    ServiceCode,
    StyleTag,
    TattooExtractionDraft,
    TattooProjectType,
    VisualColorPreference,
)

LOGGER = logging.getLogger(__name__)

_IMAGE_ONLY_MESSAGE = "Client sent reference image(s) without a text caption."
_STYLE_TAG_SET = set(STYLE_TAG_OPTIONS)
_MISSING_SET = set(MISSING_INFORMATION_OPTIONS)
_GENERIC_TATTOO_IDEA_PATTERN = re.compile(
    r"^(?:(?:i|we|the client)\s+)?"
    r"(?:(?:want|wants|need|needs|would like|request|requests|"
    r"is interested in)\s+)?"
    r"(?:a\s+|an\s+|some\s+)?(?:new\s+)?tattoo"
    r"(?:\s+(?:idea|design|request|inquiry))?$",
    flags=re.IGNORECASE,
)
_GENERAL_HELP_PATTERN = re.compile(
    r"^(?:hi|hello|hey|good morning|good afternoon|good evening|"
    r"(?:can|could|would)\s+you\s+help(?:\s+me)?(?:\s+out)?|"
    r"(?:i\s+)?need\s+help|please\s+help)(?:\s+with\s+(?:a\s+)?"
    r"tattoo)?$",
    flags=re.IGNORECASE,
)
_CONCEPTLESS_INQUIRY_WORDS = {
    "a",
    "about",
    "an",
    "appointment",
    "book",
    "booking",
    "can",
    "consultation",
    "cost",
    "could",
    "do",
    "does",
    "estimate",
    "for",
    "get",
    "give",
    "hello",
    "hey",
    "hi",
    "how",
    "i",
    "inquiry",
    "is",
    "it",
    "know",
    "me",
    "much",
    "need",
    "of",
    "please",
    "price",
    "pricing",
    "quote",
    "rate",
    "request",
    "tattoo",
    "tattoos",
    "tell",
    "the",
    "this",
    "to",
    "want",
    "what",
    "will",
    "wondering",
    "would",
    "you",
}
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
_DATE_FIELD_TERMS = (
    "date",
    "day",
    "today",
    "todaye",
    "tomorrow",
    "monday",
    "tuesday",
    "wednesday",
    "thursday",
    "friday",
    "saturday",
    "sunday",
    "january",
    "february",
    "march",
    "april",
    "may",
    "june",
    "july",
    "august",
    "september",
    "october",
    "november",
    "december",
)
_TIME_FIELD_TERMS = (
    "time",
    "morning",
    "afternoon",
    "evening",
    " am",
    " pm",
)
_ARTIST_FIELD_TERMS = (
    "artist",
    "hoss",
    "nina",
    "lana",
    "sandra",
    "silva",
    "sliva",
    "no preference",
)
_SERVICE_FIELD_TERMS = (
    "consultation",
    "appointment",
    "revision session",
    "tattoo session",
    "in-person",
    "in person",
    "online",
    "visit the studio",
)
_AVAILABILITY_FIELD_TERMS = (
    "availability",
    "available",
    "date",
    "day",
    "weekday",
    "weekend",
    "morning",
    "afternoon",
    "evening",
    "anytime",
    "flexible",
    *_DATE_FIELD_TERMS,
)
_PROJECT_TYPE_FIELD_TERMS = (
    "new tattoo",
    "cover-up",
    "cover up",
    "continuation",
    "continue",
    "touch-up",
    "touch up",
)
_MONTH_NUMBERS = {
    "january": 1,
    "february": 2,
    "march": 3,
    "april": 4,
    "may": 5,
    "june": 6,
    "july": 7,
    "august": 8,
    "september": 9,
    "october": 10,
    "november": 11,
    "december": 12,
}
_WEEKDAY_NUMBERS = {
    "monday": 0,
    "tuesday": 1,
    "wednesday": 2,
    "thursday": 3,
    "friday": 4,
    "saturday": 5,
    "sunday": 6,
}
_NEGATION_PREFIX_PATTERN = re.compile(
    r"(?:\bnot|\bno|instead\s+of|rather\s+than|\bfrom)\s+$",
    flags=re.IGNORECASE,
)


class _ExtractionSubset(BaseModel):
    """Subset schema used to parse extraction fields from the LLM."""

    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)

    client_name: str = Field(
        default="",
        description="Client's full name.",
    )
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
    date: str = Field(
        default="",
        description="Preferred appointment date in YYYY-MM-DD format.",
    )
    time: str = Field(
        default="",
        description="Preferred appointment time in 24-hour HH:MM format.",
    )
    preferred_artist: PreferredArtist = Field(
        default="",
        description="Preferred artist or No preference.",
    )
    service_code: ServiceCode = Field(
        default="",
        description="Selected service code.",
    )
    availability: str = Field(
        default="",
        description="Preferred dates or general availability.",
    )
    tattoo_project_type: TattooProjectType = Field(
        default="",
        description="New tattoo, cover-up, continuation, or touch-up.",
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
                client_name=resolved_output.client_name,
                tattoo_idea=resolved_output.tattoo_idea,
                style_tags=normalized_tags,
                placement=resolved_output.placement,
                size_estimate_cm=resolved_output.size_estimate_cm,
                color_preference=resolved_output.color_preference,
                date=resolved_output.date,
                time=resolved_output.time,
                preferred_artist=resolved_output.preferred_artist,
                service_code=resolved_output.service_code,
                availability=resolved_output.availability,
                tattoo_project_type=resolved_output.tattoo_project_type,
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
            client_name=llm_output.client_name,
            tattoo_idea=llm_output.tattoo_idea,
            placement=llm_output.placement,
            size_estimate_cm=llm_output.size_estimate_cm,
            color_preference=color_preference,
            date=llm_output.date,
            time=llm_output.time,
            preferred_artist=llm_output.preferred_artist,
            service_code=llm_output.service_code,
            availability=llm_output.availability,
            tattoo_project_type=llm_output.tattoo_project_type,
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
            "client full name": self._is_missing_client_name(
                llm_output.client_name
            ),
            "tattoo idea": self._is_missing_tattoo_idea(
                llm_output.tattoo_idea
            ),
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
                or self._declines_reference_image(conversation_text)
            ),
            "preferred artist": self._is_blank(llm_output.preferred_artist),
            "service type": self._is_blank(llm_output.service_code),
            "preferred dates or availability": self._is_blank(
                llm_output.availability
            ),
            "tattoo project type": self._is_blank(
                llm_output.tattoo_project_type
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
        fallback_idea = (
            ""
            if (
                current_message == _IMAGE_ONLY_MESSAGE
                or self._last_assistant_asked_for_name(
                    recent_chat_history
                )
            )
            else current_message[:220]
        )
        if self._is_missing_tattoo_idea(fallback_idea):
            fallback_idea = ""
        fallback = _ExtractionSubset(
            client_name=self._extract_client_name_from_text(current_message),
            tattoo_idea=fallback_idea,
            placement="",
            size_estimate_cm=self._extract_size_from_text(current_message),
            color_preference="",
            date=self._extract_date_from_text(current_message),
            time=self._extract_time_from_text(current_message),
            preferred_artist=self._extract_preferred_artist_from_text(
                current_message
            ),
            service_code=self._extract_service_code_from_text(current_message),
            availability=self._extract_availability_from_text(current_message),
            tattoo_project_type=self._extract_project_type_from_text(
                current_message
            ),
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
            client_name=resolved_fallback.client_name,
            tattoo_idea=resolved_fallback.tattoo_idea,
            style_tags=style_tags,
            placement=resolved_fallback.placement,
            size_estimate_cm=resolved_fallback.size_estimate_cm,
            color_preference=resolved_fallback.color_preference,
            date=resolved_fallback.date,
            time=resolved_fallback.time,
            preferred_artist=resolved_fallback.preferred_artist,
            service_code=resolved_fallback.service_code,
            availability=resolved_fallback.availability,
            tattoo_project_type=resolved_fallback.tattoo_project_type,
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
            client_name=self._resolve_client_name(
                llm_value=llm_output.client_name,
                current_message=current_message,
                recent_chat_history=recent_chat_history,
                existing_db_state=existing_db_state,
            ),
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
            date=self._normalize_date_value(
                self._resolve_context_field(
                    llm_value=self._normalize_date_value(llm_output.date),
                    current_message=current_message,
                    recent_chat_history=recent_chat_history,
                    existing_db_state=existing_db_state,
                    state_keys=(
                        "date",
                        "preferred_date",
                        "appointment_date",
                        "requested_date",
                        "scheduled_date",
                    ),
                    value_extractor=self._extract_date_from_text,
                    field_terms=_DATE_FIELD_TERMS,
                )
            ),
            time=self._normalize_time_value(
                self._resolve_context_field(
                    llm_value=self._normalize_time_value(llm_output.time),
                    current_message=current_message,
                    recent_chat_history=recent_chat_history,
                    existing_db_state=existing_db_state,
                    state_keys=(
                        "time",
                        "preferred_time",
                        "appointment_time",
                        "requested_time",
                        "scheduled_time",
                    ),
                    value_extractor=self._extract_time_from_text,
                    field_terms=_TIME_FIELD_TERMS,
                )
            ),
            preferred_artist=self._normalize_preferred_artist(
                self._resolve_context_field(
                    llm_value=self._normalize_preferred_artist(
                        llm_output.preferred_artist
                    ),
                    current_message=current_message,
                    recent_chat_history=recent_chat_history,
                    existing_db_state=existing_db_state,
                    state_keys=(
                        "preferred_artist",
                        "requested_artist",
                        "assigned_artist",
                    ),
                    value_extractor=self._extract_preferred_artist_from_text,
                    field_terms=_ARTIST_FIELD_TERMS,
                )
            ),
            service_code=self._normalize_service_code(
                self._resolve_context_field(
                    llm_value=self._normalize_service_code(
                        llm_output.service_code
                    ),
                    current_message=current_message,
                    recent_chat_history=recent_chat_history,
                    existing_db_state=existing_db_state,
                    state_keys=(
                        "service_code",
                        "consultation_code",
                        "appointment_type",
                        "service_type",
                    ),
                    value_extractor=self._extract_service_code_from_text,
                    field_terms=_SERVICE_FIELD_TERMS,
                )
            ),
            availability=self._resolve_context_field(
                llm_value=llm_output.availability,
                current_message=current_message,
                recent_chat_history=recent_chat_history,
                existing_db_state=existing_db_state,
                state_keys=(
                    "availability",
                    "preferred_availability",
                    "preferred_dates",
                    "date",
                    "preferred_date",
                    "appointment_date",
                    "requested_date",
                    "scheduled_date",
                ),
                value_extractor=self._extract_availability_from_text,
                field_terms=_AVAILABILITY_FIELD_TERMS,
            ),
            tattoo_project_type=self._normalize_project_type(
                self._resolve_context_field(
                    llm_value=self._normalize_project_type(
                        llm_output.tattoo_project_type
                    ),
                    current_message=current_message,
                    recent_chat_history=recent_chat_history,
                    existing_db_state=existing_db_state,
                    state_keys=(
                        "tattoo_project_type",
                        "tattoo_type",
                        "project_type",
                        "work_type",
                    ),
                    value_extractor=self._extract_project_type_from_text,
                    field_terms=_PROJECT_TYPE_FIELD_TERMS,
                )
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

    def _resolve_client_name(
        self,
        llm_value: str,
        current_message: str,
        recent_chat_history: list[Message],
        existing_db_state: dict[str, Any],
    ) -> str:
        """Resolve explicit names and short answers to the full-name question."""
        current_name = self._extract_client_name_from_text(current_message)
        if current_name:
            return current_name
        if self._last_assistant_asked_for_name(recent_chat_history):
            short_answer = self._name_only_response(current_message)
            if short_answer:
                return short_answer
        if not self._is_missing_client_name(llm_value):
            return llm_value

        for index in range(len(recent_chat_history) - 1, 0, -1):
            message = recent_chat_history[index]
            previous = recent_chat_history[index - 1]
            if message.role != "user" or previous.role != "assistant":
                continue
            if "full name" not in previous.content.casefold():
                continue
            historical_name = self._name_only_response(message.content)
            if historical_name:
                return historical_name
        return self._get_state_text(
            existing_db_state,
            ("client_name", "full_name", "lead_name", "name"),
        )

    def _last_assistant_asked_for_name(
        self,
        recent_chat_history: list[Message],
    ) -> bool:
        """Return whether the most recent assistant turn requested a name."""
        for message in reversed(recent_chat_history):
            if message.role == "assistant":
                return "full name" in message.content.casefold()
        return False

    def _name_only_response(self, text: str) -> str:
        """Accept a two-to-four-word answer containing only name characters."""
        normalized = " ".join(text.split())
        if not re.fullmatch(
            r"[A-Za-z][A-Za-z'’-]*(?:\s+[A-Za-z][A-Za-z'’-]*){1,3}",
            normalized,
        ):
            return ""
        return normalized

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
        for record in self._state_records(existing_db_state):
            for key in state_keys:
                value = record.get(key)
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
        for record in self._state_records(existing_db_state):
            for key in state_keys:
                value = record.get(key)
                if isinstance(value, str) and not self._is_blank(value):
                    return True
                if isinstance(value, (list, tuple, set, dict)) and value:
                    return True
                if value is not None and not isinstance(
                    value,
                    (str, list, tuple, set, dict),
                ):
                    return True
        return False

    def _state_records(
        self,
        existing_db_state: dict[str, Any],
    ) -> list[dict[str, Any]]:
        """Return root and supported nested backend intake records."""
        records = [existing_db_state]
        for key in ("intake", "lead", "latest_ai_analysis"):
            value = existing_db_state.get(key)
            if isinstance(value, dict):
                records.append(value)
        return records

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

    def _declines_reference_image(self, text: str) -> bool:
        """Detect that the client cannot or does not have a reference."""
        normalized = " ".join(text.casefold().split())
        patterns = (
            r"\b(?:i\s+)?(?:do not|don't|dont|does not|doesn't|doesnt)\s+"
            r"have\b.{0,25}\b(?:reference|image|photo|picture)s?\b",
            r"\b(?:i\s+)?(?:have|got)\s+no\b.{0,25}"
            r"\b(?:reference|image|photo|picture)s?\b",
            r"\b(?:no|without)\s+(?:a\s+|any\s+)?(?:reference|"
            r"reference image|image|photo|picture)s?\b",
            r"\b(?:cannot|can't|cant|unable to)\s+"
            r"(?:send|share|provide|upload)\b.{0,25}"
            r"\b(?:reference|image|photo|picture)s?\b",
        )
        return any(re.search(pattern, normalized) for pattern in patterns)

    def _is_missing_tattoo_idea(self, value: str) -> bool:
        """Reject blank and generic values that contain no tattoo concept."""
        if self._is_blank(value):
            return True
        normalized = " ".join(
            re.sub(r"[^\w\s'-]", " ", value.casefold()).split()
        )
        if normalized in {
            "unknown",
            "not provided",
            "general inquiry",
            "general tattoo inquiry",
            "tattoo help",
        }:
            return True
        words = set(re.findall(r"[a-z]+", normalized))
        if words and words.issubset(_CONCEPTLESS_INQUIRY_WORDS):
            return True
        return bool(
            _GENERIC_TATTOO_IDEA_PATTERN.fullmatch(normalized)
            or _GENERAL_HELP_PATTERN.fullmatch(normalized)
        )

    def _is_missing_client_name(self, value: str) -> bool:
        """Require at least two name parts for the requested full name."""
        if self._is_blank(value):
            return True
        return len(value.split()) < 2

    def _extract_client_name_from_text(self, text: str) -> str:
        """Extract an explicitly introduced client full name."""
        pattern = re.compile(
            r"\b(?:my\s+full\s+name\s+is|my\s+name\s+is|call\s+me)\s+"
            r"([A-Za-z][A-Za-z'’-]*(?:\s+[A-Za-z][A-Za-z'’-]*){1,3})"
            r"(?=\s+(?:and|but|i|my|the)\b|[,.;!?\n]|$)",
            flags=re.IGNORECASE,
        )
        matches = list(pattern.finditer(text))
        if not matches:
            return ""
        return " ".join(matches[-1].group(1).split())

    def _extract_preferred_artist_from_text(self, text: str) -> str:
        """Extract one of the five client-selectable artist preferences."""
        normalized = " ".join(text.casefold().split())
        if re.search(
            r"\b(?:no\s+(?:artist\s+)?preference|any\s+artist|"
            r"whoever\s+(?:is|you\s+think)|you\s+(?:can\s+)?choose)\b",
            normalized,
        ):
            return "No preference"

        aliases = {
            "hoss": "Hoss",
            "nina": "Nina",
            "lana": "Lana",
            "sandra": "Sandra",
            "silva": "Silva",
            "sliva": "Silva",
        }
        matches: list[tuple[int, str]] = []
        for alias, display_name in aliases.items():
            for match in re.finditer(rf"\b{alias}\b", normalized):
                matches.append((match.start(), display_name))
        if matches:
            return max(matches, key=lambda item: item[0])[1]

        service_code = self._extract_service_code_from_text(text)
        if service_code.endswith("H"):
            return "Hoss"
        if service_code.endswith("N"):
            return "Nina"
        return ""

    def _normalize_preferred_artist(self, value: str) -> str:
        """Normalize artist spelling and reject unsupported preferences."""
        if self._is_blank(value):
            return ""
        extracted = self._extract_preferred_artist_from_text(value)
        return extracted

    def _extract_service_code_from_text(self, text: str) -> str:
        """Extract a service code from a code or natural-language choice."""
        code_match = re.search(
            r"\b(OCH|OCN|ORH|ORN|CH|CN|RH|RN|TH|TN)\b",
            text,
            flags=re.IGNORECASE,
        )
        if code_match:
            return code_match.group(1).upper()

        normalized = " ".join(text.casefold().split())
        artist = ""
        if re.search(r"\bhoss\b", normalized):
            artist = "H"
        elif re.search(r"\bnina\b", normalized):
            artist = "N"
        if not artist:
            return ""

        is_online = bool(re.search(r"\bonline\b", normalized))
        if re.search(r"\brevision(?:\s+session)?\b", normalized):
            return f"OR{artist}" if is_online else f"R{artist}"
        if re.search(r"\btattoo\s+session\b", normalized):
            return f"T{artist}"
        if re.search(r"\b(?:consultation|appointment)\b", normalized):
            return f"OC{artist}" if is_online else f"C{artist}"
        return ""

    def _normalize_service_code(self, value: str) -> str:
        """Normalize a selected service to its uppercase public code."""
        if self._is_blank(value):
            return ""
        return self._extract_service_code_from_text(value)

    def _extract_project_type_from_text(self, text: str) -> str:
        """Extract the tattoo project category requested by the client."""
        normalized = text.casefold()
        patterns = (
            (r"\bcover[- ]?up\b", "cover-up"),
            (r"\btouch[- ]?up\b", "touch-up"),
            (r"\b(?:continuation|continue|continuing)\b", "continuation"),
            (r"\bnew\s+tattoo\b", "new tattoo"),
        )
        matches: list[tuple[int, str]] = []
        for pattern, project_type in patterns:
            for match in re.finditer(pattern, normalized):
                matches.append((match.start(), project_type))
        if not matches:
            return ""
        return max(matches, key=lambda item: item[0])[1]

    def _normalize_project_type(self, value: str) -> str:
        """Normalize the tattoo project type to the response taxonomy."""
        if self._is_blank(value):
            return ""
        return self._extract_project_type_from_text(value)

    def _extract_availability_from_text(self, text: str) -> str:
        """Extract an exact date or a useful general availability phrase."""
        date_value = self._extract_date_from_text(text)
        if date_value:
            time_value = self._extract_time_from_text(text)
            if time_value:
                return f"{date_value} at {time_value}"
            return date_value

        patterns = (
            r"\b(?:weekdays?|weekends?)(?:\s+(?:mornings?|afternoons?|"
            r"evenings?))?\b",
            r"\b(?:mornings?|afternoons?|evenings?)\b",
            r"\b(?:any\s+day|any\s+time|anytime|flexible)\b",
            r"\bnext\s+(?:week|month)\b",
            r"\bafter\s+work\b",
        )
        matches: list[re.Match[str]] = []
        for pattern in patterns:
            matches.extend(re.finditer(pattern, text, flags=re.IGNORECASE))
        if not matches:
            return ""
        latest = max(matches, key=lambda match: match.start())
        return " ".join(latest.group(0).split())

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

    def _extract_date_from_text(self, text: str) -> str:
        """Extract and normalize the latest explicit preferred date."""
        today = calendar_date.today()
        candidates: list[tuple[int, calendar_date]] = []

        for match in re.finditer(r"\b\d{4}-\d{1,2}-\d{1,2}\b", text):
            parsed = self._safe_calendar_date(match.group(0))
            if parsed is not None:
                candidates.append((match.start(), parsed))

        numeric_pattern = r"\b(\d{1,2})[/\-](\d{1,2})[/\-](\d{2,4})\b"
        for match in re.finditer(numeric_pattern, text):
            day, month, year = (int(value) for value in match.groups())
            if year < 100:
                year += 2000
            parsed = self._safe_calendar_date(
                f"{year:04d}-{month:02d}-{day:02d}"
            )
            if parsed is not None:
                candidates.append((match.start(), parsed))

        month_names = "|".join(_MONTH_NUMBERS)
        month_first = re.compile(
            rf"\b({month_names})\s+(\d{{1,2}})(?:st|nd|rd|th)?"
            rf"(?:,?\s+(\d{{4}}))?\b",
            flags=re.IGNORECASE,
        )
        day_first = re.compile(
            rf"\b(\d{{1,2}})(?:st|nd|rd|th)?\s+({month_names})"
            rf"(?:\s+(\d{{4}}))?\b",
            flags=re.IGNORECASE,
        )
        for pattern, month_index, day_index, year_index in (
            (month_first, 1, 2, 3),
            (day_first, 2, 1, 3),
        ):
            for match in pattern.finditer(text):
                month = _MONTH_NUMBERS[match.group(month_index).casefold()]
                day = int(match.group(day_index))
                year_text = match.group(year_index)
                year = int(year_text) if year_text else today.year
                parsed = self._safe_calendar_date(
                    f"{year:04d}-{month:02d}-{day:02d}"
                )
                if parsed is None:
                    continue
                if year_text is None and parsed < today:
                    parsed = self._safe_calendar_date(
                        f"{year + 1:04d}-{month:02d}-{day:02d}"
                    )
                if parsed is not None:
                    candidates.append((match.start(), parsed))

        for match in re.finditer(
            r"\b(todaye?|tomorrow)\b",
            text,
            flags=re.IGNORECASE,
        ):
            offset = 1 if match.group(1).casefold() == "tomorrow" else 0
            candidates.append((match.start(), today + timedelta(days=offset)))

        weekday_names = "|".join(_WEEKDAY_NUMBERS)
        weekday_pattern = re.compile(
            rf"\b(next\s+)?({weekday_names})\b",
            flags=re.IGNORECASE,
        )
        for match in weekday_pattern.finditer(text):
            weekday = _WEEKDAY_NUMBERS[match.group(2).casefold()]
            offset = (weekday - today.weekday()) % 7
            if offset == 0:
                offset += 7
            candidates.append((match.start(), today + timedelta(days=offset)))

        if not candidates:
            return ""
        return max(candidates, key=lambda item: item[0])[1].isoformat()

    def _extract_time_from_text(self, text: str) -> str:
        """Extract and normalize the latest explicit preferred time."""
        candidates: list[tuple[int, str]] = []
        twelve_hour_pattern = re.compile(
            r"\b(1[0-2]|0?[1-9])(?::([0-5]\d))?\s*(am|pm)\b",
            flags=re.IGNORECASE,
        )
        for match in twelve_hour_pattern.finditer(text):
            hour = int(match.group(1)) % 12
            if match.group(3).casefold() == "pm":
                hour += 12
            minute = int(match.group(2) or 0)
            candidates.append((match.start(), f"{hour:02d}:{minute:02d}"))

        twenty_four_hour_pattern = re.compile(
            r"(?<!\d)([01]?\d|2[0-3]):([0-5]\d)(?!\d)"
        )
        for match in twenty_four_hour_pattern.finditer(text):
            candidates.append(
                (
                    match.start(),
                    f"{int(match.group(1)):02d}:{int(match.group(2)):02d}",
                )
            )

        if not candidates:
            return ""
        return max(candidates, key=lambda item: item[0])[1]

    def _normalize_date_value(self, value: str) -> str:
        """Normalize model or backend date text to the public contract."""
        if self._is_blank(value):
            return ""
        return self._extract_date_from_text(value)

    def _normalize_time_value(self, value: str) -> str:
        """Normalize model or backend time text to the public contract."""
        if self._is_blank(value):
            return ""
        return self._extract_time_from_text(value)

    def _safe_calendar_date(self, value: str) -> calendar_date | None:
        """Parse an ISO-like calendar date without allowing invalid dates."""
        try:
            return calendar_date.fromisoformat(value)
        except ValueError:
            return None

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
