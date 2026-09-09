"""Context-aware client reply composition for natural studio chat."""

from __future__ import annotations

import re
from collections.abc import Mapping, Sequence

from .schemas import (
    ARTIST_PREFERENCE_OPTIONS,
    MISSING_INFORMATION_OPTIONS,
    SERVICE_OPTIONS,
    Message,
    RiskLevel,
    TattooExtractionDraft,
)

_GREETING_PATTERN = re.compile(
    r"^(?:hi|hello|hey|good morning|good afternoon|good evening)"
    r"(?:\s+(?:team|inkflow|ink flow))*[!.\s]*$",
    flags=re.IGNORECASE,
)
_CORRECTION_TERMS = (
    "i said",
    "actually",
    "i meant",
    "i told you",
)
_CONFIRMATION_PATTERN = re.compile(
    r"\b(?:yes|correct|confirmed|all good|looks good|that's right|"
    r"that is right|details are right)\b",
    flags=re.IGNORECASE,
)
_CHANGE_PATTERN = re.compile(
    r"\b(?:change|make it|instead|different|update)\b",
    flags=re.IGNORECASE,
)
_SCHEDULE_TERMS = (
    "today",
    "tomorrow",
    "tonight",
    "preferred date",
    "preferred time",
    "availability",
    " am",
    " pm",
)
_MANUAL_REVIEW_PATTERNS = (
    re.compile(r"\b(?:price|pricing|cost|quote|how much|budget)\b"),
    re.compile(r"\b(?:book|booking|booked|deposit)\b"),
    re.compile(
        r"\b(?:cancel|cancellation|reschedule|rescheduling|refund)\b"
    ),
    re.compile(
        r"\b(?:complaint|bad experience|unhappy|dissatisfied)\b"
    ),
    re.compile(r"\b(?:complex design advice|design advice|medical advice)\b"),
    re.compile(r"\b(?:specific|preferred|requested)\s+artist\b"),
    re.compile(
        r"\b(?:underage|minor|pregnant|pregnancy|infection|allergy|"
        r"medical condition)\b"
    ),
)
_SERVICE_CHOICE_LINES = tuple(
    f"{code} - {description}"
    for code, description in SERVICE_OPTIONS.items()
)
_SERVICE_CHOICE_QUESTION = (
    "Can you visit the studio, or do you need an online appointment? "
    "Please reply with one service code:\n"
    + "\n".join(_SERVICE_CHOICE_LINES)
)
_MISSING_QUESTIONS = {
    "client full name": "What is your full name?",
    "tattoo idea": "What is your tattoo idea or background story?",
    "size in cm": "What size would you prefer in centimetres?",
    "placement": "Where on your body would you like the tattoo?",
    "color preference": "Would you like colour or black and grey?",
    "tattoo style": "What tattoo style would you prefer?",
    "reference images": (
        "Could you share any reference or inspiration images?"
    ),
    "preferred artist": (
        "Do you have a preferred artist? Please choose "
        + ", ".join(ARTIST_PREFERENCE_OPTIONS[:-1])
        + f", {ARTIST_PREFERENCE_OPTIONS[-1]}, or say no preference."
    ),
    "service type": _SERVICE_CHOICE_QUESTION,
    "preferred dates or availability": (
        "What are your preferred dates or general availability?"
    ),
    "tattoo project type": (
        "Is this a new tattoo, cover-up, continuation, or touch-up?"
    ),
    "preferred date": "What date works best for you?",
    "preferred time": "What time works best for you?",
}
_MISSING_EMAIL_REQUESTS = dict(_MISSING_QUESTIONS)
_MISSING_EMAIL_REQUESTS["service type"] = (
    "Can you visit the studio, or do you need an online appointment? "
    "Please choose one service code:\n"
    + "\n".join(f"  - {line}" for line in _SERVICE_CHOICE_LINES)
)
_QUESTION_MARKERS = {
    "client full name": ("full name", "your name"),
    "tattoo idea": (
        "tattoo idea",
        "background story",
    ),
    "size in cm": ("size", "centimetres", "centimeters"),
    "placement": ("where on your body", "placement"),
    "color preference": ("black and grey", "colour", "color"),
    "tattoo style": ("tattoo style",),
    "reference images": ("reference", "inspiration image"),
    "preferred artist": ("preferred artist", "choose hoss"),
    "service type": ("service code", "visit the studio", "online appointment"),
    "preferred dates or availability": ("preferred dates", "availability"),
    "tattoo project type": (
        "new tattoo",
        "cover-up",
        "continuation",
        "touch-up",
    ),
    "preferred date": ("date", "day", "when"),
    "preferred time": ("time", " am", " pm"),
}


def requires_manual_review(
    current_message: str,
    recent_chat_history: Sequence[Message] | None = None,
) -> bool:
    """Return whether current or recent client intent requires staff review."""
    user_messages = [
        message.content
        for message in (recent_chat_history or [])
        if message.role == "user"
    ]
    combined = " ".join([*user_messages, current_message]).casefold()
    return any(pattern.search(combined) for pattern in _MANUAL_REVIEW_PATTERNS)


class ConversationReplyComposer:
    """Create concise replies that feel like an ongoing human conversation."""

    def compose(
        self,
        extracted: TattooExtractionDraft,
        current_message: str,
        recent_chat_history: list[Message] | None,
        risk_level: RiskLevel,
    ) -> str:
        """Acknowledge the latest turn and ask only the next useful question."""
        history = recent_chat_history or []
        manual_review = risk_level == "high" and requires_manual_review(
            current_message,
            history,
        )
        if self._is_greeting_only(current_message) and not manual_review:
            return self._greeting_reply(history)

        acknowledgement = self._acknowledgement(
            current_message=current_message,
            history=history,
        )
        if manual_review:
            reply = (
                f"{acknowledgement} I'll have the studio team review this "
                "and get back to you."
            )
            return self._avoid_exact_repeat(reply, history)

        questions = self._select_questions(
            missing_information=extracted.missing_information,
            history=history,
        )
        if questions:
            reply = " ".join([acknowledgement, *questions])
            return self._avoid_exact_repeat(reply, history)

        if risk_level == "high":
            reply = (
                f"{acknowledgement} I'll have the studio team review this "
                "and get back to you."
            )
            return self._avoid_exact_repeat(reply, history)

        reply = (
            f"{acknowledgement} I've got the main details now. "
            "I'll pass this to the team for a quick review."
        )
        return self._avoid_exact_repeat(reply, history)

    def compose_validation(
        self,
        extracted: TattooExtractionDraft,
        current_message: str,
        recent_chat_history: list[Message] | None,
        risk_level: RiskLevel,
    ) -> str:
        """Summarize extracted facts once, unless already confirmed."""
        history = recent_chat_history or []
        if risk_level == "high":
            return self.compose(
                extracted=extracted,
                current_message=current_message,
                recent_chat_history=history,
                risk_level=risk_level,
            )
        if self._is_greeting_only(current_message):
            return self.compose(
                extracted=extracted,
                current_message=current_message,
                recent_chat_history=history,
                risk_level=risk_level,
            )
        if self._details_already_confirmed(current_message, history):
            return self.compose(
                extracted=extracted,
                current_message=current_message,
                recent_chat_history=history,
                risk_level=risk_level,
            )

        summary = self._natural_summary(extracted)
        if not summary:
            return self.compose(
                extracted=extracted,
                current_message=current_message,
                recent_chat_history=history,
                risk_level=risk_level,
            )

        questions = self._select_questions(
            missing_information=extracted.missing_information,
            history=history,
        )[:1]
        reply_parts = [
            summary,
            "Does that sound right, or would you like to change anything?",
            *questions,
        ]
        return self._avoid_exact_repeat(
            " ".join(reply_parts),
            history,
        )

    def compose_outlook_email(
        self,
        extracted: TattooExtractionDraft,
        existing_db_state: Mapping[str, object] | None = None,
    ) -> str:
        """Create one professional email containing every missing request."""
        missing_information = list(extracted.missing_information)
        sections = [
            self._outlook_salutation(
                existing_db_state,
                extracted.client_name,
            ),
            (
                "Thank you for contacting Tattoo Hysteria. We have received "
                "your tattoo inquiry."
            ),
        ]

        known_details = self._outlook_known_details(extracted)
        if known_details:
            sections.append(
                "We have recorded the following details:\n"
                + "\n".join(
                    f"- {label}: {value}" for label, value in known_details
                )
            )

        if missing_information:
            requested_details = "\n".join(
                f"- {_MISSING_EMAIL_REQUESTS[item]}"
                for item in missing_information
            )
            sections.extend(
                [
                    (
                        "To help us review your request, please reply to this "
                        "email with all of the following information:\n"
                        f"{requested_details}"
                    ),
                    (
                        "Once we receive these details, our studio team will "
                        "review your inquiry and contact you with the next "
                        "steps."
                    ),
                ]
            )
        else:
            sections.append(
                "Our studio team will review your inquiry and contact you "
                "with the next steps."
            )

        sections.append("Kind regards,\nTattoo Hysteria")
        return "\n\n".join(sections)

    def _outlook_salutation(
        self,
        existing_db_state: Mapping[str, object] | None,
        extracted_client_name: str = "",
    ) -> str:
        """Address an Outlook lead by first name when backend data has it."""
        state = existing_db_state or {}
        lead = state.get("lead")
        name: object = None
        if isinstance(lead, Mapping):
            name = lead.get("name")
        if not name:
            name = state.get("lead_name")
        if not name:
            name = extracted_client_name
        if not isinstance(name, str) or not name.strip():
            return "Hello,"

        first_name = name.strip().split(maxsplit=1)[0]
        safe_name = re.sub(r"[^\w.'’\-]", "", first_name)
        if not safe_name:
            return "Hello,"
        return f"Dear {safe_name},"

    def _outlook_known_details(
        self,
        extracted: TattooExtractionDraft,
    ) -> list[tuple[str, str]]:
        """Return known intake facts as concise professional email rows."""
        details: list[tuple[str, str]] = []
        missing = set(extracted.missing_information)
        if (
            "client full name" not in missing
            and self._is_known_reply_value(extracted.client_name)
        ):
            details.append(("Client name", extracted.client_name))
        if (
            "tattoo idea" not in missing
            and self._is_known_reply_value(extracted.tattoo_idea)
        ):
            details.append(
                ("Tattoo concept", self._email_value(extracted.tattoo_idea))
            )

        style_tags = [
            tag for tag in extracted.style_tags if tag != "unknown"
        ]
        if style_tags and "tattoo style" not in missing:
            details.append(("Style", ", ".join(style_tags)))
        for missing_item, label, value in (
            ("placement", "Placement", extracted.placement),
            ("size in cm", "Approximate size", extracted.size_estimate_cm),
            (
                "color preference",
                "Color preference",
                extracted.color_preference,
            ),
            ("preferred date", "Preferred date", extracted.date),
            ("preferred time", "Preferred time", extracted.time),
            (
                "preferred artist",
                "Preferred artist",
                extracted.preferred_artist,
            ),
            (
                "preferred dates or availability",
                "Availability",
                extracted.availability,
            ),
            (
                "tattoo project type",
                "Tattoo project type",
                extracted.tattoo_project_type,
            ),
        ):
            if (
                missing_item not in missing
                and self._is_known_reply_value(value)
            ):
                details.append((label, self._email_value(value)))
        if extracted.service_code and "service type" not in missing:
            service = SERVICE_OPTIONS[extracted.service_code]
            details.append(
                ("Selected service", f"{extracted.service_code} - {service}")
            )
        return details

    def _email_value(self, value: str, limit: int = 240) -> str:
        """Keep a potentially long extracted value within reply limits."""
        normalized = " ".join(value.split())
        if len(normalized) <= limit:
            return normalized
        return normalized[: limit - 3].rstrip() + "..."

    def _is_greeting_only(self, message: str) -> bool:
        """Return whether the latest message contains only a greeting."""
        return bool(_GREETING_PATTERN.fullmatch(message.strip()))

    def _details_already_confirmed(
        self,
        current_message: str,
        history: list[Message],
    ) -> bool:
        """Detect the latest applicable client confirmation or correction."""
        user_messages = [
            message.content
            for message in history
            if message.role == "user"
        ]
        if not user_messages or user_messages[-1] != current_message:
            user_messages.append(current_message)

        confirmed = False
        for message in user_messages:
            normalized = message.casefold()
            if (
                any(term in normalized for term in _CORRECTION_TERMS)
                or _CHANGE_PATTERN.search(message)
            ):
                confirmed = False
            if _CONFIRMATION_PATTERN.search(message):
                confirmed = True
        return confirmed

    def _natural_summary(
        self,
        extracted: TattooExtractionDraft,
    ) -> str:
        """Summarize only known details in one conversational sentence."""
        style_tags = [
            tag for tag in extracted.style_tags if tag != "unknown"
        ]
        color_preference = extracted.color_preference
        if color_preference == "color":
            color_preference = "full-colour"
        if color_preference == "black-and-grey":
            style_tags = [
                tag for tag in style_tags if tag != "black-and-grey"
            ]
        descriptors = [
            value.strip()
            for value in (
                extracted.size_estimate_cm,
                color_preference,
                " and ".join(style_tags),
            )
            if self._is_known_reply_value(value)
        ]
        placement = ""
        if self._is_known_reply_value(extracted.placement):
            placement = extracted.placement.strip()
        if not descriptors and not placement:
            return ""

        subject = " ".join([*descriptors, "tattoo"])
        if placement:
            subject = f"{subject} on your {placement}"
        return f"Got it, a {subject}."

    def _is_known_reply_value(self, value: str) -> bool:
        """Reject empty and placeholder values from client-facing summaries."""
        normalized = value.strip().casefold()
        return normalized not in {
            "",
            "unknown",
            "none",
            "n/a",
            "not provided",
        }

    def _greeting_reply(self, history: list[Message]) -> str:
        """Start the required intake sequence with the client's full name."""
        return self._avoid_exact_repeat(
            "Hi! What is your full name?",
            history,
        )

    def _acknowledgement(
        self,
        current_message: str,
        history: list[Message],
    ) -> str:
        """Acknowledge corrections and newly supplied scheduling details."""
        normalized = current_message.casefold()
        if any(term in normalized for term in _CORRECTION_TERMS):
            return "You're right - I've noted that now."
        if any(term in normalized for term in _SCHEDULE_TERMS):
            return "Got it - I've noted the timing."

        alternatives = (
            "Got it.",
            "Okay, noted.",
            "Perfect, I've got that.",
        )
        index = self._assistant_message_count(history) % len(alternatives)
        return alternatives[index]

    def _assistant_message_count(self, history: list[Message]) -> int:
        """Count prior assistant turns for deterministic wording variation."""
        return sum(message.role == "assistant" for message in history)

    def _select_questions(
        self,
        missing_information: Sequence[str],
        history: list[Message],
    ) -> list[str]:
        """Choose one follow-up question, or two on the first intake turn."""
        previous = self._last_assistant_message(history).casefold()
        canonical = [
            item
            for item in MISSING_INFORMATION_OPTIONS
            if item in missing_information
        ]
        legacy = [
            item for item in missing_information if item not in canonical
        ]
        ordered_missing = [*canonical, *legacy]
        not_recently_asked = [
            item
            for item in ordered_missing
            if not any(
                marker in previous
                for marker in _QUESTION_MARKERS[item]
            )
        ]
        candidates = not_recently_asked or ordered_missing
        limit = 1 if self._assistant_message_count(history) else 2
        return [_MISSING_QUESTIONS[item] for item in candidates[:limit]]

    def _last_assistant_message(self, history: list[Message]) -> str:
        """Return the most recent assistant message or an empty string."""
        return next(
            (
                message.content
                for message in reversed(history)
                if message.role == "assistant"
            ),
            "",
        )

    def _avoid_exact_repeat(
        self,
        reply: str,
        history: list[Message],
    ) -> str:
        """Avoid returning the immediately previous assistant message verbatim."""
        previous = self._last_assistant_message(history)
        if not previous or previous.strip().casefold() != reply.casefold():
            return reply
        return f"I've got that. {reply}"
