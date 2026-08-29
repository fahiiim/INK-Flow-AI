"""Context-aware client reply composition for natural studio chat."""

from __future__ import annotations

import re
from collections.abc import Mapping, Sequence

from .schemas import Message, RiskLevel, TattooExtractionDraft

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
_MISSING_QUESTIONS = {
    "tattoo idea": "What tattoo idea or design do you have in mind?",
    "size in cm": "What rough size in cm are you thinking?",
    "placement": "Where on the body would you like it?",
    "reference images": "Do you have a reference image you can send?",
    "tattoo style": "What tattoo style would you like?",
    "color preference": "Would you like black-and-grey or colour?",
    "preferred date": "What date works best for you?",
    "preferred time": "What time works best for you?",
}
_MISSING_EMAIL_REQUESTS = {
    "tattoo idea": "Tattoo idea, design concept, wording, or story",
    "size in cm": "Approximate tattoo size in centimeters",
    "placement": "Intended body placement",
    "reference images": (
        "Reference images showing the design or style you have in mind"
    ),
    "tattoo style": "Preferred tattoo style",
    "color preference": "Color preference (black-and-grey or color)",
    "preferred date": "Preferred appointment date",
    "preferred time": "Preferred appointment time",
}
_QUESTION_MARKERS = {
    "tattoo idea": (
        "tattoo idea",
        "design do",
        "have in mind",
        "concept",
        "background story",
    ),
    "size in cm": ("size", " cm"),
    "placement": ("where", "body", "placement"),
    "reference images": ("reference", "image", "photo"),
    "tattoo style": ("tattoo style", "style would", "style do"),
    "color preference": ("black-and-grey", "colour", "color"),
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
        )
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
            self._outlook_salutation(existing_db_state),
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
    ) -> str:
        """Address an Outlook lead by first name when backend data has it."""
        state = existing_db_state or {}
        lead = state.get("lead")
        name: object = None
        if isinstance(lead, Mapping):
            name = lead.get("name")
        if not name:
            name = state.get("lead_name")
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
        if self._is_known_reply_value(extracted.tattoo_idea):
            details.append(
                ("Tattoo concept", self._email_value(extracted.tattoo_idea))
            )

        style_tags = [
            tag for tag in extracted.style_tags if tag != "unknown"
        ]
        if style_tags:
            details.append(("Style", ", ".join(style_tags)))
        for label, value in (
            ("Placement", extracted.placement),
            ("Approximate size", extracted.size_estimate_cm),
            ("Color preference", extracted.color_preference),
            ("Preferred date", extracted.date),
            ("Preferred time", extracted.time),
        ):
            if self._is_known_reply_value(value):
                details.append((label, self._email_value(value)))
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
        """Start naturally without sending the full intake checklist."""
        replies = (
            "Hey! What kind of tattoo are you thinking about?",
            "Hi! Tell me a little about the tattoo you have in mind.",
        )
        index = self._assistant_message_count(history) % len(replies)
        return self._avoid_exact_repeat(replies[index], history)

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
        if (
            "tattoo idea" in missing_information
            and not any(
                marker in previous
                for marker in _QUESTION_MARKERS["tattoo idea"]
            )
        ):
            return [_MISSING_QUESTIONS["tattoo idea"]]

        if (
            "reference images" in missing_information
            and not any(
                marker in previous
                for marker in _QUESTION_MARKERS["reference images"]
            )
        ):
            return [_MISSING_QUESTIONS["reference images"]]

        unresolved_visual_fields = [
            item
            for item in ("tattoo style", "color preference")
            if item in missing_information
        ]
        if unresolved_visual_fields:
            limit = 1 if self._assistant_message_count(history) else 2
            return [
                _MISSING_QUESTIONS[item]
                for item in unresolved_visual_fields[:limit]
            ]
        not_recently_asked = [
            item
            for item in missing_information
            if not any(
                marker in previous
                for marker in _QUESTION_MARKERS[item]
            )
        ]
        candidates = not_recently_asked or missing_information
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
