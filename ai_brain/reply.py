"""Context-aware client reply composition for natural studio chat."""

from __future__ import annotations

import re
from collections.abc import Sequence

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
_SCHEDULE_TERMS = (
    "today",
    "tomorrow",
    "tonight",
    "preferred date",
    " am",
    " pm",
)
_MANUAL_REVIEW_TERMS = (
    "price",
    "pricing",
    "cost",
    "quote",
    "booking",
    "book an appointment",
    "complaint",
    "refund",
    "bad experience",
    "complex design advice",
)
_MISSING_QUESTIONS = {
    "size in cm": "What rough size in cm are you thinking?",
    "placement": "Where on the body would you like it?",
    "reference images": "Do you have a reference image you can send?",
    "color preference": "Would you like black-and-grey or colour?",
    "preferred date": "What date or time works best for you?",
}
_QUESTION_MARKERS = {
    "size in cm": ("size", " cm"),
    "placement": ("where", "body", "placement"),
    "reference images": ("reference", "image", "photo"),
    "color preference": ("black-and-grey", "colour", "color"),
    "preferred date": ("date", "time", "when"),
}


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
        if self._is_greeting_only(current_message):
            return self._greeting_reply(history)

        acknowledgement = self._acknowledgement(
            current_message=current_message,
            history=history,
        )
        if risk_level == "high" and self._requires_manual_review(
            current_message
        ):
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

    def _is_greeting_only(self, message: str) -> bool:
        """Return whether the latest message contains only a greeting."""
        return bool(_GREETING_PATTERN.fullmatch(message.strip()))

    def _requires_manual_review(self, message: str) -> bool:
        """Detect intents that should be handed to studio staff directly."""
        normalized = message.casefold()
        return any(term in normalized for term in _MANUAL_REVIEW_TERMS)

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