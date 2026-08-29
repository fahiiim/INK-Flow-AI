"""Staff-facing high-risk inquiry summary composition."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from .schemas import AIExtractionOutput, TattooInquiryInput


class HighRiskSummaryBuilder:
    """Create concise staff summaries from validated inquiry analysis."""

    def build(
        self,
        inquiry: TattooInquiryInput,
        analysis: AIExtractionOutput,
    ) -> str:
        """Build one narrative paragraph for a high-risk inquiry."""
        if analysis.risk_level != "high":
            raise ValueError("A Telegram summary requires high risk.")

        state = inquiry.existing_db_state
        client = self._client_description(state)
        idea = self._clean_text(analysis.tattoo_idea)
        if idea:
            opening = (
                f'{client} submitted a high-risk tattoo inquiry for "{idea}".'
            )
        else:
            opening = f"{client} submitted a high-risk tattoo inquiry."

        sentences = [opening]
        design_description = self._design_description(analysis)
        if design_description:
            sentences.append(design_description)
        sentences.append(self._reference_description(inquiry))

        appointment = self._appointment_description(analysis)
        if appointment:
            sentences.append(appointment)

        current_message = self._clean_text(inquiry.current_message)
        if current_message:
            sentences.append(
                f'The latest client message says, "{current_message}".'
            )

        if analysis.missing_information:
            sentences.append(
                "The remaining information needed is "
                f"{self._natural_list(analysis.missing_information)}."
            )
        else:
            sentences.append("No intake information is missing.")

        if analysis.suggested_artist != "Unclear":
            sentences.append(
                f"{analysis.suggested_artist} is the suggested artist with "
                f"{analysis.confidence_level} confidence."
            )
        else:
            sentences.append(
                "An artist still needs to be assigned manually, and the "
                f"routing confidence is {analysis.confidence_level}."
            )

        reasoning = self._clean_text(analysis.ai_reasoning)
        if reasoning:
            sentences.append(f"Staff review is required because {reasoning}")

        return " ".join(
            sentence if sentence.endswith((".", "!", "?")) else sentence + "."
            for sentence in sentences
        )

    def combine_with_draft(self, summary: str, draft_reply: str) -> str:
        """Place the client draft after the staff summary for Telegram."""
        return f"{summary}\n\nDRAFT REPLY\n{draft_reply}"

    def _client_description(
        self,
        state: dict[str, Any],
    ) -> str:
        """Describe known client identity and contact details naturally."""
        name = self._state_value(state, "lead_name", "name")
        lead_id = self._state_value(state, "lead_id", "id")
        phone = self._state_value(
            state,
            "lead_phone",
            "phone_number",
            "phone",
        )
        email = self._state_value(state, "lead_email", "email")

        client = name or "The client"
        details: list[str] = []
        if lead_id:
            details.append(f"lead ID {lead_id}")
        if phone:
            details.append(f"phone {phone}")
        if email:
            details.append(f"email {email}")
        if details:
            return f"{client} ({', '.join(details)})"
        return client

    def _design_description(self, analysis: AIExtractionOutput) -> str:
        """Describe known design attributes as one natural sentence."""
        clauses: list[str] = []
        style = self._style(analysis)
        if style:
            if len(style) == 1:
                clauses.append(f"uses a {style[0]} style")
            else:
                clauses.append(
                    f"uses {self._natural_list(style)} styles"
                )
        if self._is_known(analysis.size_estimate_cm):
            clauses.append(
                "measures approximately "
                f"{self._clean_text(analysis.size_estimate_cm)}"
            )
        if self._is_known(analysis.placement):
            clauses.append(
                f"is intended for the {self._clean_text(analysis.placement)}"
            )
        if self._is_known(analysis.color_preference):
            clauses.append(
                "uses "
                f"{self._clean_text(analysis.color_preference)} ink"
            )
        if not clauses:
            return ""
        return f"The requested design {self._natural_list(clauses)}."

    def _reference_description(self, inquiry: TattooInquiryInput) -> str:
        """Describe how many new reference images accompany the inquiry."""
        count = len(inquiry.new_image_urls)
        if count == 0:
            return "No new reference images were provided."
        noun = "image" if count == 1 else "images"
        return f"The client provided {count} new reference {noun}."

    def _appointment_description(self, analysis: AIExtractionOutput) -> str:
        """Describe known preferred scheduling information."""
        if analysis.date and analysis.time:
            return (
                f"The preferred appointment is {analysis.date} at "
                f"{analysis.time}."
            )
        if analysis.date:
            return f"The preferred appointment date is {analysis.date}."
        if analysis.time:
            return f"The preferred appointment time is {analysis.time}."
        return ""

    def _style(self, analysis: AIExtractionOutput) -> list[str]:
        """Return known style tags for the narrative description."""
        return [tag for tag in analysis.style_tags if tag != "unknown"]

    def _state_value(self, state: dict[str, Any], *keys: str) -> str:
        """Read a known scalar from root state or a nested lead record."""
        records: list[Mapping[str, Any]] = [state]
        lead = state.get("lead")
        if isinstance(lead, Mapping):
            records.append(lead)
        for record in records:
            for key in keys:
                value = record.get(key)
                if isinstance(value, (str, int, float)):
                    normalized = self._clean_text(str(value))
                    if normalized:
                        return normalized
        return ""

    def _natural_list(self, values: list[str]) -> str:
        """Join values using natural English punctuation."""
        if not values:
            return ""
        if len(values) == 1:
            return values[0]
        if len(values) == 2:
            return f"{values[0]} and {values[1]}"
        return f"{', '.join(values[:-1])}, and {values[-1]}"

    def _is_known(self, value: str) -> bool:
        """Return whether a value is useful in a staff-facing summary."""
        return value.strip().casefold() not in {
            "",
            "unknown",
            "unclear",
            "none",
            "n/a",
            "not provided",
        }

    def _clean_text(self, value: str) -> str:
        """Collapse whitespace so the summary always remains one paragraph."""
        return " ".join(value.split())
