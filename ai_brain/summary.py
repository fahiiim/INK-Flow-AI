"""Staff-facing high-risk inquiry summary composition."""

from __future__ import annotations

from typing import Any

from .schemas import AIExtractionOutput, TattooInquiryInput


class HighRiskSummaryBuilder:
    """Create concise staff summaries from validated inquiry analysis."""

    def build(
        self,
        inquiry: TattooInquiryInput,
        analysis: AIExtractionOutput,
    ) -> str:
        """Build a plain-text summary for a high-risk inquiry."""
        if analysis.risk_level != "high":
            raise ValueError("A Telegram summary requires high risk.")

        state = inquiry.existing_db_state
        lines = ["HIGH-RISK TATTOO INQUIRY"]
        self._append_state_value(lines, "Lead ID", state, "lead_id")
        self._append_state_value(lines, "Client", state, "lead_name")
        self._append_state_value(lines, "Phone", state, "lead_phone")
        lines.extend(
            [
                f"Current message: {self._current_message(inquiry)}",
                f"Reference images: {len(inquiry.new_image_urls)}",
                f"Tattoo idea: {self._display(analysis.tattoo_idea)}",
                f"Style: {self._style(analysis)}",
                f"Placement: {self._display(analysis.placement)}",
                f"Size: {self._display(analysis.size_estimate_cm)}",
                f"Color: {self._display(analysis.color_preference)}",
                f"Suggested artist: {analysis.suggested_artist}",
                f"Confidence: {analysis.confidence_level}",
                f"Missing information: {self._missing(analysis)}",
                f"AI reason: {self._display(analysis.ai_reasoning)}",
            ]
        )
        return "\n".join(lines)

    def combine_with_draft(self, summary: str, draft_reply: str) -> str:
        """Place the client draft after the staff summary for Telegram."""
        return f"{summary}\n\nDRAFT REPLY\n{draft_reply}"

    def _append_state_value(
        self,
        lines: list[str],
        label: str,
        state: dict[str, Any],
        key: str,
    ) -> None:
        """Append one known scalar database value without dumping state."""
        value = state.get(key)
        if isinstance(value, (str, int, float)) and str(value).strip():
            lines.append(f"{label}: {value}")

    def _current_message(self, inquiry: TattooInquiryInput) -> str:
        """Represent image-only messages clearly for studio staff."""
        if inquiry.current_message:
            return inquiry.current_message
        return "Image-only message"

    def _display(self, value: str) -> str:
        """Return a useful staff-facing value for empty fields."""
        normalized = value.strip()
        return normalized or "Not provided"

    def _style(self, analysis: AIExtractionOutput) -> str:
        """Render detected style tags for staff."""
        tags = [tag for tag in analysis.style_tags if tag != "unknown"]
        return ", ".join(tags) if tags else "Unknown"

    def _missing(self, analysis: AIExtractionOutput) -> str:
        """Render remaining missing information for staff."""
        if not analysis.missing_information:
            return "None"
        return ", ".join(analysis.missing_information)