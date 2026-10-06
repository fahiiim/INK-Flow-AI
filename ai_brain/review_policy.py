"""Deterministic policies for review-sensitive tattoo conversations."""

from __future__ import annotations

import re

SPECIALISED_PLACEMENT_REASON = "specialised_placement_requires_approval"
STATUS_UPDATE_REASON = "client_status_update_required"

_SPECIALISED_PLACEMENT_PATTERN = re.compile(
    r"\b(?:tongue|oral cavity|inside (?:the )?mouth)\b",
    flags=re.IGNORECASE,
)
_STATUS_UPDATE_PATTERN = re.compile(
    r"\bis\s+there\s+(?:any|an)\s+updates?\b|"
    r"\b(?:what|where)\s+is\s+(?:(?:the|my|your)\s+)?update\b|"
    r"\bwhat(?:'s|\s+is)\s+(?:the\s+)?status\b|"
    r"\bstatus\s+(?:update|of\s+my\s+request)\b|"
    r"\b(?:any|an)\s+updates?\s*(?:please)?[?.!\s]*$|"
    r"\b(?:have|did)\s+you\s+(?:heard|reviewed|checked)\b|"
    r"\bwhen\s+will\s+(?:i|you|the\s+team)\b",
    flags=re.IGNORECASE,
)
_PROCEED_PATTERN = re.compile(
    r"^(?:okay|ok|yes|sure|alright|fine)?[,.!\s]*"
    r"(?:then\s+)?(?:please\s+)?(?:proceed|go\s+ahead|move\s+forward)"
    r"[.!\s]*$",
    flags=re.IGNORECASE,
)
_REFERENCE_LED_STYLE_PATTERN = re.compile(
    r"\b(?:style|look|design)\b.{0,45}\b(?:same|similar|like|match(?:ing)?)\b"
    r".{0,30}\b(?:image|photo|picture|reference)\b|"
    r"\b(?:same|similar|like|match(?:ing)?)\b.{0,45}"
    r"\b(?:image|photo|picture|reference)\b",
    flags=re.IGNORECASE,
)


def specialised_placement_requires_review(placement: str) -> bool:
    """Return whether the requested placement needs studio approval first."""
    return bool(_SPECIALISED_PLACEMENT_PATTERN.search(placement))


def is_status_update_request(message: str) -> bool:
    """Return whether a client is asking for a real workflow update."""
    return bool(_STATUS_UPDATE_PATTERN.search(message))


def is_proceed_confirmation(message: str) -> bool:
    """Return whether a client explicitly asks the studio to proceed."""
    return bool(_PROCEED_PATTERN.fullmatch(" ".join(message.split())))


def is_reference_led_style_request(message: str) -> bool:
    """Return whether style is delegated to an attached reference image."""
    return bool(_REFERENCE_LED_STYLE_PATTERN.search(message))
