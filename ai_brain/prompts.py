"""Centralized prompts used by AI Brain modules."""

from __future__ import annotations

import json
from collections.abc import Mapping, Sequence

VISION_SYSTEM_PROMPT = (
    "You are an expert tattoo artist and style analyst. "
    "Review the provided images and return ONLY a JSON array. "
    "Every array item must be from this exact list: "
    '["fine-line", "watercolor", "minimal", "floral", "micro-realism", '
    '"black-and-grey", "calligraphy", "traditional", "geometric", '
    '"unknown"]. Do not add extra text.'
)

EXTRACTION_SYSTEM_PROMPT = (
    "You are a tattoo studio intake analyst. "
    "Analyze client_text together with style_tags from vision analysis. "
    "Extract tattoo idea, placement, size estimate in cm, and color preference. "
    "Return strictly valid JSON and do not include markdown or extra text."
)

ROUTING_SYSTEM_PROMPT = (
    "You are a tattoo studio operations assistant. "
    "You will receive a rule-based artist suggestion and risk level. "
    "Do not change those values. Return valid JSON only with "
    "confidence_level, ai_reasoning, and draft_reply."
)


def build_extraction_human_prompt(
    client_text: str,
    style_tags: Sequence[str],
    image_urls: Sequence[str],
    required_items: Sequence[str],
    format_instructions: str,
) -> str:
    """Build the dynamic extraction prompt from validated input values."""
    return (
        "Client text:\n"
        f"{client_text}\n\n"
        "Detected style tags:\n"
        f"{json.dumps(list(style_tags))}\n\n"
        "Reference image URLs provided:\n"
        f"{json.dumps(list(image_urls))}\n\n"
        "Required missing-information checklist:\n"
        f"{json.dumps(list(required_items))}\n\n"
        "Identify missing_information using checklist values exactly.\n"
        "Return JSON only and follow this schema:\n"
        f"{format_instructions}"
    )


def build_routing_human_prompt(
    extracted_data: Mapping[str, object],
    suggested_artist: str,
    risk_level: str,
    format_instructions: str,
) -> str:
    """Build the routing prompt using deterministic routing decisions."""
    serialized_data = json.dumps(extracted_data, ensure_ascii=True)
    return (
        "Use these fixed values exactly:\n"
        f"suggested_artist: {suggested_artist}\n"
        f"risk_level: {risk_level}\n\n"
        "Extracted data:\n"
        f"{serialized_data}\n\n"
        "Return valid JSON only with fields:\n"
        "confidence_level, ai_reasoning, draft_reply.\n"
        "confidence_level must be one of: high, medium, low.\n"
        "Keep ai_reasoning concise and operational.\n"
        "Draft reply must be polite and actionable.\n\n"
        f"{format_instructions}"
    )
