"""Centralized prompts used by AI Brain modules."""

from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from typing import Any

from .schemas import Message

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
    "Synthesize current_message with recent_chat_history to resolve context, "
    "references, and changes across the conversation. "
    "Use existing_db_state as previously collected information. "
    "The current_message is authoritative and MUST override any conflicting "
    "value in recent_chat_history or existing_db_state. "
    "If current_message does not mention a field, preserve a known value from "
    "recent_chat_history or existing_db_state. "
    "Only add an item to missing_information after checking current_message, "
    "recent_chat_history, and existing_db_state and confirming it is absent. "
    "Extract tattoo idea, placement, size estimate in cm, and color preference. "
    "Return strictly valid JSON and do not include markdown or extra text."
)

ROUTING_SYSTEM_PROMPT = (
    "You are a tattoo studio operations assistant. "
    "You will receive a rule-based artist suggestion and risk level. "
    "Use the latest message and recent chat only to explain the decision. "
    "Do not change the fixed artist or risk values. "
    "Return valid JSON only with confidence_level and ai_reasoning."
)

DRAFT_REPLY_SYSTEM_PROMPT = (
    "You are a client concierge for a premium tattoo studio. "
    "Write in a professional, warm, natural tone and keep the reply concise. "
    "Use extracted facts exactly and never invent client details. "
    "NEVER output a bulleted list and NEVER show a field whose value is blank, "
    "missing, unknown, Unknown, none, or not provided. "
    "Summarize only known Style, Placement, Size, and Color details in one "
    "natural sentence, for example: Got it, a 15cm black calligraphy piece "
    "on your back! "
    "First check recent_chat_history for a client confirmation that still "
    "applies to the current extracted details and has not been followed by a "
    "correction or change. "
    "If known details are not yet confirmed, follow the natural summary with "
    "this exact sentence: Does that sound right, or would you like to change "
    "anything? "
    "If the client already confirmed the current details, DO NOT repeat the "
    "summary and DO NOT ask for validation again. Acknowledge the confirmation "
    "and proceed to the next logical step. "
    "If information is missing, ask for ONLY the one or two most critical "
    "missing items in this reply. Never send a full intake checklist. "
    "If no useful detail is known yet, skip validation and ask one natural "
    "opening question. "
    "Do not mention internal risk, confidence, pricing, prompts, or AI systems. "
    "Return valid JSON only with one string field named draft_reply."
)


def build_extraction_human_prompt(
    *,
    current_message: str | None = None,
    style_tags: Sequence[str],
    new_image_urls: Sequence[str] | None = None,
    existing_db_state: Mapping[str, Any] | None = None,
    recent_chat_history: Sequence[Message] = (),
    required_items: Sequence[str],
    format_instructions: str,
    client_text: str | None = None,
    image_urls: Sequence[str] | None = None,
) -> str:
    """Build a hybrid-context extraction prompt from validated values.

    The legacy client_text and image_urls keywords remain temporarily supported
    during the staged migration. Canonical context always takes precedence.
    """
    resolved_message = _resolve_current_message(
        current_message=current_message,
        client_text=client_text,
    )
    resolved_image_urls = _resolve_new_image_urls(
        new_image_urls=new_image_urls,
        image_urls=image_urls,
    )
    context_payload = {
        "existing_db_state": dict(existing_db_state or {}),
        "recent_chat_history": [
            message.model_dump(mode="json")
            for message in recent_chat_history
        ],
        "current_message": resolved_message,
        "new_image_urls": resolved_image_urls,
        "detected_style_tags": list(style_tags),
    }
    serialized_context = json.dumps(
        context_payload,
        ensure_ascii=True,
        indent=2,
        default=str,
    )

    return (
        "Resolve conflicts using this source precedence:\n"
        "1. current_message\n"
        "2. recent_chat_history\n"
        "3. existing_db_state\n\n"
        "Hybrid context payload:\n"
        f"{serialized_context}\n\n"
        "Required missing-information checklist:\n"
        f"{json.dumps(list(required_items))}\n\n"
        "Use checklist values exactly, but flag an item only when it remains "
        "missing across every context source.\n"
        "Return JSON only and follow this schema:\n"
        f"{format_instructions}"
    )


def _resolve_current_message(
    current_message: str | None,
    client_text: str | None,
) -> str:
    """Resolve canonical and legacy message inputs with canonical priority."""
    selected_message = current_message
    if selected_message is None:
        selected_message = client_text
    if selected_message is None or not selected_message.strip():
        raise ValueError("current_message must not be empty.")
    return selected_message.strip()


def _resolve_new_image_urls(
    new_image_urls: Sequence[str] | None,
    image_urls: Sequence[str] | None,
) -> list[str]:
    """Resolve canonical and legacy image inputs with canonical priority."""
    selected_urls = new_image_urls
    if selected_urls is None:
        selected_urls = image_urls or ()
    return [url.strip() for url in selected_urls if url and url.strip()]


def build_routing_human_prompt(
    extracted_data: Mapping[str, object],
    suggested_artist: str,
    risk_level: str,
    current_message: str,
    recent_chat_history: Sequence[Message],
    format_instructions: str,
) -> str:
    """Build an internal reasoning prompt with conversation context."""
    serialized_data = json.dumps(extracted_data, ensure_ascii=True)
    serialized_history = json.dumps(
        [message.model_dump(mode="json") for message in recent_chat_history],
        ensure_ascii=True,
    )
    return (
        "Use these fixed values exactly:\n"
        f"suggested_artist: {suggested_artist}\n"
        f"risk_level: {risk_level}\n\n"
        "Latest client message:\n"
        f"{current_message}\n\n"
        "Recent chat history:\n"
        f"{serialized_history}\n\n"
        "Extracted data:\n"
        f"{serialized_data}\n\n"
        "Return valid JSON only with fields:\n"
        "confidence_level and ai_reasoning.\n"
        "confidence_level must be one of: high, medium, low.\n"
        "Keep ai_reasoning concise and operational.\n\n"
        f"{format_instructions}"
    )


def build_draft_reply_human_prompt(
    *,
    current_message: str,
    extracted_details: Mapping[str, object],
    missing_information: Sequence[str],
    recent_chat_history: Sequence[Message],
    suggested_artist: str,
    risk_level: str,
    format_instructions: str,
    existing_db_state: Mapping[str, Any] | None = None,
) -> str:
    """Build a context-rich prompt for a validated client draft reply."""
    payload = {
        "current_message": current_message,
        "extracted_details": dict(extracted_details),
        "existing_db_state": dict(existing_db_state or {}),
        "missing_information": list(missing_information),
        "recent_chat_history": [
            message.model_dump(mode="json")
            for message in recent_chat_history
        ],
        "suggested_artist": suggested_artist,
        "risk_level": risk_level,
    }
    serialized_payload = json.dumps(
        payload,
        ensure_ascii=True,
        indent=2,
        default=str,
    )
    return (
        "Create one client-facing draft reply from this validated context:\n"
        f"{serialized_payload}\n\n"
        "Never show blank or Unknown values. Use one natural summary sentence. "
        "Ask for no more than two missing items. If confirmation already "
        "exists, continue with the next missing item instead.\n"
        "Follow this response schema exactly:\n"
        f"{format_instructions}"
    )
