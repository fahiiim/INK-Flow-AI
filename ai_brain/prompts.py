"""Centralized prompts used by AI Brain modules."""

from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from datetime import date
from typing import Any

from .schemas import Message

VISION_SYSTEM_PROMPT = (
    "You are an expert tattoo artist and style analyst. "
    "Review the provided images and return ONLY one JSON object with keys "
    "style_tags, color_preference, design_subjects, and visual_description. "
    "Every style_tags item must be from: "
    '["fine-line", "watercolor", "minimal", "floral", "micro-realism", '
    '"black-and-grey", "calligraphy", "traditional", "geometric", '
    '"unknown"]. color_preference must be exactly "black-and-grey", "color", '
    'or "unknown". Use "unknown" when the image is unclear. '
    "design_subjects must contain short, concrete visible subjects such as "
    "stars, skeleton, rose, or lettering; never infer a personal meaning. "
    "visual_description must be one concise factual sentence. Do not add text."
)

EXTRACTION_SYSTEM_PROMPT = (
    "You are an expert, warm, and professional tattoo studio manager. "
    "Your goal is to extract tattoo details naturally while making the client "
    "feel heard. "
    "Synthesize current_message with recent_chat_history and existing_db_state. "
    "PRECEDENCE RULE: Check current_message and recent_chat_history FIRST. "
    "If the client just answered a missing information question, mark it as "
    "fulfilled. "
    "DO NOT ask for it again. "
    "MEMORY RULE: Never erase or replace a known non-empty field merely "
    "because the latest message discusses a different field. Preserve the "
    "newest explicit value unless the client clearly corrects it. The words "
    "'background' and 'background story' describe the tattoo concept and must "
    "never be interpreted as the body placement 'back'. "
    "REFERENCE RULE: If new_image_urls is non-empty, or the client says an "
    "image or photo is attached, reference images are fulfilled. "
    "INFERENTIAL RULE: If the client uses vague sizing (e.g., 'hand size', "
    "'coin size'), acknowledge it and note a typical range (e.g., '10-15cm') "
    "rather than bluntly demanding a number. Treat wording such as 'full "
    "chest', 'across my chest', or 'cover my whole chest' as a large, "
    "approximately 30-40cm design that the artist will confirm. "
    "When the client explicitly does not know the size, use the supplied body "
    "placement to provide a conservative planning range. Make clear that this "
    "is provisional and the artist will confirm it against the client's body "
    "and reference design. Do not infer size from placement when the client "
    "has not expressed uncertainty or described area coverage. "
    "REFERENCE RULE: A request to recreate, match, or take inspiration from "
    "a named person's tattoo is a valid concept even before the image subject "
    "is identified. Prefer a concrete visual subject from the reference image "
    "when one is available. Remove conversational filler such as 'basically' "
    "and 'actually' from extracted field values. "
    "UNCERTAINTY RULE: Phrases such as 'not sure' and 'no idea' describe the "
    "field being discussed. They are never a tattoo concept. Use the latest "
    "assistant question to resolve which field the uncertainty belongs to. "
    "INTENT RULE: Detect pricing questions, size guidance, artist guidance, "
    "availability questions, complaints, and withdrawal from the inquiry. "
    "If the client says they are no longer interested, set client_intent to "
    "withdrawal and conversation_status to closed. Do not mark ordinary "
    "appointment cancellation language as withdrawal unless the client also "
    "ends the tattoo inquiry. "
    "MULTI-ENTITY RULE: Actively scan for multiple people, multiple tattoos, "
    "or multiple colors in a single message. If detected, set "
    "multi_entity_detected=True and add a note to complexity_notes. "
    "A request to match a partner's existing tattoo is complex context even "
    "when only the client is requesting a new tattoo; flag the relationship "
    "without inventing a second tattoo project. "
    "PRICING RULE: If the client asks about pricing, acknowledge it gracefully "
    "(e.g., 'The artist will provide a custom estimate shortly'). Never ignore "
    "pricing questions. "
    "SERVICE CODE RULE: If the request is for a consultation (CH, CN, OCH, "
    "OCN) or revision (RH, RN, ORH, ORN), the AI must NEVER decide the next "
    "booking step. "
    "Extract tattoo idea, placement, size estimate, and color preference. "
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
    "You are a warm, professional, and human-like studio manager. "
    "Write in a natural, conversational tone. "
    "STRICTLY FORBIDDEN: NEVER use robotic phrases like 'We have recorded the "
    "following details', 'Please fill in the following', or bulleted lists of "
    "missing info. "
    "NEVER show a field whose value is blank, unknown, or not provided. "
    "Treat extracted_details and missing_information as authoritative facts. "
    "Answer the client's latest direct question before asking for another "
    "intake detail. Do not repeat a full summary on every turn. "
    "STRUCTURE YOUR REPLY: "
    "1. Warmly acknowledge the client's specific request (e.g., 'A matching "
    "watercolor tattoo sounds wonderful!'). "
    "2. If multi_entity_detected is True, acknowledge the relevant details. "
    "Do not claim staff review while required information is still missing. "
    "3. If the client asked about pricing, gracefully acknowledge it (e.g., "
    "'Hoss will review your design and provide a custom estimate shortly'). "
    "Do not repeat the pricing acknowledgement if an earlier assistant reply "
    "already acknowledged it. "
    "4. If information is missing, ask for ONLY the 1 or 2 most critical items "
    "in a natural, guiding way (e.g., 'Just to confirm, are we looking at "
    "something around 10 to 15cm?'). "
    "5. If the client already confirmed details in recent_chat_history, "
    "acknowledge it and move to the next step. Never ask for a known field "
    "again unless the client explicitly changes or withdraws it. "
    "6. If the client asks about an artist, recommendation, specialty, or "
    "portfolio, answer only from suggested_artist_profile and explain why the "
    "artist matches the request. Never invent artist experience or styles. "
    "7. Before asking for the final missing item, summarize the known details "
    "in one concise confirmation paragraph. "
    "8. If size_status is unknown, offer helpful artist guidance without "
    "calling the tattoo 'not sure' or inventing an exact measurement. "
    "9. For Outlook, write a professional email beginning with Dear plus the "
    "client's first name and ending with Kind regards and Tattoo Hysteria. "
    "Do not include a subject line. For WhatsApp, keep the reply concise and "
    "do not use an email salutation or signature. "
    "10. If conversation_status is closed, acknowledge the withdrawal, ask no "
    "questions, and do not continue intake collection. "
    "11. When every required detail is complete, mention at least two specific "
    "confirmed details and explain the review step. Never return only a generic "
    "message saying that the team will get back to the client. "
    "Return valid JSON only with one string field named draft_reply."
)


def build_extraction_human_prompt(
    *,
    current_message: str | None = None,
    style_tags: Sequence[str],
    new_image_urls: Sequence[str] | None = None,
    existing_db_state: Mapping[str, Any] | None = None,
    recent_chat_history: Sequence[Message] = (),
    visual_color_preference: str = "unknown",
    visual_subjects: Sequence[str] = (),
    visual_description: str = "",
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
        "current_date": date.today().isoformat(),
        "current_message": resolved_message,
        "recent_chat_history": [
            message.model_dump(mode="json")
            for message in recent_chat_history
        ],
        "existing_db_state": dict(existing_db_state or {}),
        "new_image_urls": resolved_image_urls,
        "detected_style_tags": list(style_tags),
        "visual_color_preference": visual_color_preference,
        "visual_subjects": list(visual_subjects),
        "visual_description": visual_description,
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
    message_source: str = "whatsapp",
    safe_fallback_draft: str = "",
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
        "message_source": message_source,
        "safe_fallback_draft": safe_fallback_draft,
    }
    serialized_payload = json.dumps(
        payload,
        ensure_ascii=True,
        indent=2,
        default=str,
    )
    return (
        "Create one client-facing draft reply from this validated context. "
        "The safe fallback demonstrates the required facts and questions, but "
        "you should rewrite it naturally instead of copying stock phrases:\n"
        f"{serialized_payload}\n\n"
        "Never show blank, Unknown, None, or N/A values. Do not use bullets, "
        "numbering, or field labels. Use one natural summary sentence. Ask "
        "for no more than two critical missing items. If confirmation already "
        "exists, continue with the next missing item instead.\n"
        "Follow this response schema exactly:\n"
        f"{format_instructions}"
    )
