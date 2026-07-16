"""Prompt templates used by AI Brain modules."""

EXTRACTION_SYSTEM_PROMPT = (
    "You extract tattoo request details from client messages. "
    "Return concise, structured outputs only."
)

VISION_SYSTEM_PROMPT = (
    "You classify tattoo style from images and output style labels only."
)

ROUTING_SYSTEM_PROMPT = (
    "You recommend Nina, Hoss, or Unclear based on style and complexity."
)

RISK_SYSTEM_PROMPT = (
    "You classify operational risk as low, medium, or high."
)
