"""Defensive cleanup for client-authored email reply text."""

from __future__ import annotations

import html
import re

_HTML_QUOTE_MARKERS = (
    re.compile(r"<blockquote\b", flags=re.IGNORECASE),
    re.compile(
        r"<(?:div|table)\b[^>]*(?:class|id)\s*=\s*['\"][^'\"]*"
        r"(?:gmail_quote|yahoo_quoted|divRplyFwdMsg|appendonsend)"
        r"[^'\"]*['\"]",
        flags=re.IGNORECASE,
    ),
)
_PLAIN_TEXT_QUOTE_MARKERS = (
    re.compile(
        r"^[ \t]*On[ \t]+.{1,500}?\bwrote:[ \t]*$",
        flags=re.IGNORECASE | re.MULTILINE | re.DOTALL,
    ),
    re.compile(
        r"(?im)^[ \t]*(?:From|Sent):[ \t]+.+$",
    ),
    re.compile(
        r"(?im)^[ \t]*-{2,}[ \t]*Original Message[ \t]*-{2,}[ \t]*$",
    ),
    re.compile(r"(?m)^[ \t]*>"),
)
_HTML_LINE_BREAKS = re.compile(
    r"(?i)<br\s*/?>|</(?:p|div|li|tr|h[1-6])\s*>",
)
_HTML_TAG = re.compile(r"(?s)<[^>]+>")


def strip_quoted_email_content(value: str) -> str:
    """Return only the newest client-authored portion of an email body."""
    text = value.replace("\r\n", "\n").replace("\r", "\n")
    cutoff = len(text)
    for marker in (*_HTML_QUOTE_MARKERS, *_PLAIN_TEXT_QUOTE_MARKERS):
        match = marker.search(text)
        if match is not None:
            cutoff = min(cutoff, match.start())

    newest_reply = text[:cutoff]
    newest_reply = _HTML_LINE_BREAKS.sub("\n", newest_reply)
    newest_reply = _HTML_TAG.sub("", newest_reply)
    newest_reply = html.unescape(newest_reply)
    lines = [line.rstrip() for line in newest_reply.split("\n")]
    return "\n".join(lines).strip()
