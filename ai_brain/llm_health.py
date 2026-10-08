"""Track per-request model failures that make automated replies unsafe."""

from __future__ import annotations

from contextvars import ContextVar
from datetime import datetime, timezone

_QUOTA_ERROR_CODES = frozenset(
    {
        "insufficient_quota",
        "credit_balance_exhausted",
        "billing_hard_limit_reached",
        "billing_not_active",
    }
)
_QUOTA_ERROR_PHRASES = (
    "insufficient_quota",
    "credit_balance_exhausted",
    "no credits remaining",
    "exceeded your current quota",
    "billing_hard_limit_reached",
)

_quota_exhausted: ContextVar[bool] = ContextVar(
    "llm_quota_exhausted",
    default=False,
)


def reset_llm_health() -> None:
    """Start a new request with no recorded model failures."""
    _quota_exhausted.set(False)


def record_llm_failure(exc: BaseException) -> None:
    """Remember a provider quota failure raised anywhere in the pipeline."""
    if is_quota_exhausted_error(exc):
        _quota_exhausted.set(True)


def llm_quota_exhausted() -> bool:
    """Return whether any model call in this request hit an exhausted quota."""
    return _quota_exhausted.get()


def llm_failure_context(exc: BaseException) -> str:
    """Describe a model failure with OpenAI's request ID and the UTC time."""
    request_id = ""
    status_code: object = None
    for error in _error_chain(exc):
        request_id = request_id or str(getattr(error, "request_id", "") or "")
        if not request_id:
            response = getattr(error, "response", None)
            headers = getattr(response, "headers", None)
            if headers is not None:
                request_id = str(headers.get("x-request-id", "") or "")
        status_code = status_code or getattr(error, "status_code", None)
    utc_time = datetime.now(timezone.utc).isoformat(timespec="seconds")
    parts = [
        f"request_id={request_id or 'unavailable'}",
        f"utc_time={utc_time}",
    ]
    if status_code:
        parts.append(f"status={status_code}")
    return " ".join(parts)


def _error_chain(exc: BaseException) -> list[BaseException]:
    """Return an exception followed by its causes, without cycles."""
    chain: list[BaseException] = []
    current: BaseException | None = exc
    while current is not None and all(current is not seen for seen in chain):
        chain.append(current)
        current = current.__cause__ or current.__context__
    return chain


def is_quota_exhausted_error(exc: BaseException) -> bool:
    """Detect OpenAI quota or billing errors, including wrapped causes."""
    for current in _error_chain(exc):
        code = getattr(current, "code", None)
        if isinstance(code, str) and code in _QUOTA_ERROR_CODES:
            return True
        body = getattr(current, "body", None)
        if isinstance(body, dict):
            error = body.get("error", body)
            if isinstance(error, dict) and (
                error.get("code") in _QUOTA_ERROR_CODES
                or error.get("type") in _QUOTA_ERROR_CODES
            ):
                return True
        message = str(current).casefold()
        if any(phrase in message for phrase in _QUOTA_ERROR_PHRASES):
            return True
    return False
