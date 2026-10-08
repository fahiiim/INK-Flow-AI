"""Track per-request model failures that make automated replies unsafe."""

from __future__ import annotations

from contextvars import ContextVar

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


def is_quota_exhausted_error(exc: BaseException) -> bool:
    """Detect OpenAI quota or billing errors, including wrapped causes."""
    seen: set[int] = set()
    current: BaseException | None = exc
    while current is not None and id(current) not in seen:
        seen.add(id(current))
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
        current = current.__cause__ or current.__context__
    return False
