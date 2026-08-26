"""Container health probe for the FastAPI readiness endpoint."""

from __future__ import annotations

from http import HTTPStatus
from urllib.error import URLError
from urllib.request import urlopen

_READINESS_URL = "http://127.0.0.1:8001/health/ready"
_TIMEOUT_SECONDS = 3


def main() -> None:
    """Exit successfully only when the service reports HTTP 200."""
    try:
        with urlopen(
            _READINESS_URL,
            timeout=_TIMEOUT_SECONDS,
        ) as response:
            if response.status != HTTPStatus.OK:
                raise SystemExit(1)
    except (TimeoutError, URLError) as exc:
        raise SystemExit(1) from exc


if __name__ == "__main__":
    main()

