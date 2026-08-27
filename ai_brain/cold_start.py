"""Cold-start safety controls for history-driven studio decisions."""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .vector_store import VectorStoreManager

LOGGER = logging.getLogger(__name__)

COLD_START_THRESHOLD = 10


class ColdStartManager:
    """Enforce manual review until enough verified cases are available."""

    def get_record_count(
        self,
        vector_store_manager: VectorStoreManager,
    ) -> int:
        """Return the verified record count, failing closed on store errors.

        A vector-store inspection failure is treated as zero records because
        allowing automatic routing without a trustworthy count is unsafe.
        """
        try:
            record_count = vector_store_manager.record_count
            if isinstance(record_count, bool) or not isinstance(
                record_count,
                int,
            ):
                raise TypeError("record_count must be an integer.")
            if record_count < 0:
                raise ValueError("record_count cannot be negative.")
            return record_count
        except Exception as exc:  # pragma: no cover - defensive branch
            try:
                return len(vector_store_manager.records)
            except Exception:
                LOGGER.warning(
                    "Unable to inspect vector-store records; assuming zero: %s",
                    exc,
                )
                return 0

    def check_cold_start(
        self,
        vector_store_manager: VectorStoreManager,
    ) -> bool:
        """Return whether the vector store has fewer than ten records."""
        return (
            self.get_record_count(vector_store_manager)
            < COLD_START_THRESHOLD
        )

    def get_remaining_records_needed(
        self,
        vector_store_manager: VectorStoreManager,
    ) -> int:
        """Return the additional verified records needed for automation."""
        record_count = self.get_record_count(vector_store_manager)
        return max(0, COLD_START_THRESHOLD - record_count)

    def build_status_message(self, collected_records: int) -> str:
        """Build the canonical backend-facing cold-start explanation."""
        safe_count = max(0, min(collected_records, COLD_START_THRESHOLD))
        return (
            "Cold start mode: manual assignment required "
            f"({safe_count}/{COLD_START_THRESHOLD} records)."
        )

    def log_cold_start_warning(self) -> None:
        """Log that all artist and price decisions require manual approval."""
        LOGGER.warning(
            "Cold start mode active: manual approval is required until "
            "%s verified records are collected.",
            COLD_START_THRESHOLD,
        )
