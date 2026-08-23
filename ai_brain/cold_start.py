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

    def check_cold_start(
        self,
        vector_store_manager: VectorStoreManager,
    ) -> bool:
        """Return whether the vector store has fewer than ten records."""
        return len(vector_store_manager.records) < COLD_START_THRESHOLD

    def get_remaining_records_needed(
        self,
        vector_store_manager: VectorStoreManager,
    ) -> int:
        """Return the additional verified records needed for automation."""
        record_count = len(vector_store_manager.records)
        return max(0, COLD_START_THRESHOLD - record_count)

    def log_cold_start_warning(self) -> None:
        """Log that all artist and price decisions require manual approval."""
        LOGGER.warning(
            "Cold start mode active: manual approval is required until "
            "%s verified records are collected.",
            COLD_START_THRESHOLD,
        )
