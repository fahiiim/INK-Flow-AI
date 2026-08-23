"""Tests for FAISS-backed studio decision learning."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import Mock

import pytest
from langchain_core.embeddings import Embeddings

from ai_brain.decision import StudioDecisionEngine
from ai_brain.decision_schemas import (
    ArtistOption,
    DecisionHistoryExample,
    StudioDecisionContext,
)
from ai_brain.schemas import AIExtractionOutput, StyleTag
from ai_brain.vector_store import VectorStoreManager

_EMBEDDING_DIMENSION = 3


class DummyEmbeddings(Embeddings):
    """Return deterministic semantic vectors without calling OpenAI."""

    def __init__(self) -> None:
        self.query_calls = 0

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        """Embed indexed documents using their explicit style names."""
        return [self._vector_for_text(text) for text in texts]

    def embed_query(self, text: str) -> list[float]:
        """Embed one query and record that the provider was invoked."""
        self.query_calls += 1
        return self._vector_for_text(text)

    def _vector_for_text(self, text: str) -> list[float]:
        """Map supported test styles to orthogonal unit vectors."""
        normalized = text.casefold()
        if "fine-line" in normalized:
            return [1.0, 0.0, 0.0]
        if "watercolor" in normalized:
            return [0.0, 1.0, 0.0]
        if "traditional" in normalized:
            return [0.0, 0.0, 1.0]
        return [1.0, 1.0, 1.0]


def _history_example(
    example_id: str,
    style_tag: StyleTag,
    final_artist_key: str,
) -> DecisionHistoryExample:
    """Build one valid historical studio decision for vector tests."""
    return DecisionHistoryExample(
        example_id=example_id,
        channel="whatsapp",
        style_tags=[style_tag],
        placement="forearm",
        size_estimate_cm="8 cm",
        color_preference="black-and-grey",
        original_ai_artist_key=None,
        final_artist_key=final_artist_key,
        original_ai_action="artist_review",
        final_action="ready_to_book",
        ai_suggestion_outcome="corrected",
        correction_reason="Verified by studio staff.",
    )


def _analysis() -> AIExtractionOutput:
    """Build a current request that should match the fine-line example."""
    return AIExtractionOutput(
        tattoo_idea="Fine-line floral tattoo",
        style_tags=["fine-line", "floral"],
        placement="forearm",
        size_estimate_cm="8 cm",
        color_preference="black-and-grey",
        suggested_artist="Hoss",
        confidence_level="medium",
        ai_reasoning="Initial static routing result.",
        missing_information=[],
        risk_level="low",
        draft_reply="The studio will review your request.",
    )


def _context() -> StudioDecisionContext:
    """Build request-scoped artist options without static history."""
    return StudioDecisionContext(
        channel="whatsapp",
        artist_options=[
            ArtistOption(key="nina", display_name="Nina"),
            ArtistOption(key="hoss", display_name="Hoss"),
        ],
    )


def _records() -> list[DecisionHistoryExample]:
    """Return three stylistically distinct learning examples."""
    return [
        _history_example("fine-line-case", "fine-line", "nina"),
        _history_example("watercolor-case", "watercolor", "hoss"),
        _history_example("traditional-case", "traditional", "hoss"),
    ]


def _manager(embeddings: Embeddings) -> VectorStoreManager:
    """Build a small injected FAISS manager for deterministic tests."""
    return VectorStoreManager(
        embedding_model=embeddings,
        embedding_dimension=_EMBEDDING_DIMENSION,
    )


def test_vector_store_add_and_search() -> None:
    """The fine-line record is the closest semantic search result."""
    manager = _manager(DummyEmbeddings())
    manager.add_records(_records())

    results = manager.search_similar_cases(
        "fine-line floral tattoo",
        top_k=3,
    )

    assert len(results) == 3
    assert results[0][0] == pytest.approx(1.0)
    assert results[0][1].example_id == "fine-line-case"


def test_empty_vector_store_skips_query_embedding() -> None:
    """Searching an empty index returns before invoking the provider."""
    embeddings = DummyEmbeddings()
    manager = _manager(embeddings)

    assert manager.search_similar_cases("fine-line tattoo") == []
    assert embeddings.query_calls == 0


def test_vector_store_persistence_round_trip(tmp_path: Path) -> None:
    """A saved index restores both vectors and their Pydantic records."""
    index_path = tmp_path / "learning.faiss"
    manager = _manager(DummyEmbeddings())
    manager.add_records(_records())
    manager.save_index(str(index_path))

    restored = _manager(DummyEmbeddings())
    restored.load_index(str(index_path))
    results = restored.search_similar_cases("watercolor tattoo", top_k=1)

    assert results[0][1].example_id == "watercolor-case"
    assert len(restored.records) == 3


def test_decision_engine_uses_vector_search() -> None:
    """The engine uses retrieved feedback to override static artist routing."""
    fine_line_case = _records()[0]
    vector_store = Mock(spec=VectorStoreManager)
    vector_store.search_similar_cases.return_value = [
        (0.99, fine_line_case)
    ]
    engine = StudioDecisionEngine(vector_store=vector_store)

    result = engine.decide(
        analysis=_analysis(),
        current_message="I want a fine-line floral tattoo.",
        context=_context(),
    )

    vector_store.search_similar_cases.assert_called_once()
    query_text = vector_store.search_similar_cases.call_args.args[0]
    assert "fine-line" in query_text
    assert result.artist_suggestion.artist_key == "nina"
    assert result.artist_suggestion.source == "verified_history"
