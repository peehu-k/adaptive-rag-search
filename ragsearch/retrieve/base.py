"""Shared retrieval types."""

from __future__ import annotations

from typing import NamedTuple, Protocol, runtime_checkable


class Hit(NamedTuple):
    doc_id: str
    score: float


@runtime_checkable
class Retriever(Protocol):
    def search(self, query: str, k: int = 10) -> list[Hit]:
        """Return up to ``k`` hits, highest score first."""
        ...
