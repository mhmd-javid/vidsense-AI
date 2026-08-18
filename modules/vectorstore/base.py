"""Vector store interface.

The RAG layer only ever talks to this abstract interface, so the NumPy backend
below can be swapped for Chroma / FAISS / pgvector later without touching
retrieval code. Keep this minimal and backend-agnostic.
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence

import numpy as np


@dataclass
class VectorItem:
    id: str
    vector: np.ndarray
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class SearchHit:
    id: str
    score: float
    metadata: Dict[str, Any] = field(default_factory=dict)


class VectorStore(ABC):
    """Minimal cosine-similarity vector index."""

    @abstractmethod
    def upsert(self, items: Sequence[VectorItem]) -> None:
        ...

    @abstractmethod
    def search(
        self,
        query: np.ndarray,
        top_k: int = 5,
        where: Optional[Dict[str, Any]] = None,
    ) -> List[SearchHit]:
        ...

    @abstractmethod
    def delete(self, ids: Sequence[str]) -> None:
        ...

    @abstractmethod
    def clear(self) -> None:
        ...

    @abstractmethod
    def count(self) -> int:
        ...
