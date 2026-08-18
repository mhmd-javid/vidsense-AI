"""In-memory NumPy cosine vector store.

The simplest thing that works at MVP scale (thousands of chunks): keep a
normalized matrix in RAM and do a single dense dot-product for search. No extra
services, no fragile native deps. It implements the same :class:`VectorStore`
interface as any future Chroma/FAISS backend.

Vectors are assumed (or forced) to be L2-normalized, so dot product == cosine.
"""
from __future__ import annotations

from typing import Any, Dict, List, Optional, Sequence

import numpy as np

from core.utils import get_logger
from modules.vectorstore.base import SearchHit, VectorItem, VectorStore

logger = get_logger(__name__)


def _normalize_rows(mat: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(mat, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    return mat / norms


class NumpyVectorStore(VectorStore):
    def __init__(self, normalize: bool = True):
        self._normalize = normalize
        self._ids: List[str] = []
        self._index: Dict[str, int] = {}
        self._matrix: Optional[np.ndarray] = None  # (N, dim), normalized
        self._metas: List[Dict[str, Any]] = []

    # ------------------------------------------------------------------ write
    def upsert(self, items: Sequence[VectorItem]) -> None:
        if not items:
            return
        for it in items:
            vec = np.asarray(it.vector, dtype=np.float32).reshape(1, -1)
            if self._normalize:
                vec = _normalize_rows(vec)
            if it.id in self._index:  # replace existing
                row = self._index[it.id]
                self._matrix[row] = vec[0]
                self._metas[row] = dict(it.metadata)
            else:
                self._index[it.id] = len(self._ids)
                self._ids.append(it.id)
                self._metas.append(dict(it.metadata))
                self._matrix = vec if self._matrix is None else np.vstack([self._matrix, vec])

    # ----------------------------------------------------------------- search
    def search(
        self,
        query: np.ndarray,
        top_k: int = 5,
        where: Optional[Dict[str, Any]] = None,
    ) -> List[SearchHit]:
        if self._matrix is None or len(self._ids) == 0:
            return []

        q = np.asarray(query, dtype=np.float32).reshape(-1)
        if self._normalize:
            n = np.linalg.norm(q)
            if n:
                q = q / n

        scores = self._matrix @ q  # cosine similarity, shape (N,)

        # Optional metadata equality filter (e.g. {"video_id": "abc"}).
        if where:
            mask = np.array(
                [all(m.get(k) == v for k, v in where.items()) for m in self._metas]
            )
            scores = np.where(mask, scores, -np.inf)

        k = min(top_k, int(np.sum(np.isfinite(scores))))
        if k <= 0:
            return []
        # argpartition for top-k, then sort those k by score desc.
        top_idx = np.argpartition(-scores, k - 1)[:k]
        top_idx = top_idx[np.argsort(-scores[top_idx])]

        return [
            SearchHit(id=self._ids[i], score=float(scores[i]), metadata=self._metas[i])
            for i in top_idx
            if np.isfinite(scores[i])
        ]

    # ------------------------------------------------------------------ admin
    def delete(self, ids: Sequence[str]) -> None:
        drop = set(ids)
        keep = [i for i, _id in enumerate(self._ids) if _id not in drop]
        if len(keep) == len(self._ids):
            return
        self._ids = [self._ids[i] for i in keep]
        self._metas = [self._metas[i] for i in keep]
        self._matrix = self._matrix[keep] if self._matrix is not None and keep else None
        self._index = {_id: i for i, _id in enumerate(self._ids)}

    def clear(self) -> None:
        self._ids, self._index, self._metas, self._matrix = [], {}, [], None

    def count(self) -> int:
        return len(self._ids)
