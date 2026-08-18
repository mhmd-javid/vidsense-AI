"""Factory for the configured vector-store backend.

Only the NumPy backend is implemented for the MVP. Adding Chroma/FAISS later
means implementing :class:`VectorStore` and adding a branch here — no changes
to the RAG code.
"""
from __future__ import annotations

from modules.vectorstore.base import VectorStore
from modules.vectorstore.numpy_store import NumpyVectorStore


def create_vector_store(backend: str = "numpy", normalize: bool = True) -> VectorStore:
    backend = (backend or "numpy").lower()
    if backend == "numpy":
        return NumpyVectorStore(normalize=normalize)
    raise ValueError(
        f"Unsupported vector store backend: '{backend}'. "
        "Only 'numpy' is implemented in this MVP."
    )
