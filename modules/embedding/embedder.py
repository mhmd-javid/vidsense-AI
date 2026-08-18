"""Multilingual sentence embeddings (default: intfloat/multilingual-e5-small).

Runs on CPU by default so it never competes with ASR / the LLM for the 4 GB of
VRAM. e5 models require instruction prefixes ("query:" / "passage:"); we add
them automatically. Vectors are L2-normalized so cosine similarity == dot
product downstream.
"""
from __future__ import annotations

import gc
from pathlib import Path
from typing import List, Optional, Sequence

import numpy as np

from core.utils import get_logger

logger = get_logger(__name__)


class Embedder:
    def __init__(
        self,
        model_name: str = "intfloat/multilingual-e5-small",
        device: str = "cpu",
        batch_size: int = 16,
        normalize: bool = True,
        query_prefix: str = "query: ",
        passage_prefix: str = "passage: ",
        cache_folder: Optional[str] = None,
    ):
        self.model_name = model_name
        self.device = device
        self.batch_size = batch_size
        self.normalize = normalize
        self.query_prefix = query_prefix
        self.passage_prefix = passage_prefix
        self.cache_folder = cache_folder
        self._model = None
        self._dim: Optional[int] = None

    # ------------------------------------------------------------- lifecycle
    def load(self) -> None:
        if self._model is not None:
            return
        from sentence_transformers import SentenceTransformer

        logger.info("Loading embedding model '%s' on %s…", self.model_name, self.device)
        self._model = SentenceTransformer(
            self.model_name, device=self.device, cache_folder=self.cache_folder
        )
        self._dim = int(self._model.get_sentence_embedding_dimension())

    def unload(self) -> None:
        if self._model is not None:
            logger.info("Unloading embedding model.")
        self._model = None
        gc.collect()
        try:  # free GPU memory if we happened to be on CUDA
            import torch

            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except Exception:
            pass

    def __enter__(self) -> "Embedder":
        self.load()
        return self

    def __exit__(self, *exc) -> None:
        self.unload()

    @property
    def dim(self) -> int:
        if self._dim is None:
            self.load()
        return self._dim  # type: ignore[return-value]

    # --------------------------------------------------------------- encode
    def _encode(self, texts: Sequence[str]) -> np.ndarray:
        self.load()
        vecs = self._model.encode(
            list(texts),
            batch_size=self.batch_size,
            normalize_embeddings=self.normalize,
            convert_to_numpy=True,
            show_progress_bar=False,
        )
        return np.asarray(vecs, dtype=np.float32)

    def embed_passages(self, texts: Sequence[str]) -> np.ndarray:
        """Embed documents/chunks (adds the passage prefix)."""
        if not texts:
            return np.zeros((0, self.dim), dtype=np.float32)
        prefixed = [f"{self.passage_prefix}{t}" for t in texts]
        return self._encode(prefixed)

    def embed_query(self, text: str) -> np.ndarray:
        """Embed a single search query (adds the query prefix). Returns (dim,)."""
        return self._encode([f"{self.query_prefix}{text}"])[0]
