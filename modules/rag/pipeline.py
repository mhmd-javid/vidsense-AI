"""Retrieval-Augmented Generation over a single video's transcript chunks.

Flow:  question -> embed -> cosine search over the video's chunks ->
build a grounded prompt (answer ONLY from context, cite timestamps) ->
local LLM -> answer + structured timestamp citations.

The LLM is instructed to ground its answer in the retrieved excerpts and to
cite timestamp ranges. Independently, we always return the retrieved chunks
(with their timestamps) so the UI can show references even if the model's
inline citation is imperfect.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Dict, List, Optional

import numpy as np

from core.utils import format_range, get_logger
from modules.embedding.embedder import Embedder
from modules.llm.ollama_client import OllamaClient
from modules.storage.db import Database
from modules.vectorstore.base import SearchHit, VectorItem, VectorStore
from modules.vectorstore.factory import create_vector_store

logger = get_logger(__name__)


SYSTEM_PROMPT = (
    "You are a helpful assistant that answers questions about a single video, "
    "using ONLY the transcript excerpts provided by the user. Each excerpt is "
    "prefixed with its timestamp range in [MM:SS-MM:SS] form.\n\n"
    "Rules:\n"
    "1. Answer strictly from the excerpts. Do NOT use outside knowledge.\n"
    "2. If the excerpts do not contain the answer, say so honestly.\n"
    "3. ALWAYS cite the timestamp range(s) you relied on, e.g. (02:10-02:35).\n"
    "4. Reply in the SAME language as the question (e.g. Persian question -> "
    "Persian answer).\n"
    "5. Be concise and factual."
)


@dataclass
class Citation:
    start: float
    end: float
    text: str
    score: float

    @property
    def label(self) -> str:
        return format_range(self.start, self.end)

    def as_dict(self) -> dict:
        return {
            "start": self.start,
            "end": self.end,
            "label": self.label,
            "text": self.text,
            "score": self.score,
        }


@dataclass
class AnswerResult:
    question: str
    answer: str
    citations: List[Citation] = field(default_factory=list)
    grounded: bool = True

    def as_dict(self) -> dict:
        return {
            "question": self.question,
            "answer": self.answer,
            "grounded": self.grounded,
            "citations": [c.as_dict() for c in self.citations],
        }


class RAGPipeline:
    def __init__(
        self,
        db: Database,
        embedder: Embedder,
        llm: OllamaClient,
        backend: str = "numpy",
        top_k: int = 5,
        min_score: float = 0.10,
    ):
        self.db = db
        self.embedder = embedder
        self.llm = llm
        self.backend = backend
        self.top_k = top_k
        self.min_score = min_score
        self._stores: Dict[str, VectorStore] = {}  # cache per video

    # --------------------------------------------------------- vector index
    def _get_store(self, video_id: str) -> VectorStore:
        if video_id in self._stores:
            return self._stores[video_id]
        store = create_vector_store(self.backend, normalize=True)
        rows = self.db.get_chunks(video_id, with_embeddings=True)
        items = [
            VectorItem(
                id=str(r.id),
                vector=r.embedding,
                metadata={
                    "video_id": video_id,
                    "start": r.start,
                    "end": r.end,
                    "text": r.text,
                    "chunk_index": r.chunk_index,
                },
            )
            for r in rows
            if r.embedding is not None
        ]
        store.upsert(items)
        self._stores[video_id] = store
        logger.info("Built vector index for %s (%d chunks).", video_id, store.count())
        return store

    def invalidate(self, video_id: Optional[str] = None) -> None:
        """Drop cached index (call after (re)processing a video)."""
        if video_id is None:
            self._stores.clear()
        else:
            self._stores.pop(video_id, None)

    # ----------------------------------------------------------- retrieval
    def retrieve(self, question: str, video_id: str) -> List[Citation]:
        store = self._get_store(video_id)
        if store.count() == 0:
            return []
        qv = self.embedder.embed_query(question)
        hits: List[SearchHit] = store.search(qv, top_k=self.top_k, where={"video_id": video_id})

        strong = [h for h in hits if h.score >= self.min_score]
        chosen = strong if strong else hits[:1]  # always give the model something
        return [
            Citation(
                start=h.metadata["start"],
                end=h.metadata["end"],
                text=h.metadata["text"],
                score=h.score,
            )
            for h in chosen
        ]

    # -------------------------------------------------------------- answer
    def _build_context(self, citations: List[Citation]) -> str:
        return "\n\n".join(f"[{c.label}] {c.text}" for c in citations)

    def answer(self, question: str, video_id: str) -> AnswerResult:
        question = (question or "").strip()
        if not question:
            return AnswerResult(question, "Please enter a question.", grounded=False)

        citations = self.retrieve(question, video_id)
        if not citations:
            return AnswerResult(
                question,
                "No transcript content is available for this video yet, or nothing "
                "relevant was found. / محتوایی برای پاسخ یافت نشد.",
                grounded=False,
            )

        context = self._build_context(citations)
        user_msg = (
            f"Transcript excerpts:\n\n{context}\n\n"
            f"Question: {question}\n\n"
            "Answer using only the excerpts above and cite the timestamp range(s)."
        )
        answer_text = self.llm.chat(
            [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": user_msg},
            ]
        )
        return AnswerResult(question, answer_text, citations=citations, grounded=True)
