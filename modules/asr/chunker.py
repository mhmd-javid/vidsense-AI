"""Merge short raw ASR segments into larger, semantically-coherent chunks.

Whisper emits many tiny segments (a few seconds each). Those are poor units to
embed and retrieve. We merge consecutive segments into ~``target_seconds`` /
``max_chars`` windows, preferring to cut on sentence boundaries, while always
preserving the accurate ``start`` / ``end`` timestamps of the merged span.
"""
from __future__ import annotations

from dataclasses import dataclass, asdict
from typing import Iterable, List, Sequence

# Sentence-final punctuation, incl. Persian question mark (؟) and ellipsis.
_SENTENCE_ENDERS = tuple(".!?؟…\n")


@dataclass
class Chunk:
    index: int
    start: float
    end: float
    text: str

    def as_dict(self) -> dict:
        return asdict(self)


def _has_segment_fields(obj) -> bool:
    return hasattr(obj, "start") and hasattr(obj, "end") and hasattr(obj, "text")


def _normalize(segments: Iterable) -> List[tuple[float, float, str]]:
    out: List[tuple[float, float, str]] = []
    for s in segments:
        if _has_segment_fields(s):
            start, end, text = s.start, s.end, s.text
        else:  # dict-like
            start, end, text = s["start"], s["end"], s["text"]
        text = (text or "").strip()
        if text:
            out.append((float(start), float(end), text))
    return out


def chunk_segments(
    segments: Sequence,
    target_seconds: float = 30.0,
    max_seconds: float = 45.0,
    max_chars: int = 700,
    min_chars: int = 40,
) -> List[Chunk]:
    """Group ``segments`` into a list of :class:`Chunk`.

    A chunk is flushed when either:
      * it has reached ``target_seconds`` *and* the last segment ends a
        sentence (natural boundary), or
      * it hits the hard caps ``max_seconds`` / ``max_chars`` (forced boundary).
    A trailing chunk shorter than ``min_chars`` is merged back into the
    previous chunk when possible.
    """
    norm = _normalize(segments)
    if not norm:
        return []

    chunks: List[Chunk] = []
    cur_start = 0.0
    cur_end = 0.0
    cur_parts: List[str] = []
    cur_ends_sentence = False

    def cur_text() -> str:
        return " ".join(cur_parts).strip()

    def flush():
        nonlocal cur_parts
        text = cur_text()
        if text:
            chunks.append(Chunk(index=len(chunks), start=cur_start, end=cur_end, text=text))
        cur_parts = []

    for start, end, text in norm:
        # Decide whether to close the current chunk BEFORE adding this segment,
        # so a single long segment or a hard-cap overshoot splits cleanly.
        if cur_parts:
            new_dur = end - cur_start
            new_chars = len(cur_text()) + 1 + len(text)
            natural = (cur_end - cur_start) >= target_seconds and cur_ends_sentence
            if natural or new_dur > max_seconds or new_chars > max_chars:
                flush()

        if not cur_parts:
            cur_start = start
        cur_parts.append(text)
        cur_end = end
        cur_ends_sentence = text.endswith(_SENTENCE_ENDERS)

    if cur_parts:  # trailing remainder
        flush()

    # Mop up a too-short trailing fragment by merging it back into the previous
    # chunk — but only when that split was arbitrary (previous chunk did NOT end
    # on a sentence boundary) and the merge still fits the hard caps. Otherwise
    # respect the boundary and leave the short chunk standing.
    if len(chunks) >= 2 and len(chunks[-1].text) < min_chars:
        last = chunks[-1]
        prev = chunks[-2]
        prev_ended_sentence = prev.text.endswith(_SENTENCE_ENDERS)
        merged_dur = last.end - prev.start
        merged_chars = len(prev.text) + 1 + len(last.text)
        if not prev_ended_sentence and merged_dur <= max_seconds and merged_chars <= max_chars:
            chunks.pop()
            chunks[-1] = Chunk(
                index=prev.index,
                start=prev.start,
                end=last.end,
                text=f"{prev.text} {last.text}".strip(),
            )

    return chunks
