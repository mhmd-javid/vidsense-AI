"""Confidence scoring for the final transcript.

Two signals are available from this pipeline (the batched WhisperX path does
**not** expose ``no_speech_prob``, so we don't invent it):

  * ``avg_logprob`` — whisper's mean token log-probability for the segment.
    ``exp(avg_logprob)`` maps it into ``(0, 1]`` as a probability-like score.
  * per-word alignment ``score`` — wav2vec2 forced-alignment confidence, when
    the alignment stage ran.

Per segment we take a weighted mean of whichever signals are present
(weights renormalized over the available ones; neutral ``0.5`` when none), flag
segments below ``low_threshold``, and roll up a duration-weighted
``quality_score`` for the whole transcript. Nothing here changes the text.
"""
from __future__ import annotations

import math
from typing import Any, Dict, List, Optional

from core.config import ConfidenceSection
from core.utils import get_logger

logger = get_logger(__name__)


def _clamp01(x: float) -> float:
    return max(0.0, min(1.0, x))


def _mean(values: List[float]) -> Optional[float]:
    vals = [v for v in values if v is not None]
    return sum(vals) / len(vals) if vals else None


def _word_score(seg: Dict[str, Any]) -> Optional[float]:
    scores = [w.get("score") for w in (seg.get("words") or []) if w.get("score") is not None]
    return _mean(scores) if scores else None


def _segment_confidence(seg: Dict[str, Any], cfg: ConfidenceSection) -> float:
    """Weighted mean of the available confidence signals, in [0, 1]."""
    signals: List[tuple[float, float]] = []  # (value, weight)

    avg_logprob = seg.get("avg_logprob")
    if avg_logprob is not None:
        signals.append((_clamp01(math.exp(avg_logprob)), cfg.weight_logprob))

    word_score = _word_score(seg)
    if word_score is not None:
        signals.append((_clamp01(word_score), cfg.weight_word_score))

    if not signals:
        return 0.5  # neutral — no signal to judge on

    total_weight = sum(w for _, w in signals) or 1.0
    return _clamp01(sum(v * w for v, w in signals) / total_weight)


def score_transcript(result: Dict[str, Any], cfg: ConfidenceSection) -> Dict[str, Any]:
    """Attach per-segment ``confidence``/``low_confidence`` and a ``quality_score``.

    Returns *result* (mutated in place).
    """
    segments = result.get("segments", [])
    weighted_sum = 0.0
    total_duration = 0.0
    low_count = 0

    for seg in segments:
        conf = round(_segment_confidence(seg, cfg), 4)
        seg["confidence"] = conf
        is_low = conf < cfg.low_threshold
        seg["low_confidence"] = is_low
        low_count += int(is_low)

        duration = max(0.0, float(seg.get("end", 0.0)) - float(seg.get("start", 0.0)))
        # Duration-weight so long segments dominate; fall back to equal weight.
        weight = duration if duration > 0 else 1.0
        weighted_sum += conf * weight
        total_duration += weight

    quality_score = round(weighted_sum / total_duration, 4) if total_duration else 0.0
    result["quality_score"] = quality_score
    result["low_confidence_segments"] = low_count
    logger.info(
        "Confidence scored: quality=%.3f, %d/%d low-confidence segments.",
        quality_score,
        low_count,
        len(segments),
    )
    return result
