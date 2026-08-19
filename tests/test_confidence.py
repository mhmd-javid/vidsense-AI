"""Unit tests for confidence scoring (math + low-confidence flagging).

The batched WhisperX path exposes ``avg_logprob`` and (after alignment) per-word
``score``; the scorer blends whichever are present. Nothing here alters text.
"""
import math
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from core.config import ConfidenceSection  # noqa: E402
from modules.confidence.scorer import score_transcript  # noqa: E402


def test_logprob_only_confidence():
    cfg = ConfidenceSection()
    result = {"segments": [{"start": 0.0, "end": 10.0, "avg_logprob": 0.0}]}  # exp(0)=1.0
    score_transcript(result, cfg)
    seg = result["segments"][0]
    assert abs(seg["confidence"] - 1.0) < 1e-6
    assert seg["low_confidence"] is False


def test_word_score_only_confidence():
    cfg = ConfidenceSection()
    result = {"segments": [{
        "start": 0.0, "end": 1.0,
        "words": [{"word": "a", "score": 0.8}, {"word": "b", "score": 1.0}],
    }]}
    score_transcript(result, cfg)
    assert abs(result["segments"][0]["confidence"] - 0.9) < 1e-6  # mean(0.8,1.0)


def test_blended_confidence_weights():
    cfg = ConfidenceSection()  # weights 0.6 logprob / 0.4 word_score
    result = {"segments": [{
        "start": 0.0, "end": 1.0,
        "avg_logprob": math.log(0.5),                       # exp -> 0.5
        "words": [{"word": "a", "score": 1.0}],             # word score 1.0
    }]}
    score_transcript(result, cfg)
    expected = (0.5 * 0.6 + 1.0 * 0.4) / (0.6 + 0.4)        # = 0.7
    assert abs(result["segments"][0]["confidence"] - expected) < 1e-6


def test_neutral_when_no_signals():
    cfg = ConfidenceSection()
    result = {"segments": [{"start": 0.0, "end": 1.0, "text": "بدون سیگنال"}]}
    score_transcript(result, cfg)
    assert result["segments"][0]["confidence"] == 0.5


def test_low_confidence_flag_and_quality_rollup():
    cfg = ConfidenceSection()  # low_threshold 0.5
    result = {"segments": [
        {"start": 0.0, "end": 10.0, "avg_logprob": 0.0},              # conf 1.0
        {"start": 10.0, "end": 20.0, "avg_logprob": math.log(0.2)},   # conf 0.2 -> low
    ]}
    score_transcript(result, cfg)
    s1, s2 = result["segments"]
    assert s1["low_confidence"] is False
    assert s2["low_confidence"] is True
    assert result["low_confidence_segments"] == 1
    # Duration-weighted mean of 1.0 and 0.2 over equal 10s spans.
    assert abs(result["quality_score"] - 0.6) < 1e-6


def test_empty_transcript_scores_zero():
    cfg = ConfidenceSection()
    result = {"segments": []}
    score_transcript(result, cfg)
    assert result["quality_score"] == 0.0
    assert result["low_confidence_segments"] == 0


if __name__ == "__main__":
    import traceback
    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    ok = 0
    for fn in fns:
        try:
            fn(); print(f"PASS {fn.__name__}"); ok += 1
        except Exception:
            print(f"FAIL {fn.__name__}"); traceback.print_exc()
    print(f"\n{ok}/{len(fns)} passed")
    sys.exit(0 if ok == len(fns) else 1)
