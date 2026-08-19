"""Final transcript assembly + serialization (JSON / SRT / VTT).

We write our own exporters rather than using WhisperX's ``get_writer`` so the
output matches the mission's schema exactly and we control speaker labels and
confidence fields:

    {
      "video_id", "language", "language_probability",
      "duration", "num_speakers", "quality_score", "low_confidence_segments",
      "segments": [
        {"speaker", "start", "end", "text", "confidence", "low_confidence",
         "words": [{"word", "start", "end", "score", "speaker"}]}
      ]
    }
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Optional

from core.utils import ensure_dir

DEFAULT_SPEAKER = "SPEAKER_00"


def _clean_word(w: Dict[str, Any]) -> Dict[str, Any]:
    out: Dict[str, Any] = {"word": w.get("word", "")}
    if w.get("start") is not None:
        out["start"] = round(float(w["start"]), 3)
    if w.get("end") is not None:
        out["end"] = round(float(w["end"]), 3)
    if w.get("score") is not None:
        out["score"] = round(float(w["score"]), 4)
    if w.get("speaker"):
        out["speaker"] = w["speaker"]
    return out


def _clean_segment(seg: Dict[str, Any]) -> Dict[str, Any]:
    out: Dict[str, Any] = {
        "speaker": seg.get("speaker") or DEFAULT_SPEAKER,
        "start": round(float(seg.get("start", 0.0)), 3),
        "end": round(float(seg.get("end", 0.0)), 3),
        "text": (seg.get("text") or "").strip(),
    }
    # Optional LLM-corrected text, stored beside the untouched original. Present
    # only when llm_postprocess ran; absent otherwise, so output is unchanged
    # when the feature is disabled.
    if seg.get("text_corrected") is not None:
        out["text_corrected"] = (seg.get("text_corrected") or "").strip()
    out["confidence"] = seg.get("confidence")
    out["low_confidence"] = bool(seg.get("low_confidence", False))
    out["words"] = [_clean_word(w) for w in (seg.get("words") or [])]
    return out


def assemble(
    video_id: str,
    result: Dict[str, Any],
    duration: Optional[float] = None,
    title: Optional[str] = None,
) -> Dict[str, Any]:
    """Shape the pipeline *result* into the final transcript dict."""
    segments = sorted(result.get("segments", []), key=lambda s: float(s.get("start", 0.0)))
    clean_segments = [_clean_segment(s) for s in segments]

    speakers = {s["speaker"] for s in clean_segments}
    final: Dict[str, Any] = {
        "video_id": video_id,
        "language": result.get("language", "fa"),
        "language_probability": result.get("language_probability"),
        "duration": round(float(duration), 3) if duration else None,
        "num_speakers": len(speakers) or 1,
        "quality_score": result.get("quality_score"),
        "low_confidence_segments": result.get("low_confidence_segments", 0),
        "segments": clean_segments,
    }
    if title:
        final["title"] = title
    return final


# --------------------------------------------------------------- timestamps --
def _fmt(seconds: float, sep: str) -> str:
    seconds = max(0.0, float(seconds))
    h = int(seconds // 3600)
    m = int((seconds % 3600) // 60)
    s = int(seconds % 60)
    ms = int(round((seconds - int(seconds)) * 1000))
    if ms == 1000:  # rounding spillover
        s, ms = s + 1, 0
    return f"{h:02d}:{m:02d}:{s:02d}{sep}{ms:03d}"


def _srt_ts(seconds: float) -> str:
    return _fmt(seconds, ",")


def _vtt_ts(seconds: float) -> str:
    return _fmt(seconds, ".")


def _line(seg: Dict[str, Any], multi_speaker: bool) -> str:
    text = seg["text"]
    speaker = seg.get("speaker")
    if multi_speaker and speaker:
        return f"[{speaker}] {text}"
    return text


def to_srt(segments: List[Dict[str, Any]]) -> str:
    multi = len({s.get("speaker") for s in segments}) > 1
    blocks = []
    for i, seg in enumerate(segments, start=1):
        blocks.append(
            f"{i}\n{_srt_ts(seg['start'])} --> {_srt_ts(seg['end'])}\n{_line(seg, multi)}\n"
        )
    return "\n".join(blocks)


def to_vtt(segments: List[Dict[str, Any]]) -> str:
    multi = len({s.get("speaker") for s in segments}) > 1
    blocks = ["WEBVTT\n"]
    for seg in segments:
        blocks.append(
            f"{_vtt_ts(seg['start'])} --> {_vtt_ts(seg['end'])}\n{_line(seg, multi)}\n"
        )
    return "\n".join(blocks)


def save_all(final: Dict[str, Any], transcripts_dir: str | Path, video_id: str) -> Dict[str, str]:
    """Write JSON/SRT/VTT for *final* and return their absolute paths."""
    out_dir = ensure_dir(transcripts_dir)
    segments = final.get("segments", [])

    json_path = Path(out_dir) / f"{video_id}.json"
    srt_path = Path(out_dir) / f"{video_id}.srt"
    vtt_path = Path(out_dir) / f"{video_id}.vtt"

    json_path.write_text(json.dumps(final, ensure_ascii=False, indent=2), encoding="utf-8")
    srt_path.write_text(to_srt(segments), encoding="utf-8")
    vtt_path.write_text(to_vtt(segments), encoding="utf-8")

    return {
        "transcript_path": str(json_path.resolve()),
        "srt_path": str(srt_path.resolve()),
        "vtt_path": str(vtt_path.resolve()),
    }
