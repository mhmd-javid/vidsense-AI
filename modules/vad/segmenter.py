"""Speech segmentation (stage 2) + the bridge that feeds VAD windows to ASR.

WhisperX couples VAD + chunk-merging + batched ASR inside
``FasterWhisperPipeline.transcribe``: it calls ``vad_model.preprocess_audio`` →
``vad_model(...)`` → ``vad_model.merge_chunks(...)``. To keep VAD a *distinct,
observable* stage while running it **exactly once**, we:

  1. run VAD in :class:`~modules.vad.detector.VoiceActivityDetector` → windows
     ``[{start, end, segments}]`` (the mission's ``{start_time, end_time}``
     speech regions), then
  2. wrap those windows in :func:`build_precomputed_vad` — a ``Vad`` subclass
     whose ``__call__`` returns the pre-computed windows and whose
     ``merge_chunks`` is the identity. Handing it to ``load_model(vad_model=…)``
     makes WhisperX reuse our windows instead of recomputing VAD.

So there is **no double VAD inference**. If the standalone VAD stage fails, the
ASR engine falls back to building VAD itself from config (``vad_method``).
"""
from __future__ import annotations

from typing import Any, Dict, List

from core.utils import get_logger

logger = get_logger(__name__)


def build_precomputed_vad(windows: List[Dict[str, Any]]):
    """Return a WhisperX ``Vad`` instance that replays *windows* verbatim.

    ``Vad`` (and the ``issubclass(type(vad_model), Vad)`` check in
    ``transcribe``) is imported lazily so this module stays importable without
    the speech stack. Defining the subclass inside the factory keeps that import
    deferred.
    """
    from modules.whisperx.vads import Vad

    class _PrecomputedVad(Vad):
        def __init__(self, w: List[Dict[str, Any]]):
            super().__init__(0.5)  # onset must be in (0, 1); unused for replay
            self._windows = list(w)

        def __call__(self, audio=None, **kwargs):  # noqa: D401 - replay
            return self._windows

        @staticmethod
        def preprocess_audio(audio):
            return audio

        @staticmethod
        def merge_chunks(segments, chunk_size, onset: float = 0.5, offset=None):
            # Already merged in the VAD stage — identity keeps them unchanged.
            return segments

    return _PrecomputedVad(windows)


def references(windows: List[Dict[str, Any]]) -> List[Dict[str, float]]:
    """The mission's segmentation output: ``[{start_time, end_time}]`` regions."""
    return [
        {"start_time": round(float(w["start"]), 3), "end_time": round(float(w["end"]), 3)}
        for w in (windows or [])
    ]


def speech_stats(windows: List[Dict[str, Any]]) -> Dict[str, float]:
    """Observable VAD/segmentation telemetry derived from the speech windows."""
    windows = windows or []
    speech_seconds = sum(max(0.0, float(w["end"]) - float(w["start"])) for w in windows)
    return {
        "num_speech_regions": len(windows),
        "speech_seconds": round(speech_seconds, 3),
    }


def windows_from_segments(segments: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Fallback: reconstruct ``{start, end}`` windows from ASR segments.

    Used for telemetry when the ASR engine built its own VAD (i.e. the
    standalone VAD stage was skipped or failed), so the segmentation stage stays
    observable either way.
    """
    out: List[Dict[str, Any]] = []
    for seg in segments or []:
        if seg.get("start") is None or seg.get("end") is None:
            continue
        out.append({"start": float(seg["start"]), "end": float(seg["end"]), "segments": []})
    return out
