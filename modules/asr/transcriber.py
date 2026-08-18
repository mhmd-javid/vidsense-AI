"""ASR via faster-whisper (CTranslate2 backend).

Key behaviours for the 4 GB-VRAM target machine:
  * ``load()`` / ``unload()`` are explicit so the workflow can free the GPU
    before the LLM stage. ``unload()`` drops the model and forces GC so
    CTranslate2 releases device memory.
  * ``device: auto`` tries CUDA and *transparently falls back to CPU* if the
    GPU (or cuDNN) is unavailable — the pipeline never hard-fails on GPU issues.
  * int8 quantization by default. large-v3 is intentionally not the default.
"""
from __future__ import annotations

import gc
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, List, Optional

from core.utils import get_logger

logger = get_logger(__name__)

ProgressCb = Optional[Callable[[float, str], None]]


def _add_nvidia_dll_dirs() -> None:
    """Best-effort: expose CUDA DLLs shipped by nvidia-* pip packages.

    If the user installs ``nvidia-cublas-cu12`` / ``nvidia-cudnn-cu12``, their
    DLLs land in site-packages\\nvidia\\*/bin but are not on PATH. Registering
    those dirs lets CTranslate2 find cublas/cudnn and run ASR on the GPU. Harmless
    (no-op) when the packages aren't installed.
    """
    if os.name != "nt":
        return
    try:
        import importlib.util

        spec = importlib.util.find_spec("nvidia")
        if not spec or not spec.submodule_search_locations:
            return
        nvidia_root = Path(list(spec.submodule_search_locations)[0])
        for bin_dir in nvidia_root.glob("*/bin"):
            if bin_dir.is_dir():
                os.add_dll_directory(str(bin_dir))
    except Exception as exc:  # pragma: no cover
        logger.debug("Could not register NVIDIA DLL dirs: %s", exc)


def _is_gpu_library_error(exc: Exception) -> bool:
    msg = str(exc).lower()
    return any(k in msg for k in ("cublas", "cudnn", "cuda", "library", "cudart"))


@dataclass
class Segment:
    start: float
    end: float
    text: str

    def as_dict(self) -> dict:
        return {"start": self.start, "end": self.end, "text": self.text}


@dataclass
class TranscriptResult:
    video_id: str
    language: str
    language_probability: float
    duration: float
    segments: List[Segment] = field(default_factory=list)

    @property
    def full_text(self) -> str:
        return " ".join(s.text.strip() for s in self.segments).strip()

    def as_dict(self) -> dict:
        return {
            "video_id": self.video_id,
            "language": self.language,
            "language_probability": self.language_probability,
            "duration": self.duration,
            "full_text": self.full_text,
            "segments": [s.as_dict() for s in self.segments],
        }


class Transcriber:
    def __init__(
        self,
        model_size: str = "small",
        device: str = "auto",
        compute_type: str = "int8",
        language: Optional[str] = None,
        beam_size: int = 5,
        vad_filter: bool = True,
        download_root: Optional[str] = None,
    ):
        self.model_size = model_size
        self.requested_device = device
        self.requested_compute = compute_type
        self.language = language
        self.beam_size = beam_size
        self.vad_filter = vad_filter
        self.download_root = download_root
        self._model = None
        self.active_device: Optional[str] = None
        self.active_compute: Optional[str] = None

    # ------------------------------------------------------------- lifecycle
    def _resolve_device(self) -> tuple[str, str]:
        if self.requested_device != "auto":
            return self.requested_device, self.requested_compute
        try:
            import ctranslate2

            if ctranslate2.get_cuda_device_count() > 0:
                return "cuda", self.requested_compute
        except Exception as exc:  # pragma: no cover
            logger.debug("CUDA probe failed: %s", exc)
        return "cpu", "int8"

    def load(self) -> None:
        if self._model is not None:
            return
        from faster_whisper import WhisperModel

        device, compute = self._resolve_device()
        if device == "cuda":
            _add_nvidia_dll_dirs()
        try:
            logger.info(
                "Loading ASR model '%s' on %s (%s)…", self.model_size, device, compute
            )
            self._model = WhisperModel(
                self.model_size,
                device=device,
                compute_type=compute,
                download_root=self.download_root,
            )
            self.active_device, self.active_compute = device, compute
        except Exception as exc:
            # Graceful GPU -> CPU fallback (e.g. missing cuDNN / OOM).
            if device != "cpu":
                logger.warning(
                    "GPU ASR load failed (%s). Falling back to CPU int8.", exc
                )
                self._model = WhisperModel(
                    self.model_size,
                    device="cpu",
                    compute_type="int8",
                    download_root=self.download_root,
                )
                self.active_device, self.active_compute = "cpu", "int8"
            else:
                raise

    def unload(self) -> None:
        """Release the model and free GPU memory (critical on 4 GB VRAM)."""
        if self._model is not None:
            logger.info("Unloading ASR model to free memory.")
        self._model = None
        self.active_device = None
        gc.collect()

    def __enter__(self) -> "Transcriber":
        self.load()
        return self

    def __exit__(self, *exc) -> None:
        self.unload()

    # -------------------------------------------------------------- transcribe
    def transcribe(
        self,
        audio_path: str | Path,
        video_id: str,
        progress_cb: ProgressCb = None,
    ) -> TranscriptResult:
        audio_path = Path(audio_path)
        if not audio_path.exists():
            raise FileNotFoundError(f"Audio file not found: {audio_path}")
        self.load()
        try:
            return self._run(audio_path, video_id, progress_cb)
        except Exception as exc:
            # CUDA runtime libs (cublas/cudnn) are loaded lazily at inference,
            # so GPU errors surface here, not at load(). Fall back to CPU once.
            if self.active_device != "cpu" and _is_gpu_library_error(exc):
                logger.warning(
                    "GPU ASR inference failed (%s). Falling back to CPU int8.", exc
                )
                self.unload()
                self.requested_device = "cpu"
                self.requested_compute = "int8"
                self.load()
                return self._run(audio_path, video_id, progress_cb)
            raise

    def _run(
        self, audio_path: Path, video_id: str, progress_cb: ProgressCb
    ) -> TranscriptResult:
        segments_iter, info = self._model.transcribe(
            str(audio_path),
            language=self.language,
            beam_size=self.beam_size,
            vad_filter=self.vad_filter,
        )
        total = float(getattr(info, "duration", 0.0)) or 0.0
        logger.info(
            "Transcribing (%s, lang=%s p=%.2f, %.1fs)…",
            self.active_device,
            info.language,
            info.language_probability,
            total,
        )

        segments: List[Segment] = []
        for seg in segments_iter:
            text = (seg.text or "").strip()
            if text:
                segments.append(Segment(start=seg.start, end=seg.end, text=text))
            if progress_cb and total > 0:
                frac = min(0.99, seg.end / total)
                progress_cb(frac, f"Transcribing… {frac*100:.0f}%")

        if progress_cb:
            progress_cb(1.0, "Transcription complete")

        return TranscriptResult(
            video_id=video_id,
            language=info.language,
            language_probability=float(info.language_probability),
            duration=total,
            segments=segments,
        )
