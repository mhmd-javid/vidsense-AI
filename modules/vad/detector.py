"""Voice-activity detection (stage 1 of segmentation).

Thin wrapper over the vendored WhisperX VAD backends:
  * ``pyannote`` — uses the **bundled local weights**
    (``modules/whisperx/assets/pytorch_model.bin``), so it runs fully offline
    with no Hugging Face token.
  * ``silero`` — torch.hub model (downloaded once).

``segment()`` runs VAD and merges the detected speech into ``chunk_size``-bounded
windows — the ``{start, end}`` speech regions the mission calls for. Those windows
are then handed to the ASR engine unchanged (see ``segmenter.PrecomputedVad``),
so VAD runs exactly once.

Heavy imports (torch, pyannote, whisperx) are deferred to ``load()`` so this
module can be imported without the speech stack installed.
"""
from __future__ import annotations

from typing import Any, Dict, List, Optional

from core.config import VADSection
from core.utils import get_logger

logger = get_logger(__name__)


class VoiceActivityDetector:
    def __init__(self, cfg: VADSection, device: str = "cpu"):
        self.cfg = cfg
        self.device = device
        self._model = None  # a whisperx Vad subclass (Pyannote | Silero)

    # ------------------------------------------------------------- lifecycle
    def load(self) -> None:
        if self._model is not None:
            return
        import torch  # noqa: F401  (ensures torch present; used by backends)

        vad_options = {
            "vad_onset": float(self.cfg.onset),
            "vad_offset": float(self.cfg.offset),
            "chunk_size": int(self.cfg.chunk_size),
        }
        method = (self.cfg.method or "pyannote").lower()
        logger.info("Loading VAD backend: %s", method)
        if method == "silero":
            from modules.whisperx.vads import Silero

            self._model = Silero(**vad_options)
        elif method == "pyannote":
            import torch as _torch
            from modules.whisperx.vads import Pyannote

            self._model = Pyannote(
                _torch.device(self.device),
                token=None,
                model_fp=self.cfg.model_fp,
                **vad_options,
            )
        else:
            raise ValueError(f"Unknown vad.method: {self.cfg.method!r}")

    def unload(self) -> None:
        import gc

        if self._model is not None:
            logger.info("Unloading VAD model.")
        self._model = None
        gc.collect()
        try:
            import torch

            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except Exception:
            pass

    # --------------------------------------------------------------- segment
    def segment(self, audio) -> List[Dict[str, Any]]:
        """Run VAD + merge into speech windows ``[{start, end, segments}]``."""
        self.load()
        waveform = self._model.preprocess_audio(audio)
        raw = self._model({"waveform": waveform, "sample_rate": 16000})
        windows = self._model.merge_chunks(
            raw,
            int(self.cfg.chunk_size),
            onset=float(self.cfg.onset),
            offset=float(self.cfg.offset),
        )
        return windows or []

    def __enter__(self) -> "VoiceActivityDetector":
        self.load()
        return self

    def __exit__(self, *exc) -> None:
        self.unload()
