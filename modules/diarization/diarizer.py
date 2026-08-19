"""Speaker diarization over the vendored WhisperX (pyannote community-1).

**Optional and graceful by contract.** Diarization is the heaviest, most
fragile stage (gated model, needs a Hugging Face token, ~1–2 GB). It must never
fail the pipeline. If it is disabled, has no token, or errors at any point, we
label every segment and word ``SPEAKER_00`` and continue — the transcript is
still complete, just single-speaker.

When it does run: ``DiarizationPipeline`` produces speaker turns and
``assign_word_speakers`` attaches the dominant speaker to each segment/word.

Heavy imports (torch, pyannote, whisperx) are deferred to ``load()``.
"""
from __future__ import annotations

import gc
import os
from typing import Any, Dict, Optional, Tuple

from core.config import DiarizationSection
from core.utils import get_logger

logger = get_logger(__name__)

DEFAULT_SPEAKER = "SPEAKER_00"


def _resolve_token(cfg: DiarizationSection) -> Optional[str]:
    return (
        cfg.hf_token
        or os.environ.get("HF_TOKEN")
        or os.environ.get("HUGGINGFACE_TOKEN")
        or os.environ.get("HUGGING_FACE_HUB_TOKEN")
    )


def _label_single_speaker(result: Dict[str, Any]) -> Tuple[Dict[str, Any], int]:
    """Assign the default speaker to every segment/word. Returns (result, 1)."""
    for seg in result.get("segments", []):
        seg.setdefault("speaker", DEFAULT_SPEAKER)
        for word in seg.get("words", []) or []:
            word.setdefault("speaker", DEFAULT_SPEAKER)
    return result, 1


class SpeakerDiarizer:
    def __init__(
        self,
        cfg: DiarizationSection,
        device: str = "cpu",
        cache_dir: Optional[str] = None,
    ):
        self.cfg = cfg
        self.device = self._safe_device(device)
        self.cache_dir = cache_dir
        self._pipeline = None

    @staticmethod
    def _safe_device(device: str) -> str:
        if (device or "cpu").lower().startswith("cuda"):
            try:
                import torch

                if torch.cuda.is_available():
                    return "cuda"
            except Exception:
                pass
            return "cpu"
        return "cpu"

    # -------------------------------------------------------------- lifecycle
    def load(self) -> bool:
        if self._pipeline is not None:
            return True
        token = _resolve_token(self.cfg)
        if not token:
            raise RuntimeError(
                "Diarization needs a Hugging Face token (config diarization.hf_token "
                "or $HF_TOKEN) to access the gated model."
            )
        from modules.whisperx.diarize import DiarizationPipeline

        logger.info("Loading diarization model '%s' on %s…", self.cfg.model_name, self.device)
        self._pipeline = DiarizationPipeline(
            model_name=self.cfg.model_name,
            token=token,
            device=self.device,
            cache_dir=self.cache_dir,
        )
        return True

    def unload(self) -> None:
        if self._pipeline is not None:
            logger.info("Unloading diarization model to free memory.")
        self._pipeline = None
        gc.collect()
        try:
            import torch

            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except Exception:
            pass

    def __enter__(self) -> "SpeakerDiarizer":
        return self

    def __exit__(self, *exc) -> None:
        self.unload()

    # ---------------------------------------------------------------- diarize
    def diarize(self, result: Dict[str, Any], audio) -> Tuple[Dict[str, Any], int]:
        """Attach speakers to *result*. Falls back to single-speaker on any issue.

        Returns ``(result, num_speakers)``.
        """
        if not self.cfg.enabled:
            logger.info("Diarization disabled — labeling all as %s.", DEFAULT_SPEAKER)
            return _label_single_speaker(result)

        try:
            self.load()
            from modules.whisperx.diarize import assign_word_speakers

            diarize_df = self._pipeline(
                audio,
                num_speakers=self.cfg.num_speakers,
                min_speakers=self.cfg.min_speakers,
                max_speakers=self.cfg.max_speakers,
            )
            result = assign_word_speakers(diarize_df, result, fill_nearest=True)

            # Ensure every segment/word has *some* label, then count speakers.
            speakers = set()
            for seg in result.get("segments", []):
                seg.setdefault("speaker", DEFAULT_SPEAKER)
                speakers.add(seg["speaker"])
                for word in seg.get("words", []) or []:
                    word.setdefault("speaker", seg["speaker"])
            num_speakers = len(speakers) or 1
            logger.info("Diarization complete: %d speaker(s).", num_speakers)
            return result, num_speakers
        except Exception as exc:
            logger.warning(
                "Diarization failed (%s). Falling back to single-speaker %s.",
                exc,
                DEFAULT_SPEAKER,
            )
            return _label_single_speaker(result)
