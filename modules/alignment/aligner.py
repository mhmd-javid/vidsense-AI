"""Word-level forced alignment (wav2vec2) over the vendored WhisperX.

For Persian, ``load_align_model`` auto-selects
``jonatasgrosman/wav2vec2-large-xlsr-53-persian`` and ``align`` produces
per-word ``{word, start, end, score}`` timings.

**Graceful by design:** alignment is a quality *enhancement*, never a hard
dependency. If it's disabled, the model can't load (offline / unsupported
language), or alignment raises, we log and return the ASR segments unchanged
with empty ``words`` — the pipeline keeps segment-level timings and continues.

Heavy imports (torch, transformers, whisperx) are deferred to ``load()``.
"""
from __future__ import annotations

import gc
from typing import Any, Dict, List, Optional

from core.config import AlignmentSection
from core.utils import get_logger

logger = get_logger(__name__)


def _passthrough(segments: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Return segments unchanged with empty word lists (alignment skipped)."""
    out = []
    for seg in segments or []:
        s = dict(seg)
        s.setdefault("words", [])
        out.append(s)
    return {"segments": out, "word_segments": []}


class WordAligner:
    def __init__(
        self,
        cfg: AlignmentSection,
        language: str = "fa",
        device: str = "cpu",
        model_dir: Optional[str] = None,
    ):
        self.cfg = cfg
        self.language = language or "fa"
        self.device = self._safe_device(device)
        self.model_dir = model_dir
        self._model = None
        self._metadata: Optional[Dict[str, Any]] = None

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
        """Load the alignment model. Returns True on success, False if skipped."""
        if self._model is not None:
            return True
        from modules.whisperx.alignment import load_align_model

        logger.info("Loading alignment model for '%s' on %s…", self.language, self.device)
        self._model, self._metadata = load_align_model(
            language_code=self.language,
            device=self.device,
            model_name=self.cfg.model_name,
            model_dir=self.model_dir,
        )
        return True

    def unload(self) -> None:
        if self._model is not None:
            logger.info("Unloading alignment model to free memory.")
        self._model = None
        self._metadata = None
        gc.collect()
        try:
            import torch

            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except Exception:
            pass

    def __enter__(self) -> "WordAligner":
        return self

    def __exit__(self, *exc) -> None:
        self.unload()

    # ----------------------------------------------------------------- align
    def align(self, segments: List[Dict[str, Any]], audio) -> Dict[str, Any]:
        """Align *segments* against *audio* → per-word timings, or pass-through."""
        if not self.cfg.enabled:
            logger.info("Alignment disabled — keeping segment-level timings.")
            return _passthrough(segments)
        if not segments:
            return {"segments": [], "word_segments": []}

        try:
            self.load()
            from modules.whisperx.alignment import align as _align

            result = _align(
                transcript=segments,
                model=self._model,
                align_model_metadata=self._metadata,
                audio=audio,
                device=self.device,
                interpolate_method=self.cfg.interpolate_method,
                return_char_alignments=self.cfg.return_char_alignments,
            )
            logger.info("Alignment complete: %d word segments.", len(result.get("word_segments", [])))
            return result
        except Exception as exc:
            # Faithful degradation — never let alignment fail the pipeline.
            logger.warning("Alignment failed (%s). Keeping segment-level timings.", exc)
            return _passthrough(segments)
