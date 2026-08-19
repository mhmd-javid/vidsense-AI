"""Faithful audio preprocessing before ASR.

Applies only *content-preserving* cleanup via an ffmpeg filter chain:
  * EBU R128 loudness normalization (``loudnorm``) — evens out level so quiet
    speech is not under-transcribed. Safe; does not change what was said.
  * optional high-pass filter — removes sub-speech rumble/hum when enabled.
  * optional light denoise (``afftdn``) — OFF by default because aggressive
    denoise can smear consonants and *hurt* ASR accuracy.

Nothing here invents or removes speech content. When ``enabled`` is false (or no
filter is active) it is a pass-through that returns the input path unchanged.
"""
from __future__ import annotations

import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import List

from core.config import PreprocessSection
from core.utils import ensure_dir, get_logger, resolve_path

logger = get_logger(__name__)


class AudioPreprocessError(Exception):
    """Raised when the ffmpeg preprocessing pass fails."""


def _ffmpeg_exe() -> str:
    try:
        import imageio_ffmpeg

        return imageio_ffmpeg.get_ffmpeg_exe()
    except Exception as exc:  # pragma: no cover
        raise AudioPreprocessError(
            "ffmpeg is not available (pip install imageio-ffmpeg)."
        ) from exc


@dataclass
class AudioPreprocessor:
    cfg: PreprocessSection
    audio_dir: Path
    sample_rate: int = 16000

    def __post_init__(self) -> None:
        self.audio_dir = ensure_dir(self.audio_dir)

    def _filters(self) -> List[str]:
        f: List[str] = []
        if self.cfg.highpass_hz and self.cfg.highpass_hz > 0:
            f.append(f"highpass=f={int(self.cfg.highpass_hz)}")
        if self.cfg.denoise:
            f.append("afftdn")
        if self.cfg.loudnorm:
            # I=-16 LUFS integrated, typical for speech; TP=-1.5 true-peak ceiling.
            f.append("loudnorm=I=-16:TP=-1.5:LRA=11")
        return f

    def process(self, wav_path: str | Path, video_id: str) -> str:
        """Return path to a cleaned WAV, or the original path if disabled/no-op."""
        src = resolve_path(wav_path)
        if not src.exists():
            raise AudioPreprocessError(f"Audio file not found: {src}")

        if not self.cfg.enabled:
            logger.info("Preprocess disabled — using raw audio.")
            return str(src)

        filters = self._filters()
        if not filters:
            logger.info("Preprocess enabled but no active filters — pass-through.")
            return str(src)

        out_path = self.audio_dir / f"{video_id}.clean.wav"
        cmd = [
            _ffmpeg_exe(),
            "-y",
            "-i", str(src),
            "-af", ",".join(filters),
            "-ac", "1",
            "-ar", str(self.sample_rate),
            "-c:a", "pcm_s16le",
            "-loglevel", "error",
            str(out_path),
        ]
        logger.info("Preprocessing audio (%s)…", ", ".join(filters))
        try:
            proc = subprocess.run(
                cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, timeout=3600
            )
        except subprocess.TimeoutExpired as exc:
            raise AudioPreprocessError("Audio preprocessing timed out.") from exc

        if proc.returncode != 0 or not out_path.exists():
            # Faithful degradation: if cleanup fails, fall back to raw audio
            # rather than aborting the whole transcription.
            logger.warning(
                "Preprocess ffmpeg failed (code %s); using raw audio. %s",
                proc.returncode,
                (proc.stderr or "").strip()[-300:],
            )
            return str(src)

        return str(out_path.resolve())
