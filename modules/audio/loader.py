"""Load audio into the numpy array format the speech models expect.

Why this exists: the vendored WhisperX ``load_audio`` shells out to a bare
``ffmpeg`` on the system PATH. On this project's target (Windows, no system
ffmpeg) the only ffmpeg is the static binary bundled by ``imageio-ffmpeg``. So
we decode with *that* binary and hand a ready numpy array to WhisperX's
``transcribe`` / ``align`` / diarization calls — all of which accept an ndarray
and therefore never touch their internal ``load_audio``.

The returned array matches WhisperX exactly: float32, mono, 16 kHz, normalized
to [-1, 1] (``int16 / 32768``).
"""
from __future__ import annotations

import subprocess
from pathlib import Path

import numpy as np

from core.utils import get_logger, resolve_path

logger = get_logger(__name__)

SAMPLE_RATE = 16000


def _ffmpeg_exe() -> str:
    """Path to the bundled ffmpeg binary (imageio-ffmpeg)."""
    try:
        import imageio_ffmpeg

        return imageio_ffmpeg.get_ffmpeg_exe()
    except Exception as exc:  # pragma: no cover - optional dependency
        raise RuntimeError(
            "ffmpeg is not available. Install with `pip install imageio-ffmpeg` "
            "or add ffmpeg to your PATH."
        ) from exc


def load_audio(path: str | Path, sample_rate: int = SAMPLE_RATE) -> np.ndarray:
    """Decode *path* to a mono float32 waveform at *sample_rate*.

    Raises ``FileNotFoundError`` if the file is missing and ``RuntimeError`` if
    ffmpeg fails to decode it.
    """
    src = resolve_path(path)
    if not src.exists():
        raise FileNotFoundError(f"Audio file not found: {src}")

    cmd = [
        _ffmpeg_exe(),
        "-nostdin",
        "-threads", "0",
        "-i", str(src),
        "-f", "s16le",
        "-ac", "1",
        "-acodec", "pcm_s16le",
        "-ar", str(sample_rate),
        "-loglevel", "error",
        "-",
    ]
    try:
        out = subprocess.run(cmd, capture_output=True, check=True).stdout
    except subprocess.CalledProcessError as exc:  # pragma: no cover
        detail = (exc.stderr or b"").decode(errors="replace")[-500:]
        raise RuntimeError(f"Failed to decode audio: {detail}") from exc

    return np.frombuffer(out, np.int16).flatten().astype(np.float32) / 32768.0
