"""Audio extraction using the bundled ffmpeg (via imageio-ffmpeg).

Produces the exact format faster-whisper likes best: 16 kHz, mono, 16-bit PCM
WAV. We shell out to ffmpeg (matches the project's "use FFmpeg" requirement)
and use PyAV only as a lightweight probe for media duration, since
imageio-ffmpeg ships ffmpeg but not ffprobe.
"""
from __future__ import annotations

import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

from core.utils import ensure_dir, get_logger, resolve_path

logger = get_logger(__name__)

ASR_SAMPLE_RATE = 16000


class AudioExtractionError(Exception):
    """Raised when audio cannot be extracted from a media file."""


@dataclass
class AudioResult:
    audio_path: str
    sample_rate: int
    duration: Optional[float] = None


def _ffmpeg_exe() -> str:
    try:
        import imageio_ffmpeg

        return imageio_ffmpeg.get_ffmpeg_exe()
    except Exception as exc:  # pragma: no cover
        raise AudioExtractionError(
            "ffmpeg is not available. Install with `pip install imageio-ffmpeg` "
            "or add ffmpeg to your PATH."
        ) from exc


def probe_duration(media_path: str | Path) -> Optional[float]:
    """Return media duration in seconds using PyAV, or None if unknown."""
    try:
        import av

        with av.open(str(media_path)) as container:
            if container.duration is not None:
                return float(container.duration) / av.time_base
    except Exception as exc:
        logger.debug("Duration probe failed for %s: %s", media_path, exc)
    return None


class AudioExtractor:
    """Extract normalized 16 kHz mono WAV audio from any media file."""

    def __init__(self, audio_dir: str | Path, sample_rate: int = ASR_SAMPLE_RATE):
        self.audio_dir = ensure_dir(audio_dir)
        self.sample_rate = sample_rate

    def extract(self, media_path: str | Path, video_id: str) -> AudioResult:
        media_path = resolve_path(media_path)
        if not media_path.exists():
            raise AudioExtractionError(f"Media file not found: {media_path}")

        out_path = self.audio_dir / f"{video_id}.wav"
        ffmpeg = _ffmpeg_exe()
        cmd = [
            ffmpeg,
            "-y",                      # overwrite
            "-i", str(media_path),
            "-vn",                     # no video
            "-ac", "1",                # mono
            "-ar", str(self.sample_rate),
            "-c:a", "pcm_s16le",       # 16-bit PCM
            "-loglevel", "error",
            str(out_path),
        ]
        logger.info("Extracting audio: %s -> %s", media_path.name, out_path.name)
        try:
            proc = subprocess.run(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                timeout=3600,
            )
        except FileNotFoundError as exc:
            raise AudioExtractionError(f"Could not run ffmpeg: {exc}") from exc
        except subprocess.TimeoutExpired as exc:
            raise AudioExtractionError("Audio extraction timed out.") from exc

        if proc.returncode != 0 or not out_path.exists():
            detail = (proc.stderr or "").strip()[-500:]
            raise AudioExtractionError(
                f"ffmpeg failed to extract audio (code {proc.returncode}). {detail}"
            )

        duration = probe_duration(out_path) or probe_duration(media_path)
        return AudioResult(
            audio_path=str(out_path.resolve()),
            sample_rate=self.sample_rate,
            duration=duration,
        )
