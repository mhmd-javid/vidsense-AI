"""Video ingestion: download from URLs (YouTube / Aparat / any yt-dlp site) or
register a locally uploaded file.

Design goals for the MVP:
  * Never crash the caller — every failure raises a clear ``DownloadError``.
  * Do not let one flaky extractor (e.g. Aparat) take down the whole app; the
    UI can always fall back to manual upload.
  * Prefer a single progressive file so we don't *require* stream merging.
"""
from __future__ import annotations

import shutil
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Optional

from core.utils import ensure_dir, get_logger, make_video_id, resolve_path

logger = get_logger(__name__)


class DownloadError(Exception):
    """Raised when a URL cannot be downloaded. Message is user-friendly."""


@dataclass
class IngestResult:
    video_id: str
    title: str
    source: str                 # original URL or filename
    source_type: str            # "url" | "upload"
    filepath: str               # absolute path to the media file on disk
    duration: Optional[float] = None
    extractor: Optional[str] = None

    def as_dict(self) -> dict:
        return asdict(self)


def _ffmpeg_dir() -> Optional[str]:
    """Directory containing the bundled ffmpeg binary, for yt-dlp merging."""
    try:
        import imageio_ffmpeg

        return str(Path(imageio_ffmpeg.get_ffmpeg_exe()).parent)
    except Exception:  # pragma: no cover - optional
        return None


class VideoDownloader:
    """Thin, defensive wrapper around yt-dlp + local file handling."""

    def __init__(self, videos_dir: str | Path):
        self.videos_dir = ensure_dir(videos_dir)

    # ------------------------------------------------------------------ URL --
    def download_url(self, url: str) -> IngestResult:
        """Download *url* into ``videos_dir`` and return an IngestResult.

        Raises ``DownloadError`` (never lets yt-dlp exceptions escape raw).
        """
        url = (url or "").strip()
        if not url:
            raise DownloadError("No URL provided.")

        try:
            import yt_dlp
        except Exception as exc:  # pragma: no cover
            raise DownloadError(f"yt-dlp is not installed: {exc}") from exc

        video_id = make_video_id(url)
        outtmpl = str(self.videos_dir / f"{video_id}.%(ext)s")

        ydl_opts = {
            # Prefer a single progressive mp4 to avoid mandatory merging; fall
            # back to best available. Cap height to keep files small (we only
            # need the audio for ASR).
            "format": "best[ext=mp4][height<=720]/best[ext=mp4]/best",
            "outtmpl": outtmpl,
            "noplaylist": True,
            "quiet": True,
            "no_warnings": True,
            "restrictfilenames": True,
            "retries": 3,
            "socket_timeout": 30,
            "merge_output_format": "mp4",
        }
        ffdir = _ffmpeg_dir()
        if ffdir:
            ydl_opts["ffmpeg_location"] = ffdir

        try:
            with yt_dlp.YoutubeDL(ydl_opts) as ydl:
                info = ydl.extract_info(url, download=True)
                # For single video, requested_downloads holds the real path.
                filepath = self._resolve_downloaded_path(ydl, info, video_id)
        except DownloadError:
            raise
        except Exception as exc:
            hint = self._friendly_hint(url, exc)
            logger.error("Download failed for %s: %s", url, exc)
            raise DownloadError(hint) from exc

        if not filepath or not Path(filepath).exists():
            raise DownloadError(
                "Download reported success but no file was found on disk. "
                "Please try another URL or upload the file manually."
            )

        return IngestResult(
            video_id=video_id,
            title=(info.get("title") or video_id),
            source=url,
            source_type="url",
            filepath=str(Path(filepath).resolve()),
            duration=info.get("duration"),
            extractor=info.get("extractor_key") or info.get("extractor"),
        )

    def _resolve_downloaded_path(self, ydl, info, video_id: str) -> Optional[str]:
        # yt-dlp best source of truth for the final path:
        reqs = info.get("requested_downloads") or []
        if reqs and reqs[0].get("filepath"):
            return reqs[0]["filepath"]
        # Fallback: predict from template, else glob by id.
        try:
            predicted = ydl.prepare_filename(info)
            if Path(predicted).exists():
                return predicted
        except Exception:
            pass
        matches = sorted(self.videos_dir.glob(f"{video_id}.*"))
        return str(matches[0]) if matches else None

    @staticmethod
    def _friendly_hint(url: str, exc: Exception) -> str:
        msg = str(exc).lower()
        if "aparat" in url.lower():
            return (
                "Could not download from Aparat. Aparat's page structure "
                "changes often and may not be supported right now. "
                "Please download the file manually and use the Upload option."
            )
        if "unsupported url" in msg:
            return "This URL is not supported by yt-dlp. Try uploading the file instead."
        if "http error 403" in msg or "sign in" in msg or "login" in msg:
            return (
                "The site refused the download (login/region protected). "
                "Please upload the file manually instead."
            )
        if "network" in msg or "timed out" in msg or "connection" in msg:
            return "Network error while downloading. Check your connection and retry."
        return f"Could not download this URL. Details: {exc}"

    # --------------------------------------------------------------- Upload --
    def save_upload(self, data: bytes, filename: str) -> IngestResult:
        """Persist uploaded *data* under videos_dir and return an IngestResult."""
        if not data:
            raise DownloadError("Uploaded file is empty.")
        safe_name = Path(filename).name or "upload.mp4"
        ext = Path(safe_name).suffix or ".mp4"
        video_id = make_video_id(f"{safe_name}:{len(data)}")
        dest = self.videos_dir / f"{video_id}{ext}"
        try:
            dest.write_bytes(data)
        except Exception as exc:
            raise DownloadError(f"Could not save uploaded file: {exc}") from exc
        return IngestResult(
            video_id=video_id,
            title=safe_name,
            source=safe_name,
            source_type="upload",
            filepath=str(dest.resolve()),
        )

    def ingest_local(self, path: str | Path) -> IngestResult:
        """Register an existing local file (copies it into videos_dir)."""
        src = resolve_path(path)
        if not src.exists():
            raise DownloadError(f"Local file not found: {src}")
        video_id = make_video_id(f"{src.name}:{src.stat().st_size}")
        dest = self.videos_dir / f"{video_id}{src.suffix or '.mp4'}"
        if src.resolve() != dest.resolve():
            shutil.copy2(src, dest)
        return IngestResult(
            video_id=video_id,
            title=src.name,
            source=str(src),
            source_type="upload",
            filepath=str(dest.resolve()),
        )
