"""Video ingestion: download from URLs (YouTube / Aparat / any yt-dlp site /
direct media links) or register a locally uploaded file.

Design goals:
  * Never crash the caller — every failure raises a clear ``DownloadError``.
  * Robust for Iranian sources: an **Aparat direct-API bypass** resolves the
    real MP4 stream when yt-dlp's extractor lags behind Aparat's site changes,
    and a **bundled-ffmpeg fallback** pulls direct streams yt-dlp can't.
  * Prefer a single progressive file so we don't *require* stream merging.
"""
from __future__ import annotations

import re
import shutil
import subprocess
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Optional

from core.config import DownloadSection
from core.utils import ensure_dir, get_logger, make_video_id, resolve_path

logger = get_logger(__name__)

# Aparat share URLs look like https://www.aparat.com/v/<hash>?...
_APARAT_HASH_RE = re.compile(r"aparat\.com/v/([a-zA-Z0-9]+)")

_BROWSER_HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
        "(KHTML, like Gecko) Chrome/122.0.0.0 Safari/537.36"
    ),
    "Accept-Language": "fa-IR,fa;q=0.9,en-US;q=0.8,en;q=0.7",
}


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


def _ffmpeg_exe() -> Optional[str]:
    """Path to the bundled ffmpeg binary, or None if unavailable."""
    try:
        import imageio_ffmpeg

        return imageio_ffmpeg.get_ffmpeg_exe()
    except Exception:  # pragma: no cover - optional
        return None


def _ffmpeg_dir() -> Optional[str]:
    exe = _ffmpeg_exe()
    return str(Path(exe).parent) if exe else None


def resolve_aparat_direct_link(url: str, *, insecure_ssl: bool = True) -> Optional[str]:
    """Resolve an Aparat share URL to a direct MP4 stream via Aparat's JSON API.

    Returns the highest-listed direct URL, or ``None`` if the URL isn't an
    Aparat link, the API is unreachable, or no stream is listed. Uses
    ``verify=False`` by default because some domestic CDNs present certificate
    chains that ``requests`` rejects; this only affects the metadata call.
    """
    match = _APARAT_HASH_RE.search(url or "")
    if not match:
        return None

    hash_id = match.group(1)
    api_url = f"https://www.aparat.com/api/fa/v1/video/video/show/videohash/{hash_id}"
    try:
        import requests
        import urllib3

        if insecure_ssl:
            urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)
        headers = {**_BROWSER_HEADERS, "Accept": "application/json"}
        resp = requests.get(api_url, headers=headers, timeout=15, verify=not insecure_ssl)
        if resp.status_code != 200:
            logger.warning("Aparat API returned HTTP %s for %s", resp.status_code, hash_id)
            return None
        data = resp.json()
        file_link_all = (
            data.get("data", {}).get("attributes", {}).get("file_link_all", [])
        )
        for item in file_link_all:
            urls = item.get("urls") or []
            if urls:
                logger.info("Resolved Aparat direct stream for %s", hash_id)
                return urls[0]
    except Exception as exc:  # never let this bypass crash ingestion
        logger.warning("Aparat direct-API resolution failed: %s", exc)
    return None


class VideoDownloader:
    """Thin, defensive wrapper around yt-dlp + local file handling."""

    def __init__(self, videos_dir: str | Path, cfg: Optional[DownloadSection] = None):
        self.videos_dir = ensure_dir(videos_dir)
        self.cfg = cfg or DownloadSection()

    # ------------------------------------------------------------------ URL --
    def download_url(self, url: str) -> IngestResult:
        """Download *url* into ``videos_dir`` and return an IngestResult.

        Strategy: for Aparat, resolve the direct MP4 first; then try yt-dlp; if
        yt-dlp fails on a direct stream, fall back to the bundled ffmpeg.
        Never lets a raw yt-dlp/ffmpeg exception escape — always DownloadError.
        """
        url = (url or "").strip()
        if not url:
            raise DownloadError("No URL provided.")

        video_id = make_video_id(url)

        # Decide the fetch target + whether it's a direct (non-extractor) stream.
        target_url = url
        is_direct_stream = False
        if self.cfg.aparat_direct_api and "aparat.com" in url.lower():
            direct = resolve_aparat_direct_link(url, insecure_ssl=self.cfg.insecure_ssl)
            if direct:
                target_url, is_direct_stream = direct, True
        elif url.lower().split("?")[0].endswith((".mp4", ".m4a", ".webm", ".mkv", ".mp3", ".wav")):
            is_direct_stream = True

        try:
            return self._download_with_ytdlp(url, target_url, video_id, is_direct_stream)
        except DownloadError:
            raise
        except Exception as exc:
            # yt-dlp failed. For a direct stream, try the raw ffmpeg fallback.
            if is_direct_stream:
                logger.warning("yt-dlp failed (%s); trying ffmpeg fallback…", exc)
                fb = self._download_with_ffmpeg(target_url, video_id)
                if fb:
                    return IngestResult(
                        video_id=video_id,
                        title=video_id,
                        source=url,
                        source_type="url",
                        filepath=fb,
                        extractor="ffmpeg",
                    )
            hint = self._friendly_hint(url, exc)
            logger.error("Download failed for %s: %s", url, exc)
            raise DownloadError(hint) from exc

    def _download_with_ytdlp(
        self, url: str, target_url: str, video_id: str, is_direct_stream: bool
    ) -> IngestResult:
        try:
            import yt_dlp
        except Exception as exc:  # pragma: no cover
            raise DownloadError(f"yt-dlp is not installed: {exc}") from exc

        outtmpl = str(self.videos_dir / f"{video_id}.%(ext)s")
        h = int(self.cfg.max_height)
        ydl_opts = {
            "format": f"best[ext=mp4][height<={h}]/best[ext=mp4]/best",
            "outtmpl": outtmpl,
            "noplaylist": True,
            "quiet": True,
            "no_warnings": True,
            "restrictfilenames": True,
            "retries": int(self.cfg.retries),
            "socket_timeout": int(self.cfg.socket_timeout),
            "nocheckcertificate": self.cfg.insecure_ssl,
            "merge_output_format": "mp4",
            "http_headers": dict(_BROWSER_HEADERS),
        }
        ffdir = _ffmpeg_dir()
        if ffdir:
            ydl_opts["ffmpeg_location"] = ffdir
        if is_direct_stream:
            # Bypass site-specific extractors for a raw media URL.
            ydl_opts["force_generic_extractor"] = True

        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            info = ydl.extract_info(target_url, download=True)
            filepath = self._resolve_downloaded_path(ydl, info, video_id)

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

    def _download_with_ffmpeg(self, url: str, video_id: str) -> Optional[str]:
        """Last-resort direct pull via the bundled ffmpeg. Returns path or None."""
        exe = _ffmpeg_exe()
        if not exe:
            return None
        out_path = self.videos_dir / f"{video_id}.mp4"
        # Stream-copy first (fast, lossless); works for most direct MP4 CDNs.
        cmd = [
            exe, "-y", "-nostdin",
            "-headers", f"User-Agent: {_BROWSER_HEADERS['User-Agent']}\r\n",
            "-i", url,
            "-c", "copy",
            "-loglevel", "error",
            str(out_path),
        ]
        try:
            proc = subprocess.run(
                cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, timeout=3600
            )
            if proc.returncode == 0 and out_path.exists() and out_path.stat().st_size > 0:
                return str(out_path.resolve())
            logger.warning("ffmpeg stream-copy failed: %s", (proc.stderr or "").strip()[-300:])
        except Exception as exc:  # pragma: no cover
            logger.warning("ffmpeg fallback error: %s", exc)
        return None

    def _resolve_downloaded_path(self, ydl, info, video_id: str) -> Optional[str]:
        reqs = info.get("requested_downloads") or []
        if reqs and reqs[0].get("filepath"):
            return reqs[0]["filepath"]
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
                "Could not download from Aparat. The direct-API bypass and "
                "yt-dlp both failed — the video may be private/removed. "
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
