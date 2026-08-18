"""Small shared helpers: paths, timestamp formatting, ids, logging."""
from __future__ import annotations

import hashlib
import logging
import re
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]

_LOG_CONFIGURED = False


def project_root() -> Path:
    """Absolute path to the project root (the folder containing this package)."""
    return PROJECT_ROOT


def resolve_path(path: str | Path) -> Path:
    """Resolve *path* against the project root unless it is already absolute."""
    p = Path(path)
    return p if p.is_absolute() else (PROJECT_ROOT / p)


def ensure_dir(path: str | Path) -> Path:
    """Create a directory (and parents) if missing; return the resolved path."""
    p = resolve_path(path)
    p.mkdir(parents=True, exist_ok=True)
    return p


def setup_logging(level: str = "INFO") -> None:
    """Configure root logging once, with a compact console format."""
    global _LOG_CONFIGURED
    if _LOG_CONFIGURED:
        return
    logging.basicConfig(
        level=getattr(logging, str(level).upper(), logging.INFO),
        format="%(asctime)s | %(levelname)-7s | %(name)s | %(message)s",
        datefmt="%H:%M:%S",
    )
    _LOG_CONFIGURED = True


def get_logger(name: str) -> logging.Logger:
    return logging.getLogger(name)


# --------------------------------------------------------------------------- #
# Timestamps
# --------------------------------------------------------------------------- #
def format_timestamp(seconds: float) -> str:
    """Format seconds as MM:SS, or HH:MM:SS when >= 1 hour. Robust to None."""
    if seconds is None:
        return "00:00"
    seconds = max(0, int(round(float(seconds))))
    h, rem = divmod(seconds, 3600)
    m, s = divmod(rem, 60)
    if h:
        return f"{h:d}:{m:02d}:{s:02d}"
    return f"{m:02d}:{s:02d}"


def format_range(start: float, end: float) -> str:
    """Format a time range like '02:10–02:35' (en-dash separator)."""
    return f"{format_timestamp(start)}–{format_timestamp(end)}"


# --------------------------------------------------------------------------- #
# IDs
# --------------------------------------------------------------------------- #
def make_video_id(source: str) -> str:
    """Deterministic short id derived from a source URL or file path."""
    digest = hashlib.sha1(source.encode("utf-8")).hexdigest()
    return digest[:12]


def slugify(text: str, max_len: int = 60) -> str:
    """Filesystem-safe slug for filenames."""
    text = re.sub(r"[^\w\s-]", "", text, flags=re.UNICODE).strip().lower()
    text = re.sub(r"[\s_-]+", "-", text)
    return text[:max_len] or "video"
