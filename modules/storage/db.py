"""SQLite persistence layer (metadata + status; JSON is the transcript).

The canonical transcript is the JSON file at ``data/transcripts/<id>.json``.
This table holds video metadata, processing status, quality metrics, and
pointers to the transcript / subtitle files — enough to list and re-open
past runs without re-reading every JSON. All access goes through one
``Database`` class so a later move to Postgres touches only this file.

Schema
------
videos(video_id PK, title, source, source_type, filepath, audio_path,
       duration, language, language_prob, num_speakers, quality_score,
       status, error, transcript_path, srt_path, vtt_path, full_text,
       extractor, created_at, updated_at)
"""
from __future__ import annotations

import sqlite3
import threading
from pathlib import Path
from typing import Any, Dict, List, Optional

from core.utils import ensure_dir, get_logger

logger = get_logger(__name__)

# Processing status values (plain strings for portability), one per heavy stage.
STATUS_PENDING = "pending"
STATUS_DOWNLOADING = "downloading"
STATUS_EXTRACTING = "extracting"
STATUS_PREPROCESSING = "preprocessing"
STATUS_VAD = "vad"
STATUS_TRANSCRIBING = "transcribing"
STATUS_ALIGNING = "aligning"
STATUS_DIARIZING = "diarizing"
STATUS_POSTPROCESSING = "postprocessing"
STATUS_READY = "ready"
STATUS_ERROR = "error"


_SCHEMA = """
CREATE TABLE IF NOT EXISTS videos (
    video_id        TEXT PRIMARY KEY,
    title           TEXT,
    source          TEXT,
    source_type     TEXT,
    filepath        TEXT,
    audio_path      TEXT,
    duration        REAL,
    language        TEXT,
    language_prob   REAL,
    num_speakers    INTEGER,
    quality_score   REAL,
    status          TEXT NOT NULL DEFAULT 'pending',
    error           TEXT,
    transcript_path TEXT,
    srt_path        TEXT,
    vtt_path        TEXT,
    full_text       TEXT,
    extractor       TEXT,
    created_at      TEXT DEFAULT (datetime('now')),
    updated_at      TEXT DEFAULT (datetime('now'))
);
"""

# Columns expected on ``videos``, with the type used when back-filling an older
# database via ALTER TABLE. No non-constant defaults appear here: SQLite forbids
# ``ADD COLUMN ... DEFAULT datetime('now')``, so ``created_at``/``updated_at`` are
# managed explicitly by ``upsert_video`` instead of a column default on upgrade.
_EXPECTED_COLUMNS: Dict[str, str] = {
    "title": "TEXT", "source": "TEXT", "source_type": "TEXT",
    "filepath": "TEXT", "audio_path": "TEXT", "duration": "REAL",
    "language": "TEXT", "language_prob": "REAL", "num_speakers": "INTEGER",
    "quality_score": "REAL", "status": "TEXT", "error": "TEXT",
    "transcript_path": "TEXT", "srt_path": "TEXT", "vtt_path": "TEXT",
    "full_text": "TEXT", "extractor": "TEXT",
    "created_at": "TEXT", "updated_at": "TEXT",
}


class Database:
    """Thread-aware SQLite wrapper (Streamlit reruns across threads)."""

    def __init__(self, db_path: str | Path):
        self.db_path = Path(db_path)
        ensure_dir(self.db_path.parent)
        self._lock = threading.Lock()
        self._conn = sqlite3.connect(
            str(self.db_path), check_same_thread=False, timeout=30
        )
        self._conn.row_factory = sqlite3.Row
        self._conn.executescript(_SCHEMA)
        self._migrate()
        self._conn.commit()

    def _migrate(self) -> None:
        """Idempotently add columns missing from an older ``videos`` table.

        ``CREATE TABLE IF NOT EXISTS`` never alters a table that already exists,
        so a database created by an earlier schema (e.g. the pre-transcript RAG
        build) is missing newer columns such as ``num_speakers`` /
        ``quality_score`` / ``srt_path`` / ``vtt_path``. Back-fill them here so
        ``upsert_video`` cannot fail with "no such column". Runs on every open;
        no-op once the schema is current.
        """
        existing = {row["name"] for row in self._conn.execute("PRAGMA table_info(videos)")}
        missing = [(c, t) for c, t in _EXPECTED_COLUMNS.items() if c not in existing]
        for col, sqltype in missing:
            self._conn.execute(f"ALTER TABLE videos ADD COLUMN {col} {sqltype}")
        if missing:
            logger.info(
                "Migrated 'videos' table: added %d column(s): %s",
                len(missing), ", ".join(c for c, _ in missing),
            )

    def close(self) -> None:
        with self._lock:
            self._conn.close()

    # -------------------------------------------------------------- videos --
    def upsert_video(self, video_id: str, **fields: Any) -> None:
        fields = {k: v for k, v in fields.items() if v is not None}
        with self._lock:
            exists = self._conn.execute(
                "SELECT 1 FROM videos WHERE video_id=?", (video_id,)
            ).fetchone()
            if exists:
                if fields:
                    cols = ", ".join(f"{k}=?" for k in fields)
                    self._conn.execute(
                        f"UPDATE videos SET {cols}, updated_at=datetime('now') WHERE video_id=?",
                        (*fields.values(), video_id),
                    )
            else:
                cols = ", ".join(["video_id", *fields.keys()])
                ph = ", ".join(["?"] * (1 + len(fields)))
                self._conn.execute(
                    f"INSERT INTO videos ({cols}) VALUES ({ph})",
                    (video_id, *fields.values()),
                )
            self._conn.commit()

    def set_status(self, video_id: str, status: str, error: Optional[str] = None) -> None:
        self.upsert_video(video_id, status=status, error=(error or ""))

    def get_video(self, video_id: str) -> Optional[Dict[str, Any]]:
        with self._lock:
            row = self._conn.execute(
                "SELECT * FROM videos WHERE video_id=?", (video_id,)
            ).fetchone()
        return dict(row) if row else None

    def list_videos(self, ready_only: bool = False) -> List[Dict[str, Any]]:
        q = "SELECT * FROM videos"
        if ready_only:
            q += f" WHERE status='{STATUS_READY}'"
        q += " ORDER BY updated_at DESC"
        with self._lock:
            rows = self._conn.execute(q).fetchall()
        return [dict(r) for r in rows]

    def delete_video(self, video_id: str) -> None:
        with self._lock:
            self._conn.execute("DELETE FROM videos WHERE video_id=?", (video_id,))
            self._conn.commit()
