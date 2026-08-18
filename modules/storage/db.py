"""SQLite persistence layer (source of truth for the MVP).

Stores video metadata, transcript pointer + full text, chunks, embeddings
(as float32 blobs) and processing status. The access goes through a single
``Database`` class so migrating to PostgreSQL later means reimplementing this
one class, not touching the pipeline.

Schema
------
videos(video_id PK, title, source, source_type, filepath, audio_path,
       duration, language, language_prob, status, error, transcript_path,
       full_text, extractor, created_at, updated_at)
chunks(id PK, video_id FK, chunk_index, start, end, text,
       embedding BLOB, embedding_dim, embedding_model)
"""
from __future__ import annotations

import sqlite3
import threading
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

import numpy as np

from core.utils import ensure_dir, get_logger

logger = get_logger(__name__)

# Processing status values (kept as plain strings for portability).
STATUS_PENDING = "pending"
STATUS_DOWNLOADING = "downloading"
STATUS_EXTRACTING = "extracting"
STATUS_TRANSCRIBING = "transcribing"
STATUS_CHUNKING = "chunking"
STATUS_EMBEDDING = "embedding"
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
    status          TEXT NOT NULL DEFAULT 'pending',
    error           TEXT,
    transcript_path TEXT,
    full_text       TEXT,
    extractor       TEXT,
    created_at      TEXT DEFAULT (datetime('now')),
    updated_at      TEXT DEFAULT (datetime('now'))
);

CREATE TABLE IF NOT EXISTS chunks (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    video_id        TEXT NOT NULL,
    chunk_index     INTEGER NOT NULL,
    start           REAL NOT NULL,
    end             REAL NOT NULL,
    text            TEXT NOT NULL,
    embedding       BLOB,
    embedding_dim   INTEGER,
    embedding_model TEXT,
    FOREIGN KEY (video_id) REFERENCES videos(video_id) ON DELETE CASCADE
);

CREATE INDEX IF NOT EXISTS idx_chunks_video ON chunks(video_id);
"""


@dataclass
class ChunkRow:
    id: int
    video_id: str
    chunk_index: int
    start: float
    end: float
    text: str
    embedding: Optional[np.ndarray]
    embedding_model: Optional[str]


def _vec_to_blob(vec: Optional[np.ndarray]) -> Optional[bytes]:
    if vec is None:
        return None
    return np.asarray(vec, dtype=np.float32).tobytes()


def _blob_to_vec(blob: Optional[bytes]) -> Optional[np.ndarray]:
    if blob is None:
        return None
    return np.frombuffer(blob, dtype=np.float32)


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
        self._conn.execute("PRAGMA foreign_keys = ON;")
        self._conn.executescript(_SCHEMA)
        self._conn.commit()

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
            self._conn.execute("DELETE FROM chunks WHERE video_id=?", (video_id,))
            self._conn.execute("DELETE FROM videos WHERE video_id=?", (video_id,))
            self._conn.commit()

    # -------------------------------------------------------------- chunks --
    def replace_chunks(
        self,
        video_id: str,
        chunks: Sequence[Dict[str, Any]],
        embeddings: Optional[np.ndarray] = None,
        embedding_model: Optional[str] = None,
    ) -> None:
        """Delete existing chunks for the video and insert the new set.

        ``chunks`` is a list of dicts with start/end/text (and optional
        chunk_index). ``embeddings`` is an aligned (N, dim) array or None.
        """
        with self._lock:
            self._conn.execute("DELETE FROM chunks WHERE video_id=?", (video_id,))
            for i, ch in enumerate(chunks):
                emb = None if embeddings is None else embeddings[i]
                self._conn.execute(
                    """INSERT INTO chunks
                       (video_id, chunk_index, start, end, text,
                        embedding, embedding_dim, embedding_model)
                       VALUES (?,?,?,?,?,?,?,?)""",
                    (
                        video_id,
                        int(ch.get("index", i)),
                        float(ch["start"]),
                        float(ch["end"]),
                        ch["text"],
                        _vec_to_blob(emb),
                        int(emb.shape[0]) if emb is not None else None,
                        embedding_model,
                    ),
                )
            self._conn.commit()

    def get_chunks(self, video_id: str, with_embeddings: bool = True) -> List[ChunkRow]:
        with self._lock:
            rows = self._conn.execute(
                "SELECT * FROM chunks WHERE video_id=? ORDER BY chunk_index",
                (video_id,),
            ).fetchall()
        out: List[ChunkRow] = []
        for r in rows:
            out.append(
                ChunkRow(
                    id=r["id"],
                    video_id=r["video_id"],
                    chunk_index=r["chunk_index"],
                    start=r["start"],
                    end=r["end"],
                    text=r["text"],
                    embedding=_blob_to_vec(r["embedding"]) if with_embeddings else None,
                    embedding_model=r["embedding_model"],
                )
            )
        return out

    def count_chunks(self, video_id: str) -> int:
        with self._lock:
            row = self._conn.execute(
                "SELECT COUNT(*) AS n FROM chunks WHERE video_id=?", (video_id,)
            ).fetchone()
        return int(row["n"]) if row else 0
