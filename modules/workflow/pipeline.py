"""Deterministic end-to-end processing pipeline.

    URL/upload -> download -> extract audio -> ASR -> chunk -> embed -> store

No agents, no LangGraph: a plain, readable sequence of stages with a shared
``context`` dict and a ``progress_cb`` for the UI. This is intentionally
"agent-ready" — each stage is a small function over the context, so a future
orchestrator (LangGraph / an agent loop) could schedule them without rewriting
the logic — but for the MVP the deterministic order is the whole point.

VRAM discipline (4 GB target): the ASR model is loaded, used, then UNLOADED
before anything else. Embeddings run on CPU. The LLM (Ollama) is only touched
during chat. The three heavy models are never resident together.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Dict, List, Optional

from core.config import Config
from core.utils import ensure_dir, get_logger
from modules.asr.chunker import chunk_segments
from modules.asr.transcriber import Transcriber
from modules.audio.extractor import AudioExtractor
from modules.embedding.embedder import Embedder
from modules.ingestion.downloader import IngestResult, VideoDownloader
from modules.storage import db as dbmod
from modules.storage.db import Database

logger = get_logger(__name__)

# Ordered pipeline stages (used by the UI to render a checklist).
STAGES: List[tuple[str, str]] = [
    ("download", "Downloading / Loading video"),
    ("extract", "Extracting audio"),
    ("transcribe", "Transcribing (ASR)"),
    ("chunk", "Chunking transcript"),
    ("embed", "Creating knowledge base (embeddings)"),
    ("ready", "Ready"),
]

# cb(stage_key, message, fraction_or_None)
ProgressCb = Optional[Callable[[str, str, Optional[float]], None]]


@dataclass
class ProcessingResult:
    video_id: str
    success: bool
    title: str = ""
    language: str = ""
    duration: float = 0.0
    num_segments: int = 0
    num_chunks: int = 0
    asr_device: str = ""
    error: str = ""

    def as_dict(self) -> dict:
        return self.__dict__


class ProcessingPipeline:
    def __init__(self, cfg: Config, db: Database, embedder: Embedder):
        self.cfg = cfg
        self.db = db
        self.embedder = embedder  # shared, CPU, kept warm for chat
        self.downloader = VideoDownloader(cfg.videos_dir_abs)
        self.extractor = AudioExtractor(cfg.audio_dir_abs)

    # ------------------------------------------------------------ entrypoints
    def process_url(self, url: str, progress_cb: ProgressCb = None) -> ProcessingResult:
        return self._safe_process(source_kind="url", source=url, progress_cb=progress_cb)

    def process_upload(
        self, data: bytes, filename: str, progress_cb: ProgressCb = None
    ) -> ProcessingResult:
        return self._safe_process(
            source_kind="upload", data=data, filename=filename, progress_cb=progress_cb
        )

    # ------------------------------------------------------------ core steps
    def _safe_process(self, **kwargs) -> ProcessingResult:
        """Run the pipeline, converting any failure into a clean result."""
        progress_cb: ProgressCb = kwargs.get("progress_cb")
        video_id = "?"
        try:
            return self._process(**kwargs)
        except Exception as exc:  # never crash the app
            logger.exception("Processing failed")
            try:
                if video_id != "?":
                    self.db.set_status(video_id, dbmod.STATUS_ERROR, str(exc))
            except Exception:
                pass
            if progress_cb:
                progress_cb("error", f"Failed: {exc}", None)
            return ProcessingResult(video_id=video_id, success=False, error=str(exc))

    def _emit(self, cb: ProgressCb, stage: str, msg: str, frac: Optional[float] = None):
        logger.info("[%s] %s", stage, msg)
        if cb:
            cb(stage, msg, frac)

    def _process(
        self,
        source_kind: str,
        source: str = "",
        data: bytes = b"",
        filename: str = "",
        progress_cb: ProgressCb = None,
    ) -> ProcessingResult:
        # --- 1. Ingest (download or register upload) -----------------------
        self._emit(progress_cb, "download", "Fetching video…", None)
        if source_kind == "url":
            ingest: IngestResult = self.downloader.download_url(source)
        else:
            ingest = self.downloader.save_upload(data, filename)

        video_id = ingest.video_id
        self.db.upsert_video(
            video_id,
            title=ingest.title,
            source=ingest.source,
            source_type=ingest.source_type,
            filepath=ingest.filepath,
            extractor=ingest.extractor,
            duration=ingest.duration,
        )
        self.db.set_status(video_id, dbmod.STATUS_DOWNLOADING)
        self._emit(progress_cb, "download", f"Got '{ingest.title}'", 1.0)

        # --- 2. Extract audio ---------------------------------------------
        self.db.set_status(video_id, dbmod.STATUS_EXTRACTING)
        self._emit(progress_cb, "extract", "Extracting 16 kHz mono audio…", None)
        audio = self.extractor.extract(ingest.filepath, video_id)
        self.db.upsert_video(video_id, audio_path=audio.audio_path, duration=audio.duration)
        self._emit(progress_cb, "extract", "Audio ready", 1.0)

        # --- 3. Transcribe (ASR) — load, run, UNLOAD to free VRAM ---------
        self.db.set_status(video_id, dbmod.STATUS_TRANSCRIBING)
        transcriber = Transcriber(
            model_size=self.cfg.asr.model_size,
            device=self.cfg.asr.device,
            compute_type=self.cfg.asr.compute_type,
            language=self.cfg.asr.language,
            beam_size=self.cfg.asr.beam_size,
            vad_filter=self.cfg.asr.vad_filter,
            download_root=self.cfg.asr.download_root,
        )
        try:
            transcript = transcriber.transcribe(
                audio.audio_path,
                video_id,
                progress_cb=lambda f, m: self._emit(progress_cb, "transcribe", m, f),
            )
            asr_device = transcriber.active_device or "?"
        finally:
            transcriber.unload()  # critical: release GPU before next stages

        # Persist transcript (JSON file + full text/lang in DB).
        transcript_path = self._save_transcript(video_id, transcript)
        self.db.upsert_video(
            video_id,
            language=transcript.language,
            language_prob=transcript.language_probability,
            full_text=transcript.full_text,
            transcript_path=str(transcript_path),
            duration=transcript.duration or audio.duration,
        )

        # --- 4. Chunk ------------------------------------------------------
        self.db.set_status(video_id, dbmod.STATUS_CHUNKING)
        self._emit(progress_cb, "chunk", "Merging segments into chunks…", None)
        chunks = chunk_segments(
            [s.as_dict() for s in transcript.segments],
            target_seconds=self.cfg.chunking.target_seconds,
            max_seconds=self.cfg.chunking.max_seconds,
            max_chars=self.cfg.chunking.max_chars,
            min_chars=self.cfg.chunking.min_chars,
        )
        self._emit(progress_cb, "chunk", f"{len(chunks)} chunks", 1.0)

        # --- 5. Embed (CPU) + store ---------------------------------------
        self.db.set_status(video_id, dbmod.STATUS_EMBEDDING)
        self._emit(progress_cb, "embed", "Embedding chunks…", None)
        chunk_dicts = [c.as_dict() for c in chunks]
        embeddings = None
        if chunk_dicts:
            self.embedder.load()
            embeddings = self.embedder.embed_passages([c["text"] for c in chunk_dicts])
        self.db.replace_chunks(
            video_id, chunk_dicts, embeddings=embeddings, embedding_model=self.embedder.model_name
        )
        self._emit(progress_cb, "embed", "Knowledge base ready", 1.0)

        # --- 6. Done -------------------------------------------------------
        self.db.set_status(video_id, dbmod.STATUS_READY)
        self._emit(progress_cb, "ready", "Ready to chat", 1.0)

        return ProcessingResult(
            video_id=video_id,
            success=True,
            title=ingest.title,
            language=transcript.language,
            duration=transcript.duration or (audio.duration or 0.0),
            num_segments=len(transcript.segments),
            num_chunks=len(chunks),
            asr_device=asr_device,
        )

    # ---------------------------------------------------------------- helpers
    def _save_transcript(self, video_id: str, transcript) -> Path:
        out_dir = ensure_dir(self.cfg.transcripts_dir_abs)
        path = out_dir / f"{video_id}.json"
        with open(path, "w", encoding="utf-8") as fh:
            json.dump(transcript.as_dict(), fh, ensure_ascii=False, indent=2)
        return path
