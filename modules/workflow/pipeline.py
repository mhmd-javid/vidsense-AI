"""Deterministic end-to-end Persian transcription pipeline.

    URL/upload -> download -> extract -> preprocess -> VAD/segment -> ASR ->
    align -> diarize -> post-process -> confidence -> persist (JSON/SRT/VTT + DB)

A plain, readable sequence of stages over a shared ``context`` with a
``progress_cb`` for the UI -- no agents, no LangGraph. Each stage is small and
observable.

VRAM discipline (T1000, 4 GB -- models never co-resident): every heavy model is
``load -> process -> unload`` before the next. Order of heavy loads:
VAD (tiny) -> ASR (medium) -> wav2vec2 alignment -> pyannote diarization, each
released before the next. On CPU this still bounds RAM. The audio is decoded to
a numpy array **once** and reused by every stage (so WhisperX never shells out
to a bare ``ffmpeg``).
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional

from core.config import Config
from core.utils import get_logger
from modules.audio.extractor import AudioExtractor
from modules.audio.preprocess import AudioPreprocessor
from modules.ingestion.downloader import IngestResult, VideoDownloader
from modules.storage import db as dbmod
from modules.storage import transcript as tx
from modules.storage.db import Database

logger = get_logger(__name__)

# Ordered pipeline stages (the UI renders these as a checklist).
STAGES: List[tuple[str, str]] = [
    ("download", "Downloading / loading video"),
    ("extract", "Extracting audio"),
    ("preprocess", "Preprocessing audio"),
    ("vad", "Detecting speech (VAD)"),
    ("transcribe", "Transcribing (ASR)"),
    ("align", "Aligning words"),
    ("diarize", "Identifying speakers"),
    ("postprocess", "Normalizing text"),
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
    language_probability: Optional[float] = None
    duration: float = 0.0
    num_segments: int = 0
    num_speakers: int = 0
    num_speech_regions: int = 0
    quality_score: Optional[float] = None
    asr_device: str = ""
    transcript_path: str = ""
    srt_path: str = ""
    vtt_path: str = ""
    error: str = ""

    def as_dict(self) -> dict:
        return self.__dict__


def _resolve_torch_device(preferred: str) -> str:
    """Resolve a torch device for VAD/align/diarize (torch-based stages)."""
    pref = (preferred or "auto").lower()
    if pref != "auto":
        return pref
    try:
        import torch

        if torch.cuda.is_available():
            return "cuda"
    except Exception:
        pass
    return "cpu"


class ProcessingPipeline:
    def __init__(self, cfg: Config, db: Database):
        self.cfg = cfg
        self.db = db
        self.downloader = VideoDownloader(cfg.videos_dir_abs, cfg.download)
        self.extractor = AudioExtractor(cfg.audio_dir_abs, cfg.audio.sample_rate)
        self.preprocessor = AudioPreprocessor(
            cfg.preprocess, cfg.audio_dir_abs, cfg.audio.sample_rate
        )

    # ------------------------------------------------------------ entrypoints
    def process_url(self, url: str, progress_cb: ProgressCb = None) -> ProcessingResult:
        return self._safe_process(source_kind="url", source=url, progress_cb=progress_cb)

    def process_upload(
        self, data: bytes, filename: str, progress_cb: ProgressCb = None
    ) -> ProcessingResult:
        return self._safe_process(
            source_kind="upload", data=data, filename=filename, progress_cb=progress_cb
        )

    def process_local(self, path: str, progress_cb: ProgressCb = None) -> ProcessingResult:
        return self._safe_process(source_kind="local", source=path, progress_cb=progress_cb)

    # ------------------------------------------------------------ core steps
    def _safe_process(self, **kwargs) -> ProcessingResult:
        """Run the pipeline, converting any failure into a clean result."""
        progress_cb: ProgressCb = kwargs.get("progress_cb")
        self._current_video_id = "?"
        try:
            return self._process(**kwargs)
        except Exception as exc:  # never crash the app
            logger.exception("Processing failed")
            vid = getattr(self, "_current_video_id", "?")
            try:
                if vid != "?":
                    self.db.set_status(vid, dbmod.STATUS_ERROR, str(exc))
            except Exception:
                pass
            if progress_cb:
                progress_cb("error", f"Failed: {exc}", None)
            return ProcessingResult(video_id=vid, success=False, error=str(exc))

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
        # --- 1. Ingest ----------------------------------------------------
        self._emit(progress_cb, "download", "Fetching video...", None)
        if source_kind == "url":
            ingest: IngestResult = self.downloader.download_url(source)
        elif source_kind == "local":
            ingest = self.downloader.ingest_local(source)
        else:
            ingest = self.downloader.save_upload(data, filename)

        video_id = ingest.video_id
        self._current_video_id = video_id
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

        # --- 2. Extract 16 kHz mono WAV -----------------------------------
        self.db.set_status(video_id, dbmod.STATUS_EXTRACTING)
        self._emit(progress_cb, "extract", "Extracting 16 kHz mono audio...", None)
        audio_res = self.extractor.extract(ingest.filepath, video_id)
        duration = audio_res.duration or ingest.duration or 0.0
        self.db.upsert_video(video_id, audio_path=audio_res.audio_path, duration=duration)
        self._emit(progress_cb, "extract", "Audio ready", 1.0)

        # --- 3. Preprocess (faithful) -------------------------------------
        self.db.set_status(video_id, dbmod.STATUS_PREPROCESSING)
        self._emit(progress_cb, "preprocess", "Cleaning audio...", None)
        clean_path = self.preprocessor.process(audio_res.audio_path, video_id)
        self._emit(progress_cb, "preprocess", "Audio preprocessed", 1.0)

        # Decode to ndarray ONCE; every model reuses it (avoids bare ffmpeg).
        from modules.audio.loader import load_audio

        audio = load_audio(clean_path, self.cfg.audio.sample_rate)

        # --- 4. VAD + segmentation (tiny model; released before ASR) ------
        self.db.set_status(video_id, dbmod.STATUS_VAD)
        self._emit(progress_cb, "vad", "Detecting speech regions...", None)
        vad_windows = self._run_vad(audio)
        from modules.vad import segmenter as seg

        if vad_windows is not None:
            stats = seg.speech_stats(vad_windows)
        else:
            stats = {"num_speech_regions": 0, "speech_seconds": 0.0}
        self._emit(
            progress_cb, "vad",
            f"{stats['num_speech_regions']} speech region(s)", 1.0,
        )

        # --- 5. ASR (load -> transcribe -> UNLOAD) --------------------------
        self.db.set_status(video_id, dbmod.STATUS_TRANSCRIBING)
        self._emit(progress_cb, "transcribe", "Transcribing...", None)
        from modules.asr.engine import ASREngine

        engine = ASREngine(self.cfg.asr, self.cfg.vad, self.cfg.models_dir_abs.as_posix())
        try:
            result = engine.transcribe(audio, vad_windows=vad_windows)
            asr_device = engine.active_device or "?"
        finally:
            engine.unload()  # free before alignment
        num_segments = len(result.get("segments", []))
        language = result.get("language") or self.cfg.asr.language or "fa"
        if vad_windows is None:  # ASR ran its own VAD -- derive telemetry
            stats = seg.speech_stats(seg.windows_from_segments(result.get("segments", [])))
        self._emit(progress_cb, "transcribe", f"{num_segments} segments", 1.0)

        # Device for the remaining torch stages: follow ASR's actual device.
        torch_device = "cpu" if asr_device == "cpu" else _resolve_torch_device(self.cfg.asr.device)

        # --- 6. Word alignment (load -> align -> UNLOAD) --------------------
        self.db.set_status(video_id, dbmod.STATUS_ALIGNING)
        self._emit(progress_cb, "align", "Aligning words...", None)
        from modules.alignment.aligner import WordAligner

        aligner = WordAligner(
            self.cfg.alignment, language=language,
            device=torch_device, model_dir=self.cfg.models_dir_abs.as_posix(),
        )
        try:
            result = aligner.align(result.get("segments", []), audio)
            result["language"] = language
        finally:
            aligner.unload()
        self._emit(progress_cb, "align", "Alignment done", 1.0)

        # --- 7. Diarization (optional; graceful; load -> run -> UNLOAD) -----
        self.db.set_status(video_id, dbmod.STATUS_DIARIZING)
        self._emit(progress_cb, "diarize", "Identifying speakers...", None)
        from modules.diarization.diarizer import SpeakerDiarizer

        diarizer = SpeakerDiarizer(
            self.cfg.diarization, device=torch_device,
            cache_dir=self.cfg.models_dir_abs.as_posix(),
        )
        try:
            result, num_speakers = diarizer.diarize(result, audio)
        finally:
            diarizer.unload()
        self._emit(progress_cb, "diarize", f"{num_speakers} speaker(s)", 1.0)

        # --- 8. Post-process (faithful) + 9. confidence -------------------
        self.db.set_status(video_id, dbmod.STATUS_POSTPROCESSING)
        self._emit(progress_cb, "postprocess", "Normalizing Persian text...", None)
        from modules.postprocess.persian import normalize_transcript
        from modules.confidence.scorer import score_transcript

        result["language"] = language
        result["language_probability"] = result.get("language_probability")
        normalize_transcript(result, self.cfg.postprocess)

        # --- 8b. Optional LLM spell/word correction (per-segment; faithful) --
        # Runs strictly AFTER normalization. Stores each corrected string as
        # `text_corrected` beside the untouched original `text`; timestamps,
        # words, alignment and speaker labels are never touched. Off by default:
        # when disabled the module is not even imported, so behavior is
        # byte-identical to the pre-LLM pipeline.
        if self.cfg.llm_postprocess.enabled:
            lp = self.cfg.llm_postprocess
            active_model = lp.model if lp.provider == "ollama" else lp.openrouter_model
            logger.info(
                "[postprocess] LLM correction started (provider=%s, model=%s, "
                "timeout=%ds, max_word_change_ratio=%.2f)",
                lp.provider,
                active_model,
                lp.timeout_seconds,
                lp.max_word_change_ratio,
            )
            from modules.postprocess.llm_correct import correct_transcript

            self._emit(progress_cb, "postprocess", "LLM correcting Persian text...", None)
            llm_stats = correct_transcript(result, self.cfg.llm_postprocess)
            self._emit(
                progress_cb,
                "postprocess",
                f"LLM corrected {llm_stats['corrected']}/{llm_stats['total']} "
                f"segment(s); {llm_stats['rejected_similarity']} kept original "
                f"(similarity guard)",
                None,
            )

        score_transcript(result, self.cfg.confidence)

        # --- 10. Assemble + persist ---------------------------------------
        final = tx.assemble(video_id, result, duration=duration, title=ingest.title)
        final["num_speech_regions"] = stats["num_speech_regions"]
        final["speech_seconds"] = stats["speech_seconds"]
        paths = tx.save_all(final, self.cfg.transcripts_dir_abs, video_id)

        full_text = " ".join(tx.display_text(s) for s in final["segments"]).strip()
        self.db.upsert_video(
            video_id,
            language=final["language"],
            language_prob=final.get("language_probability"),
            num_speakers=final["num_speakers"],
            quality_score=final.get("quality_score"),
            full_text=full_text,
            transcript_path=paths["transcript_path"],
            srt_path=paths["srt_path"],
            vtt_path=paths["vtt_path"],
            duration=duration,
        )
        self.db.set_status(video_id, dbmod.STATUS_READY)
        self._emit(progress_cb, "ready", "Transcript ready", 1.0)

        return ProcessingResult(
            video_id=video_id,
            success=True,
            title=ingest.title,
            language=final["language"],
            language_probability=final.get("language_probability"),
            duration=duration,
            num_segments=len(final["segments"]),
            num_speakers=final["num_speakers"],
            num_speech_regions=stats["num_speech_regions"],
            quality_score=final.get("quality_score"),
            asr_device=asr_device,
            transcript_path=paths["transcript_path"],
            srt_path=paths["srt_path"],
            vtt_path=paths["vtt_path"],
        )

    # ---------------------------------------------------------------- helpers
    def _run_vad(self, audio) -> Optional[List[Dict[str, Any]]]:
        """Run the standalone VAD stage. Returns windows, or None on failure
        (in which case the ASR engine falls back to building VAD itself)."""
        from modules.vad.detector import VoiceActivityDetector

        device = _resolve_torch_device(self.cfg.asr.device)
        detector = VoiceActivityDetector(self.cfg.vad, device=device)
        try:
            windows = detector.segment(audio)
            logger.info("VAD produced %d speech window(s).", len(windows))
            return windows
        except Exception as exc:
            logger.warning("VAD stage failed (%s); ASR will run its own VAD.", exc)
            return None
        finally:
            detector.unload()