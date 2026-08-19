#!/usr/bin/env python
"""Command-line entry point for the Persian transcription pipeline.

Examples
--------
    python scripts/transcribe.py --url "https://www.aparat.com/v/XXXXX"
    python scripts/transcribe.py --file path/to/video.mp4 --model medium --diarize
    python scripts/transcribe.py --url "<youtube>" --device cpu --no-align

Runs the full pipeline (download → extract → preprocess → VAD → ASR → align →
diarize → post-process → confidence), writes ``data/transcripts/<id>.{json,srt,vtt}``,
and prints a summary (language + probability, #segments, #speakers,
quality_score, ASR device).
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

# Make the project root importable when run as `python scripts/transcribe.py`.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from core.config import load_config  # noqa: E402
from core.services import build_services  # noqa: E402


def _utf8_stdout() -> None:
    """Persian output needs UTF-8; Windows consoles default to cp1252."""
    for stream in (sys.stdout, sys.stderr):
        try:
            stream.reconfigure(encoding="utf-8")  # type: ignore[attr-defined]
        except Exception:
            pass


def _progress(stage: str, msg: str, frac):
    pct = f" ({frac*100:.0f}%)" if isinstance(frac, (int, float)) else ""
    print(f"  [{stage:>11}] {msg}{pct}", flush=True)


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Persian video → transcript pipeline.")
    src = p.add_mutually_exclusive_group(required=True)
    src.add_argument("--url", help="Video URL (YouTube / Aparat / direct media).")
    src.add_argument("--file", help="Local media file to ingest.")

    p.add_argument("--model", help="Whisper model size (overrides config, e.g. medium, large-v3).")
    p.add_argument("--device", choices=["auto", "cpu", "cuda"], help="Compute device override.")
    p.add_argument("--language", help="Force language code (default from config: fa).")
    p.add_argument("--diarize", action="store_true", help="Enable speaker diarization.")
    p.add_argument("--no-align", action="store_true", help="Disable word-level alignment.")
    p.add_argument("--config", help="Path to an alternative config.yaml.")
    return p


def main(argv=None) -> int:
    _utf8_stdout()
    args = build_parser().parse_args(argv)

    cfg = load_config(args.config)
    if args.model:
        cfg.asr.model_size = args.model
    if args.device:
        cfg.asr.device = args.device
    if args.language:
        cfg.asr.language = args.language
    if args.diarize:
        cfg.diarization.enabled = True
    if args.no_align:
        cfg.alignment.enabled = False

    svc = build_services(cfg)

    print(f"\n▶ Transcribing: {args.url or args.file}")
    print(f"  model={cfg.asr.model_size}  device={cfg.asr.device}  "
          f"align={cfg.alignment.enabled}  diarize={cfg.diarization.enabled}\n")

    start = time.time()
    if args.url:
        result = svc.processing.process_url(args.url, progress_cb=_progress)
    else:
        result = svc.processing.process_local(args.file, progress_cb=_progress)
    elapsed = time.time() - start

    print()
    if not result.success:
        print(f"✗ FAILED: {result.error}")
        return 1

    prob = result.language_probability
    print("✓ Done in %.1fs" % elapsed)
    print(f"  video_id        : {result.video_id}")
    print(f"  title           : {result.title}")
    print(f"  language         : {result.language}"
          + (f" (p={prob:.2f})" if isinstance(prob, (int, float)) else ""))
    print(f"  duration        : {result.duration:.1f}s")
    print(f"  speech regions  : {result.num_speech_regions}")
    print(f"  segments        : {result.num_segments}")
    print(f"  speakers        : {result.num_speakers}")
    if isinstance(result.quality_score, (int, float)):
        print(f"  quality_score   : {result.quality_score:.3f}")
    print(f"  ASR device      : {result.asr_device}")
    print(f"  transcript      : {result.transcript_path}")
    print(f"  subtitles       : {result.srt_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
