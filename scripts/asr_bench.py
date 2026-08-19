#!/usr/bin/env python
"""ASR-quality benchmark harness (isolated, additive — touches no pipeline code).

Runs ONLY the stages that affect the transcribed *words*:

    load_audio -> VAD (cached, identical for every config) -> ASR -> postprocess

Alignment and diarization are skipped on purpose: they add word timings and
speaker labels but never change the words (confirmed by
data/transcripts/ALIGNMENT_COMPARISON.md: "full transcribed text identical").
Skipping them keeps each run fast and isolates the ASR decode as the only
variable.

The harness reuses the production ``ASREngine`` / VAD / postprocess so results
faithfully reflect what the real pipeline would emit. It only *overrides* ASR
inference settings via CLI so we can A/B configs without editing config.yaml.

Outputs (under --out-dir, default data/benchmark):
  <tag>.meta.json          run metadata + timing + active device
  <tag>.raw.txt            ASR text BEFORE postprocess (run 1)  [for repetition metrics]
  <tag>.txt                final text AFTER faithful postprocess (run 1) [the "transcript"]
  <tag>.segments.json      per-segment {start,end,text,avg_logprob}
  <tag>.run2.txt           final text of run 2 (determinism check), if --runs>=2

Example:
  python scripts/asr_bench.py --audio data/audio/2b6d14f4f1fb.wav --model medium --tag B_medium_baseline --runs 2
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from core.config import load_config  # noqa: E402
from core.utils import get_logger  # noqa: E402

logger = get_logger("asr_bench")


def _utf8_stdout() -> None:
    for stream in (sys.stdout, sys.stderr):
        try:
            stream.reconfigure(encoding="utf-8")  # type: ignore[attr-defined]
        except Exception:
            pass


def _str2bool(v: str) -> bool:
    return str(v).strip().lower() in ("1", "true", "yes", "y", "on")


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Isolated ASR quality benchmark.")
    p.add_argument("--audio", required=True, help="Path to a 16 kHz mono WAV.")
    p.add_argument("--tag", required=True, help="Short label for output files.")
    p.add_argument("--out-dir", default="data/benchmark")
    p.add_argument("--runs", type=int, default=1, help="Repeat count (>=2 checks determinism).")

    # ---- ASR inference overrides (only these are allowed to change) ----
    p.add_argument("--model", dest="model_size")
    p.add_argument("--device")
    p.add_argument("--compute", dest="compute_type")
    p.add_argument("--language")
    p.add_argument("--beam", dest="beam_size", type=int)
    p.add_argument("--best-of", dest="best_of", type=int)
    p.add_argument("--patience", type=float)
    p.add_argument("--temps", help="Comma list, e.g. '0.0' or '0.0,0.2,0.4'. Default: 0.0 (deterministic).")
    p.add_argument("--no-speech", dest="no_speech_threshold", type=float)
    p.add_argument("--logprob", dest="log_prob_threshold", type=float)
    p.add_argument("--compression", dest="compression_ratio_threshold", type=float)
    p.add_argument("--cond", dest="condition_on_previous_text", type=_str2bool)
    p.add_argument("--initial-prompt", dest="initial_prompt",
                   help="Literal string, or @path to read prompt text from a file.")
    p.add_argument("--suppress-numerals", dest="suppress_numerals", type=_str2bool)
    p.add_argument("--batch-size", dest="batch_size", type=int)
    return p


def _apply_overrides(asr, args) -> None:
    # Deterministic by default: single temperature 0.0 (no sampling fallback).
    if args.temps is None:
        asr.temperatures = [0.0]
    else:
        asr.temperatures = [float(x) for x in args.temps.split(",") if x.strip() != ""]

    for name in (
        "model_size", "device", "compute_type", "language", "beam_size",
        "best_of", "patience", "no_speech_threshold", "log_prob_threshold",
        "compression_ratio_threshold", "condition_on_previous_text",
        "suppress_numerals", "batch_size",
    ):
        val = getattr(args, name, None)
        if val is not None:
            setattr(asr, name, val)

    if args.initial_prompt is not None:
        ip = args.initial_prompt
        if ip.startswith("@"):
            ip = Path(ip[1:]).read_text(encoding="utf-8").strip()
        asr.initial_prompt = ip if ip != "" else None


def _raw_text(segments) -> str:
    return " ".join((s.get("text") or "").strip() for s in segments).strip()


def main(argv=None) -> int:
    _utf8_stdout()
    args = build_parser().parse_args(argv)
    cfg = load_config()
    _apply_overrides(cfg.asr, args)

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    from modules.audio.loader import load_audio
    from modules.vad.detector import VoiceActivityDetector
    from modules.vad import segmenter as seg
    from modules.asr.engine import ASREngine
    from modules.postprocess.persian import normalize_transcript
    import copy

    audio = load_audio(args.audio, cfg.audio.sample_rate)
    dur = len(audio) / float(cfg.audio.sample_rate)

    # VAD once — identical windows for every config (fair, and faster).
    vad_cache = out_dir / (Path(args.audio).stem + ".vad.json")
    if vad_cache.exists():
        vad_windows = json.loads(vad_cache.read_text(encoding="utf-8"))
        for w in vad_windows:
            w.setdefault("segments", [])
    else:
        det = VoiceActivityDetector(cfg.vad, device="cpu")
        try:
            vad_windows = det.segment(audio)
        finally:
            det.unload()
        vad_cache.write_text(
            json.dumps([{"start": w["start"], "end": w["end"]} for w in vad_windows],
                       ensure_ascii=False),
            encoding="utf-8",
        )
    stats = seg.speech_stats(vad_windows)

    print(f"\n=== ASR BENCH [{args.tag}] ===")
    print(f"audio={args.audio}  dur={dur:.1f}s  speech_regions={stats['num_speech_regions']}  "
          f"speech={stats['speech_seconds']:.1f}s")
    print(f"model={cfg.asr.model_size} device={cfg.asr.device} compute={cfg.asr.compute_type} "
          f"beam={cfg.asr.beam_size} best_of={cfg.asr.best_of} temps={cfg.asr.temperatures} "
          f"cond={cfg.asr.condition_on_previous_text} no_speech={cfg.asr.no_speech_threshold} "
          f"logprob={cfg.asr.log_prob_threshold} compression={cfg.asr.compression_ratio_threshold} "
          f"initial_prompt={'YES' if cfg.asr.initial_prompt else 'no'}")

    run_texts = []
    active_device = "?"
    first_segments = None
    first_raw = None
    timings = []

    for run_i in range(1, max(1, args.runs) + 1):
        engine = ASREngine(cfg.asr, cfg.vad, cfg.models_dir_abs.as_posix())
        t0 = time.time()
        try:
            result = engine.transcribe(audio, vad_windows=copy.deepcopy(vad_windows))
            active_device = engine.active_device or "?"
        finally:
            engine.unload()
        elapsed = time.time() - t0
        timings.append(elapsed)

        segments = result.get("segments", [])
        raw = _raw_text(segments)
        # Faithful postprocess (same as production) — operate on a copy.
        norm = normalize_transcript({"segments": copy.deepcopy(segments)}, cfg.postprocess)
        final = _raw_text(norm.get("segments", []))
        run_texts.append(final)

        rtf = elapsed / dur if dur else 0.0
        print(f"  run{run_i}: {elapsed:.1f}s (RTF {rtf:.2f}x)  device={active_device}  "
              f"segments={len(segments)}  words={len(final.split())}")

        if run_i == 1:
            first_segments = segments
            first_raw = raw

    # Determinism check
    stable = all(t == run_texts[0] for t in run_texts)
    print(f"  determinism: {'STABLE' if stable else 'UNSTABLE across runs'}  "
          f"(avg {sum(timings)/len(timings):.1f}s)")

    # ---- persist ----
    (out_dir / f"{args.tag}.raw.txt").write_text(first_raw or "", encoding="utf-8")
    (out_dir / f"{args.tag}.txt").write_text(run_texts[0], encoding="utf-8")
    if len(run_texts) >= 2:
        (out_dir / f"{args.tag}.run2.txt").write_text(run_texts[1], encoding="utf-8")
    (out_dir / f"{args.tag}.segments.json").write_text(
        json.dumps(
            [{"start": s.get("start"), "end": s.get("end"),
              "text": s.get("text"), "avg_logprob": s.get("avg_logprob")}
             for s in (first_segments or [])],
            ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    meta = {
        "tag": args.tag,
        "audio": args.audio,
        "duration_s": round(dur, 2),
        "speech_regions": stats["num_speech_regions"],
        "speech_seconds": stats["speech_seconds"],
        "active_device": active_device,
        "runs": args.runs,
        "timings_s": [round(t, 2) for t in timings],
        "avg_rtf": round((sum(timings) / len(timings)) / dur, 3) if dur else None,
        "stable_across_runs": stable,
        "n_words_final": len(run_texts[0].split()),
        "asr": {
            "model_size": cfg.asr.model_size,
            "device": cfg.asr.device,
            "compute_type": cfg.asr.compute_type,
            "beam_size": cfg.asr.beam_size,
            "best_of": cfg.asr.best_of,
            "patience": cfg.asr.patience,
            "temperatures": cfg.asr.temperatures,
            "condition_on_previous_text": cfg.asr.condition_on_previous_text,
            "compression_ratio_threshold": cfg.asr.compression_ratio_threshold,
            "log_prob_threshold": cfg.asr.log_prob_threshold,
            "no_speech_threshold": cfg.asr.no_speech_threshold,
            "initial_prompt": cfg.asr.initial_prompt,
            "suppress_numerals": cfg.asr.suppress_numerals,
            "batch_size": cfg.asr.batch_size,
        },
    }
    (out_dir / f"{args.tag}.meta.json").write_text(
        json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"  saved -> {out_dir}/{args.tag}.{{txt,raw.txt,segments.json,meta.json}}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
