"""End-to-end smoke test / demo driver.

Runs the full pipeline on a video and then asks a couple of questions via RAG,
printing timestamped answers. Defaults to the local sample video in
``data/videos`` so it works offline; pass ``--url`` to test downloading.

Usage:
    python tests/run_e2e.py                 # use local sample video
    python tests/run_e2e.py --url <URL>     # download then process
    python tests/run_e2e.py --q "..." --q "..."

Requires Ollama running with the configured model for the Q&A step.
"""
from __future__ import annotations

import argparse
import subprocess
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
try:
    sys.stdout.reconfigure(encoding="utf-8")  # print Persian on Windows consoles
except Exception:
    pass

from core.services import build_services  # noqa: E402


def vram_used_mib():
    try:
        out = subprocess.run(
            ["nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=10,
        )
        return int(out.stdout.strip().splitlines()[0])
    except Exception:
        return None


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--url", default=None, help="Video URL (YouTube/Aparat/...)")
    ap.add_argument("--q", action="append", default=[], help="Question (repeatable)")
    args = ap.parse_args()

    questions = args.q or [
        "موضوع اصلی این ویدیو چیست؟",
        "گوینده درباره چه چیزی صحبت می‌کند؟",
    ]

    print("=" * 70)
    print("VideoAI — end-to-end test")
    print("=" * 70)
    svc = build_services()
    print(f"Config: ASR={svc.cfg.asr.model_size}/{svc.cfg.asr.device}  "
          f"EMB={svc.cfg.embedding.model_name}@{svc.cfg.embedding.device}  "
          f"LLM={svc.cfg.llm.model}")
    print(f"VRAM baseline: {vram_used_mib()} MiB")
    print(f"Ollama available: {svc.llm.is_available()} | "
          f"model present: {svc.llm.model_available()}")

    def cb(stage, msg, frac):
        pct = f" ({frac*100:.0f}%)" if frac is not None else ""
        print(f"   · [{stage}] {msg}{pct}")

    # ---- Process -----------------------------------------------------------
    print("\n[1/2] Processing video…")
    t0 = time.time()
    if args.url:
        res = svc.processing.process_url(args.url, progress_cb=cb)
    else:
        video = next(Path(svc.cfg.videos_dir_abs).glob("*.*"), None)
        if not video:
            print("No local video found in data/videos. Provide --url instead.")
            return 1
        print(f"   Using local file: {video.name}")
        res = svc.processing.process_upload(video.read_bytes(), video.name, progress_cb=cb)
    proc_secs = time.time() - t0

    if not res.success:
        print(f"\nFAILED: {res.error}")
        return 1

    svc.rag.invalidate(res.video_id)
    print(f"\n   Processed in {proc_secs:.1f}s")
    print(f"   video_id={res.video_id} title={res.title!r}")
    print(f"   language={res.language} duration={res.duration:.1f}s "
          f"segments={res.num_segments} chunks={res.num_chunks} asr_device={res.asr_device}")
    print(f"   VRAM after processing: {vram_used_mib()} MiB")

    # ---- Chat --------------------------------------------------------------
    print("\n[2/2] Asking questions…")
    for q in questions:
        print("\n" + "-" * 60)
        print(f"Q: {q}")
        t1 = time.time()
        ans = svc.rag.answer(q, res.video_id)
        print(f"A ({time.time()-t1:.1f}s): {ans.answer}")
        print(f"grounded={ans.grounded}  citations:")
        for c in ans.citations:
            print(f"   ({c.label}) score={c.score:.2f} :: {c.text[:90]}")
    print(f"\nVRAM during/after chat: {vram_used_mib()} MiB")
    print("\nDONE.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
