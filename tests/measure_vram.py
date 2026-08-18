"""Measure real GPU VRAM usage of the LLM stage in isolation.

The 4 GB-VRAM design keeps ASR (CPU) and embeddings (CPU) off the GPU, so the
only heavy GPU consumer is the LLM (Ollama). This script:
  1. unloads the model, samples baseline VRAM (desktop only),
  2. loads it via one chat call, samples again,
and reports the delta. Run with Ollama up.

Usage:  python tests/measure_vram.py
"""
from __future__ import annotations

import subprocess
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from core.services import build_services  # noqa: E402


def vram_used_mib():
    try:
        out = subprocess.run(
            ["nvidia-smi", "--query-gpu=memory.used,memory.total", "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=10,
        )
        used, total = out.stdout.strip().splitlines()[0].split(",")
        return int(used), int(total)
    except Exception as exc:
        print("nvidia-smi unavailable:", exc)
        return None, None


def main():
    svc = build_services()
    if not svc.llm.is_available():
        print("Ollama not reachable — start it first.")
        return 1

    print(f"LLM model: {svc.llm.model}")
    print("Unloading LLM to establish a clean baseline…")
    svc.llm.unload()
    time.sleep(8)  # Ollama evicts asynchronously
    base_used, total = vram_used_mib()
    print(f"  Baseline VRAM (no LLM): {base_used} / {total} MiB")

    print("Loading LLM via one chat call…")
    _ = svc.llm.chat([{"role": "user", "content": "سلام"}], max_tokens=16)
    time.sleep(2)
    load_used, _ = vram_used_mib()
    print(f"  VRAM with LLM loaded : {load_used} / {total} MiB")

    if base_used is not None and load_used is not None:
        print(f"\n==> LLM VRAM footprint ≈ {load_used - base_used} MiB "
              f"(headroom to {total} MiB: {total - load_used} MiB)")
    print("\nNote: ASR and embeddings run on CPU in this build → 0 MiB VRAM,")
    print("so ASR + embeddings + LLM are never GPU-resident together.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
