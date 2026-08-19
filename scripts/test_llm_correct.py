# -*- coding: utf-8 -*-
"""Manual test / demo for the optional LLM post-processing layer.

Runs the per-segment corrector against the live local Ollama server on a
transcript full of known Persian ASR error patterns, and prints a before/after
report. Also proves the ``enabled: false`` path is a byte-identical no-op.

Usage (from the project root)::

    ./venv/Scripts/python.exe scripts/test_llm_correct.py
"""
from __future__ import annotations

import copy
import json
import os
import sys
from dataclasses import replace

# Make the project root importable when run as a script.
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Windows consoles default to a legacy codepage (e.g. cp1256) that cannot encode
# Persian; force UTF-8 so the before/after report prints. (Production transcript
# output already writes UTF-8 in modules/storage/transcript.py.)
for _stream in (sys.stdout, sys.stderr):
    try:
        _stream.reconfigure(encoding="utf-8")
    except Exception:
        pass

from core.config import load_config
from modules.postprocess.llm_correct import correct_transcript
from modules.storage import transcript as tx

ZWNJ = "‌"

# Each case: the correct target word/phrase, and a segment containing a
# realistically garbled version an ASR system would produce (homophone
# confusions in the classes ص/س/ث, ز/ذ/ض/ظ, غ/ق, ع/ا/ح/ه, dropped letters, etc.).
TEST_CASES = [
    ("صنایع",              "این گزارش درباره‌ی سنایع بزرگ کشور است."),
    ("تشکیل شد",           "جلسه‌ی مهمی برای بررسی موضوع تشکیل شت."),
    ("کمترین آسیب",        "تلاش کردیم کمترین اسیب به مردم وارد شود."),
    ("محدودیت",            "دولت برای مصرف برق محدودیط تازه‌ای گذاشت."),
    ("اولویت",             "این موضوع برای ما حولویت اول دارد."),
    ("شبکه",               "پایداری شبکح سراسری برق اهمیت زیادی دارد."),
    ("اعمال",              "این قانون از فردا اامال خواهد شد."),
    ("نیروگاه",            "یک نیرگاه تازه در جنوب کشور افتتاح شد."),
    ("الگوی مصرف",         "الگوی مسرف مردم در تابستان تغییر کرد."),
    ("تصمیمات",            "در این نشست تسمیمات مهمی گرفته شد."),
    ("ابلاغ",              "این دستور به همه‌ی استان‌ها ابلاق شد."),
    ("شهرک‌های صنعتی",     "برق شهرکهای سنعتی در ساعات اوج مصرف مدیریت می‌شود."),
    ("خبرگزاری",           "به گزارش خبرگذاری رسمی این تصمیم اعلام شد."),
]


def _fake_words(text: str, start: float, end: float):
    """Cheap word-level timing so we can prove alignment is left intact."""
    toks = text.split()
    if not toks:
        return []
    step = (end - start) / len(toks)
    return [
        {"word": t, "start": round(start + i * step, 3),
         "end": round(start + (i + 1) * step, 3), "score": 0.9}
        for i, t in enumerate(toks)
    ]


def build_result():
    segments = []
    for i, (_, garbled) in enumerate(TEST_CASES):
        start = round(i * 4.0, 3)
        end = round(i * 4.0 + 3.5, 3)
        segments.append({
            "speaker": "SPEAKER_00",
            "start": start,
            "end": end,
            "text": garbled,
            "words": _fake_words(garbled, start, end),
        })
    return {"language": "fa", "segments": segments}


def _norm(s: str) -> str:
    """Loose comparison: ignore ZWNJ and spacing differences."""
    return s.replace(ZWNJ, "").replace(" ", "")


def main() -> int:
    cfg = load_config().llm_postprocess
    print("Ollama endpoint :", cfg.endpoint)
    print("Model           :", cfg.model)
    print("temperature     :", cfg.temperature)
    print("timeout_seconds :", cfg.timeout_seconds)
    print("max_word_change_ratio:", cfg.max_word_change_ratio)
    print()

    # ---------------------------------------------------------------- ENABLED
    cfg_on = replace(cfg, enabled=True)
    result = build_result()
    snapshot = copy.deepcopy(result)  # to verify timestamps/words are untouched

    stats = correct_transcript(result, cfg_on)

    if stats["total"] and stats["errors"] == stats["total"]:
        print("!! Every call errored — the Ollama server appears unreachable at")
        print("!!", cfg.endpoint, "— corrections could not be produced.")
        print("!! (Fallback safety still verified below: after == before.)\n")

    print("=" * 78)
    print("BEFORE / AFTER  (text = original ASR, text_corrected = LLM output)")
    print("=" * 78)
    recovered = 0
    for i, (expected, _) in enumerate(TEST_CASES):
        seg = result["segments"][i]
        before = seg["text"]
        after = seg["text_corrected"]
        ok = _norm(expected) in _norm(after)
        recovered += int(ok)
        print(f"\n[{i+1:02d}] target word : {expected}")
        print(f"     before      : {before}")
        print(f"     after       : {after}")
        print(f"     recovered ‘{expected}’ : {'YES' if ok else 'no'}"
              f"   (segment text changed: {'yes' if after != before else 'no'})")

    # -------------------------------------------------- integrity assertions
    print("\n" + "=" * 78)
    print("TIMESTAMP / ALIGNMENT INTEGRITY")
    print("=" * 78)
    intact = True
    for i, seg in enumerate(result["segments"]):
        base = snapshot["segments"][i]
        same = (seg["start"] == base["start"] and seg["end"] == base["end"]
                and seg["words"] == base["words"] and seg["text"] == base["text"])
        intact = intact and same
    print("start/end/words/original-text unchanged for ALL segments:",
          "PASS" if intact else "FAIL")

    print("\n" + "=" * 78)
    print("SIMILARITY-GUARD & OUTCOME STATS")
    print("=" * 78)
    total = stats["total"] or 1
    for k in ("total", "corrected", "unchanged", "rejected_similarity",
              "rejected_format", "errors", "empty"):
        pct = 100.0 * stats[k] / total if k != "total" else 100.0
        print(f"  {k:20s}: {stats[k]:3d}   ({pct:5.1f}%)")
    print(f"\n  recovered target word in {recovered}/{len(TEST_CASES)} segments")

    # --------------------------------------------------------------- DISABLED
    print("\n" + "=" * 78)
    print("DISABLED PATH  (enabled: false must be a byte-identical no-op)")
    print("=" * 78)
    cfg_off = replace(cfg, enabled=False)
    raw = build_result()

    baseline_json = json.dumps(tx.assemble("vid", copy.deepcopy(raw)),
                               ensure_ascii=False, sort_keys=True)

    off_result = build_result()
    off_stats = correct_transcript(off_result, cfg_off)
    disabled_json = json.dumps(tx.assemble("vid", copy.deepcopy(off_result)),
                               ensure_ascii=False, sort_keys=True)

    no_field = all("text_corrected" not in s for s in off_result["segments"])
    identical = baseline_json == disabled_json
    print("correct_transcript() reported enabled =", off_stats["enabled"])
    print("no 'text_corrected' field added to any segment :",
          "PASS" if no_field else "FAIL")
    print("assembled JSON identical to no-LLM baseline     :",
          "PASS" if identical else "FAIL")

    ok = intact and no_field and identical
    print("\nRESULT:", "ALL INTEGRITY CHECKS PASSED" if ok else "CHECK FAILED")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
