# -*- coding: utf-8 -*-
"""Focused check for the ASR-error words called out in the correction upgrade.

Runs the per-segment corrector (``correct_segment``) against the live local LLM
provider on the specific garbled words the prompt upgrade must fix, and on a few
already-correct control sentences to confirm no regression. Prints a compact
before/after report. Uses the SAME config, prompt and pipeline as production —
only a targeted input set.

Usage (from the project root)::

    ./venv/Scripts/python.exe scripts/test_llm_words.py
"""
from __future__ import annotations

import os
import sys
from dataclasses import replace

# Make the project root importable when run as a script.
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Windows consoles default to a legacy codepage that cannot encode Persian;
# force UTF-8 so the before/after report prints.
for _stream in (sys.stdout, sys.stderr):
    try:
        _stream.reconfigure(encoding="utf-8")
    except Exception:
        pass

from core.config import load_config
from modules.postprocess.llm_correct import correct_segment

ZWNJ = "‌"

# (target word, sentence containing a realistically garbled ASR version).
# These are exactly the recognition-error classes the upgraded prompt targets:
# missing letter, trailing/echo letter, wrong word boundary, duplicated letter,
# missing نیم‌فاصله, similar-sounding word.
ERROR_CASES = [
    ("معدن",          "استخراج از این مدن مس در جنوب کشور آغاز شد."),      # مدن -> معدن
    ("صنایع",         "این گزارش درباره‌ی صنایعه بزرگ کشور است."),        # صنایعه -> صنایع
    ("شبکه",          "پایداری شبکمون سراسری برق اهمیت زیادی دارد."),      # شبکمون -> شبکه
    ("محقق",          "این محققق در دانشگاه تهران مشغول پژوهش است."),      # محققق -> محقق
    ("تنگه" + ZWNJ + "ی", "کشتی‌ها از تنگهی هرمز عبور می‌کنند."),          # تنگهی -> تنگه‌ی
    ("تبریک",         "این پیام را برای تبلیک سال نو فرستاد."),           # تبلیک -> تبریک
]

# Already-correct sentences: the corrector must NOT damage these.
CONTROL_CASES = [
    "امروز هوا بسیار خوب است.",
    "قیمت نفت در بازار جهانی افزایش یافت.",
    "او به دانشگاه رفت تا درس بخواند.",
]


def _norm(s: str) -> str:
    """Loose comparison: ignore ZWNJ and spacing differences."""
    return (s or "").replace(ZWNJ, "").replace(" ", "")


def main() -> int:
    cfg = replace(load_config().llm_postprocess, enabled=True)
    print("provider        :", cfg.provider)
    print("model           :", cfg.model if cfg.provider == "ollama" else cfg.openrouter_model)
    print("endpoint        :", cfg.endpoint)
    print("num_ctx         :", cfg.num_ctx)
    print("num_predict     :", cfg.num_predict)
    print("keep_alive      :", cfg.keep_alive)
    print()

    errors = 0

    print("=" * 78)
    print("ASR ERROR WORDS  (must be corrected)")
    print("=" * 78)
    recovered = 0
    for target, garbled in ERROR_CASES:
        out = correct_segment(garbled, cfg)
        after = out["text_corrected"]
        status = out["status"]
        errors += int(status == "error")
        ok = _norm(target) in _norm(after)
        recovered += int(ok)
        print(f"\n  target      : {target}")
        print(f"  before      : {garbled}")
        print(f"  after       : {after}")
        print(f"  status      : {status}")
        print(f"  recovered   : {'YES' if ok else 'no'}")

    print("\n" + "=" * 78)
    print("ALREADY-CORRECT CONTROLS  (must NOT be damaged)")
    print("=" * 78)
    intact = 0
    for sentence in CONTROL_CASES:
        out = correct_segment(sentence, cfg)
        after = out["text_corrected"]
        status = out["status"]
        errors += int(status == "error")
        # "not damaged" = identical ignoring ZWNJ/spacing (punctuation/نیم‌فاصله
        # tidying is allowed and does not count as damage).
        same = _norm(after) == _norm(sentence)
        intact += int(same)
        print(f"\n  before      : {sentence}")
        print(f"  after       : {after}")
        print(f"  status      : {status}")
        print(f"  undamaged   : {'YES' if same else 'no'}")

    print("\n" + "=" * 78)
    if errors:
        print(f"!! {errors} call(s) errored — the LLM provider looks unreachable at")
        print(f"!! {cfg.endpoint}. Fallback safety still holds (after == before).")
        print("=" * 78)
    print(f"recovered {recovered}/{len(ERROR_CASES)} error words; "
          f"{intact}/{len(CONTROL_CASES)} controls undamaged")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
