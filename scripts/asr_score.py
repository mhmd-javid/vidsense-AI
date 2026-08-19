#!/usr/bin/env python
"""Deterministic ASR error scorer for the known Persian test clip (md5 e6e8720e…).

Because every benchmark run transcribes the SAME clip (a formal news report on
industrial electricity restrictions), we can score each transcript against a
FIXED rubric of that clip's actual content — applied identically to every
config. Two complementary, fully deterministic signals:

  1. ERROR-FORM COUNT (primary "word error count"): occurrences of known-wrong
     renderings, grouped into the mission's categories. Lower = better.
  2. CORRECT-FORM PRESENCE: whether the correct rendering appears (coverage).

Plus reference-free metrics that need no rubric: repetition/hallucination runs,
Latin-script (non-Persian) intrusions, token fragmentation, digit tokens.

The rubric's "correct" forms are reconstructed from the clip's context (a
coherent news report); this is an expert reconstruction, not an audio-verified
transcript — see the report's "limitations". Matching is on normalized text
(Arabic→Persian glyphs unified, ZWNJ/diacritics/tatweel stripped, digits folded
to Latin, whitespace collapsed) with a space-insensitive fallback so spacing
errors don't create false matches.

Usage:
  python scripts/asr_score.py data/benchmark/A_small_baseline.txt data/benchmark/B_medium_baseline.txt
"""
from __future__ import annotations

import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


def _utf8_stdout() -> None:
    for stream in (sys.stdout, sys.stderr):
        try:
            stream.reconfigure(encoding="utf-8")  # type: ignore[attr-defined]
        except Exception:
            pass


# --- normalization -----------------------------------------------------------
_AR2FA = str.maketrans({"ي": "ی", "ى": "ی", "ك": "ک", "ة": "ه", "أ": "ا",
                        "إ": "ا", "ٱ": "ا", "ؤ": "و", "ئ": "ی"})
_FA_DIGITS = "۰۱۲۳۴۵۶۷۸۹"
_AR_DIGITS = "٠١٢٣٤٥٦٧٨٩"
_LAT_DIGITS = "0123456789"
_DIGIT_MAP = str.maketrans(_FA_DIGITS + _AR_DIGITS, _LAT_DIGITS + _LAT_DIGITS)
_DIACRITICS = re.compile(r"[ً-ٰٟـ]")  # harakat + tatweel
_ZWNJ = "‌"
_LATIN = re.compile(r"[A-Za-z]")


def normalize(text: str) -> str:
    text = text.translate(_AR2FA).translate(_DIGIT_MAP)
    text = _DIACRITICS.sub("", text).replace(_ZWNJ, " ")
    text = re.sub(r"[.،؛؟?!:\"'()»«]", " ", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def nospace(text: str) -> str:
    return re.sub(r"\s+", "", text)


# --- fixed rubric: (category, correct_form, [wrong_forms]) -------------------
# correct_form / wrong_forms are given in normalized Persian. A "hit" for the
# correct form and each wrong-form occurrence are counted on the normalized text.
RUBRIC = [
    # ---- technical / domain terms ----
    ("tech", "صنعت", ["صناعت"]),
    ("tech", "صنعتی", ["سنایه", "سنائی", "سنعتی", "صناعی"]),
    ("tech", "معدن", ["مدن", "مدر", "وادر"]),
    ("tech", "برق", ["برگ", "برغ"]),
    ("tech", "انرژی", ["اینرژی", "اینرجی", "اینجیبر", "اینجی"]),
    ("tech", "شبکه", ["شبک "]),
    ("tech", "بورس", ["بورز"]),
    ("tech", "تعطیل", ["تحطیل", "تحتل"]),
    ("tech", "اولویت", ["حولویت", "اولبیت"]),
    ("tech", "مصرف", ["مسرف"]),
    ("tech", "الگوی", ["اولگوی", "اولگو"]),
    ("tech", "تصمیمات", ["تسمیمات", "تسمیماد"]),
    ("tech", "هماهنگ", ["همه هنگ", "هما هنگ", "همه همه هنگ"]),
    ("tech", "تأمین", ["تمین", "تبان"]),
    ("tech", "اعمال", ["ایمال"]),
    ("tech", "اقتصاد", ["اختصاد"]),
    ("tech", "محصول", ["محسول"]),
    ("tech", "عرضه", ["ارزه"]),
    # ---- org / person / proper names ----
    ("name", "وزارت نیرو", ["وزاعت نیرو", "وزارت میرو", "و زرط نیرو", "نیروب", "نیروح", "نیروگی"]),
    ("name", "رئیس جمهور", []),
    ("name", "خبرگزاری صدا و سیما", ["خبرگوزوری", "استدا", "آسیما", "خبرگزری"]),
    ("name", "صمت", ["وزیر سمت", "وزارت سمت", "وزیر سند", "وزارت سند"]),
    ("name", "کمیته", ["کومیته", "کومیتوی", "کومتوی", "کومتهی", "کومتی", "کومنتی"]),
    ("name", "وزیر", ["وزید"]),
    ("name", "نمایندگان", ["نوائندگان", "نبایندگان"]),
    # ---- numbers ----
    ("num", "80", ["8 درسد", "8 درصد"]),
    ("num", "47", ["چلو هف", "حولوش"]),
    ("num", "24", []),
    ("num", "85", ["حشتا دو پنی"]),
    # ---- common words ----
    ("common", "می گوید", ["میگویت", "می گویت"]),
    ("common", "طبق", ["تبق", "تبقه"]),
    ("common", "علی رغم", ["علا رقم", "الارغم", "الان رقم"]),
    ("common", "خرداد", ["خورداد"]),
    ("common", "گفت", ["گوف"]),
    ("common", "بزرگ", ["بزرک", "بزور"]),
    ("common", "نسبت", ["نصفت"]),
    ("common", "آسیب", ["آسی ", "ااسیب"]),
    ("common", "قطع", ["قطر برغ", "قتن", "قاد"]),
]

CATS = [("tech", "technical/domain"), ("name", "org/person names"),
        ("num", "numbers"), ("common", "common words")]


def count_occ(hay: str, needle: str) -> int:
    """Count occurrences on normalized text, with a space-insensitive fallback."""
    n = hay.count(needle)
    if n == 0 and " " in needle:
        n = nospace(hay).count(nospace(needle))
    return n


def score_file(path: Path) -> dict:
    raw = path.read_text(encoding="utf-8")
    norm = normalize(raw)
    tokens = norm.split()

    # rubric scoring
    cat_err = {c: 0 for c, _ in CATS}
    cat_correct = {c: 0 for c, _ in CATS}
    cat_total = {c: 0 for c, _ in CATS}
    misses = []
    for cat, correct, wrongs in RUBRIC:
        cat_total[cat] += 1
        has_correct = count_occ(norm, normalize(correct)) > 0
        if has_correct:
            cat_correct[cat] += 1
        werr = sum(count_occ(norm, normalize(w)) for w in wrongs)
        cat_err[cat] += werr
        if not has_correct:
            misses.append(f"{correct}({cat})")

    # reference-free metrics on the SAME normalized tokens
    max_run, cur = 1, 1
    runs3 = 0
    for i in range(1, len(tokens)):
        if tokens[i] == tokens[i - 1]:
            cur += 1
            max_run = max(max_run, cur)
        else:
            if cur >= 3:
                runs3 += 1
            cur = 1
    if cur >= 3:
        runs3 += 1
    latin = len(_LATIN.findall(raw))
    frag = sum(1 for t in tokens if len(t) <= 1)  # 1-char fragments (spacing breakage)
    digits = [t for t in tokens if re.fullmatch(r"\d+%?", t)]

    return {
        "path": path.name,
        "tokens": len(tokens),
        "cat_err": cat_err,
        "cat_correct": cat_correct,
        "cat_total": cat_total,
        "total_err": sum(cat_err.values()),
        "total_correct": sum(cat_correct.values()),
        "total_rubric": sum(cat_total.values()),
        "max_repeat_run": max_run,
        "repeat_runs_ge3": runs3,
        "latin_chars": latin,
        "frag_1char": frag,
        "digits": digits,
        "misses": misses,
    }


def main(argv=None) -> int:
    _utf8_stdout()
    paths = [Path(p) for p in (argv or sys.argv[1:])]
    if not paths:
        print("usage: asr_score.py <transcript.txt> [more.txt ...]")
        return 2
    results = [score_file(p) for p in paths if p.exists()]

    # ---- comparison table ----
    print("\n" + "=" * 100)
    print("ERROR-FORM COUNTS BY CATEGORY  (lower = better)")
    print("=" * 100)
    hdr = f"{'config':<32}" + "".join(f"{lbl[:14]:>16}" for _, lbl in CATS) + f"{'TOTAL_ERR':>12}"
    print(hdr)
    print("-" * 100)
    for r in results:
        row = f"{r['path']:<32}"
        for c, _ in CATS:
            row += f"{r['cat_err'][c]:>16}"
        row += f"{r['total_err']:>12}"
        print(row)

    print("\n" + "=" * 100)
    print("CORRECT-FORM COVERAGE  (correct/total rubric items; higher = better)")
    print("=" * 100)
    print(f"{'config':<32}" + "".join(f"{lbl[:14]:>16}" for _, lbl in CATS) + f"{'TOTAL':>12}")
    print("-" * 100)
    for r in results:
        row = f"{r['path']:<32}"
        for c, _ in CATS:
            row += f"{str(r['cat_correct'][c]) + '/' + str(r['cat_total'][c]):>16}"
        row += f"{str(r['total_correct']) + '/' + str(r['total_rubric']):>12}"
        print(row)

    print("\n" + "=" * 100)
    print("REFERENCE-FREE METRICS")
    print("=" * 100)
    print(f"{'config':<32}{'tokens':>10}{'maxRepeat':>11}{'runs>=3':>9}{'latinCh':>9}{'1charFrag':>11}{'#digits':>9}")
    print("-" * 100)
    for r in results:
        print(f"{r['path']:<32}{r['tokens']:>10}{r['max_repeat_run']:>11}{r['repeat_runs_ge3']:>9}"
              f"{r['latin_chars']:>9}{r['frag_1char']:>11}{len(r['digits']):>9}")
    for r in results:
        print(f"  {r['path']} digits: {r['digits']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
