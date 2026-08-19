"""Unit tests for faithful Persian post-processing (regex/table only, no LLM).

The hard rule under test: normalization must clean glyphs/spacing/ASR-loops
**without** adding, removing, or rewriting real words.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from core.config import PostprocessSection  # noqa: E402
from modules.postprocess.persian import (  # noqa: E402
    ZWNJ,
    normalize_text,
    normalize_transcript,
)


def _cfg(**overrides) -> PostprocessSection:
    cfg = PostprocessSection()
    for k, v in overrides.items():
        setattr(cfg, k, v)
    return cfg


def test_arabic_to_persian_glyphs():
    # ك (Arabic kaf) -> ک, ي (Arabic yeh) -> ی. Same word, correct glyphs.
    assert normalize_text("كتاب", _cfg()) == "کتاب"
    assert normalize_text("يک", _cfg()) == "یک"
    # Teh marbuta and hamza-alef variants normalize too.
    assert normalize_text("ة", _cfg()) == "ه"
    assert normalize_text("أحمد", _cfg()) == "احمد"


def test_arabic_normalization_can_be_disabled():
    out = normalize_text("كتاب", _cfg(arabic_to_persian=False, fix_zwnj=False,
                                      normalize_punctuation=False, collapse_repeats=False))
    assert out == "كتاب"  # untouched


def test_collapse_asr_repetition_loops():
    # A word repeated >=3x in a row is an ASR loop -> collapse to one.
    assert normalize_text("خوب خوب خوب خوب", _cfg()) == "خوب"


def test_natural_doubling_is_preserved():
    # 2x repetition is natural emphasis, NOT a loop -> keep verbatim.
    assert normalize_text("خیلی خیلی ممنون", _cfg()) == "خیلی خیلی ممنون"


def test_punctuation_normalization_and_spacing():
    assert normalize_text("سلام ؟", _cfg()) == "سلام؟"          # space before punct removed
    assert normalize_text("خوب ?", _cfg()) == "خوب؟"            # latin ? -> persian ؟
    assert normalize_text("یک  دو", _cfg()) == "یک دو"          # collapse double space


def test_zwnj_spacing_tidy():
    # A stray space hugging a ZWNJ is removed; the ZWNJ itself is kept.
    assert normalize_text(f"می{ZWNJ} خواهم", _cfg()) == f"می{ZWNJ}خواهم"


def test_persian_digits_off_by_default():
    assert normalize_text("۱۲۳ و 123", _cfg()) == "۱۲۳ و 123"          # kept as-is
    assert normalize_text("123", _cfg(persian_digits=True)) == "۱۲۳"   # opt-in only


def test_faithfulness_word_count_preserved():
    # No Arabic glyphs, no loops: a normal sentence is returned verbatim.
    sentence = "من به مدرسه رفتم و درس خواندم"
    out = normalize_text(sentence, _cfg())
    assert out == sentence
    assert len(out.split()) == len(sentence.split())


def test_disabled_leaves_transcript_verbatim():
    result = {"segments": [{"text": "كتاب خوب خوب خوب"}]}
    normalize_transcript(result, _cfg(enabled=False))
    assert result["segments"][0]["text"] == "كتاب خوب خوب خوب"


def test_normalize_transcript_mutates_segments_and_words():
    result = {
        "segments": [
            {
                "text": "كتاب كتاب كتاب",
                "words": [
                    {"word": "كتاب", "start": 0.0, "end": 0.4, "score": 0.9},
                    {"word": "كتاب", "start": 0.4, "end": 0.8, "score": 0.8},
                    {"word": "كتاب", "start": 0.8, "end": 1.2, "score": 0.7},
                ],
            }
        ]
    }
    normalize_transcript(result, _cfg())
    seg = result["segments"][0]
    assert seg["text"] == "کتاب"                      # glyphs fixed + loop collapsed
    assert len(seg["words"]) == 1                     # 3 identical words merged
    assert seg["words"][0]["word"] == "کتاب"
    assert seg["words"][0]["start"] == 0.0            # merged span keeps first start
    assert seg["words"][0]["end"] == 1.2              # ...and last end


if __name__ == "__main__":
    import traceback
    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    ok = 0
    for fn in fns:
        try:
            fn(); print(f"PASS {fn.__name__}"); ok += 1
        except Exception:
            print(f"FAIL {fn.__name__}"); traceback.print_exc()
    print(f"\n{ok}/{len(fns)} passed")
    sys.exit(0 if ok == len(fns) else 1)
