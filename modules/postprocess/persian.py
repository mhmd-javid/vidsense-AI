"""Faithful Persian text normalization — cleaning only, **never** rewriting.

Hard rule (from the project brief): post-processing must NOT generate new
information, must NOT use an LLM, and must NOT rewrite sentences. The output
stays faithful to what was said. Everything here is a deterministic,
meaning-preserving transform:

  * ``arabic_to_persian`` — swap Arabic code points for their Persian
    equivalents (ي→ی, ك→ک, ة→ه, أ/إ→ا, …). Same word, correct Persian glyphs.
  * ``persian_digits`` — optionally unify digit glyphs to Persian (off by
    default: keep digits exactly as transcribed).
  * ``fix_zwnj`` — tidy existing zero-width non-joiners and stray spaces around
    them; never *inserts* a ZWNJ between words (that would be guessing).
  * ``normalize_punctuation`` — Latin→Persian punctuation (?→؟, ;→؛, ,→،) and
    spacing tidy-up (no space *before* punctuation, collapse runs of spaces).
  * ``collapse_repeats`` — collapse a word repeated **≥3× in a row** to one.
    That length targets ASR repetition loops; natural doubling ("خیلی خیلی")
    is left untouched so we don't alter genuine speech.

No third-party NLP dependency — pure regex/table so it is fast and testable.
"""
from __future__ import annotations

import re
from typing import Any, Dict, List

from core.config import PostprocessSection
from core.utils import get_logger

logger = get_logger(__name__)

ZWNJ = "‌"

# Runs of an identical word this long (or longer) are treated as an ASR loop.
REPEAT_RUN = 3

# Arabic → Persian glyph normalization (meaning-preserving).
_ARABIC_TO_PERSIAN = {
    "ي": "ی",  # ي ARABIC YEH        -> ی PERSIAN YEH
    "ى": "ی",  # ى ALEF MAKSURA      -> ی
    "ك": "ک",  # ك ARABIC KAF        -> ک PERSIAN KEHEH
    "ة": "ه",  # ة TEH MARBUTA       -> ه HEH
    "أ": "ا",  # أ ALEF W/ HAMZA ABOVE -> ا ALEF
    "إ": "ا",  # إ ALEF W/ HAMZA BELOW -> ا
    "ٱ": "ا",  # ٱ ALEF WASLA        -> ا
    "ک": "ک",  # ک (identity, keep)
}

_ARABIC_INDIC_DIGITS = "٠١٢٣٤٥٦٧٨٩"      # U+0660..
_LATIN_DIGITS = "0123456789"
_PERSIAN_DIGITS = "۰۱۲۳۴۵۶۷۸۹"           # U+06F0..

_LATIN_TO_PERSIAN_PUNCT = {
    "?": "؟",
    ";": "؛",
    ",": "،",
}

# Persian/Arabic + shared punctuation we tidy spacing around.
_PUNCT_CHARS = "،؛؟!؟?.:"

_char_map = str.maketrans(_ARABIC_TO_PERSIAN)
_to_persian_digits_map = str.maketrans(
    _LATIN_DIGITS + _ARABIC_INDIC_DIGITS,
    _PERSIAN_DIGITS + _PERSIAN_DIGITS,
)
_punct_map = str.maketrans(_LATIN_TO_PERSIAN_PUNCT)

_ZWNJ_SPACES = re.compile(rf"[ \t]*{ZWNJ}[ \t]*")
_ZWNJ_DUP = re.compile(rf"{ZWNJ}{{2,}}")
_MULTISPACE = re.compile(r"[ \t]{2,}")
_SPACE_BEFORE_PUNCT = re.compile(rf"[ \t]+([{re.escape(_PUNCT_CHARS)}])")


def _map_chars(text: str) -> str:
    return text.translate(_char_map)


def _to_persian_digits(text: str) -> str:
    return text.translate(_to_persian_digits_map)


def _fix_zwnj(text: str) -> str:
    text = _ZWNJ_SPACES.sub(ZWNJ, text)   # drop spaces hugging a ZWNJ
    text = _ZWNJ_DUP.sub(ZWNJ, text)      # collapse repeated ZWNJ
    return text.strip(ZWNJ + " ")


def _normalize_punct(text: str) -> str:
    text = text.translate(_punct_map)
    text = _SPACE_BEFORE_PUNCT.sub(r"\1", text)
    text = _MULTISPACE.sub(" ", text)
    return text


def _collapse_runs(tokens: List[str]) -> List[str]:
    """Collapse runs of ``REPEAT_RUN``+ identical tokens down to one."""
    out: List[str] = []
    i, n = 0, len(tokens)
    while i < n:
        j = i
        while j < n and tokens[j] == tokens[i]:
            j += 1
        run = j - i
        if run >= REPEAT_RUN and tokens[i]:
            out.append(tokens[i])
        else:
            out.extend(tokens[i:j])
        i = j
    return out


def _collapse_repeats_text(text: str) -> str:
    tokens = text.split(" ")
    collapsed = _collapse_runs(tokens)
    return " ".join(collapsed)


def _collapse_repeats_words(words: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Collapse ≥REPEAT_RUN identical consecutive word entries, merging spans."""
    out: List[Dict[str, Any]] = []
    i, n = 0, len(words)
    while i < n:
        j = i
        while j < n and words[j].get("word") == words[i].get("word"):
            j += 1
        run = j - i
        if run >= REPEAT_RUN and words[i].get("word"):
            merged = dict(words[i])
            last = words[j - 1]
            if "end" in last:
                merged["end"] = last["end"]
            out.append(merged)
        else:
            out.extend(words[i:j])
        i = j
    return out


def normalize_text(text: str, cfg: PostprocessSection) -> str:
    """Apply the configured faithful normalizations to a single string."""
    if not text:
        return text
    if cfg.arabic_to_persian:
        text = _map_chars(text)
    if cfg.persian_digits:
        text = _to_persian_digits(text)
    if cfg.fix_zwnj:
        text = _fix_zwnj(text)
    if cfg.normalize_punctuation:
        text = _normalize_punct(text)
    if cfg.collapse_repeats:
        text = _collapse_repeats_text(text)
    return text.strip()


def _normalize_word_token(word: str, cfg: PostprocessSection) -> str:
    """Char-level normalization only (no punctuation/space rules on a token)."""
    if not word:
        return word
    if cfg.arabic_to_persian:
        word = _map_chars(word)
    if cfg.persian_digits:
        word = _to_persian_digits(word)
    if cfg.fix_zwnj:
        word = _fix_zwnj(word)
    return word


def normalize_transcript(result: Dict[str, Any], cfg: PostprocessSection) -> Dict[str, Any]:
    """Normalize every segment's text and word tokens in place. Returns *result*."""
    if not cfg.enabled:
        logger.info("Post-processing disabled — leaving transcript verbatim.")
        return result

    for seg in result.get("segments", []):
        seg["text"] = normalize_text(seg.get("text", ""), cfg)
        words = seg.get("words") or []
        for w in words:
            if w.get("word"):
                w["word"] = _normalize_word_token(w["word"], cfg)
        if cfg.collapse_repeats and words:
            seg["words"] = _collapse_repeats_words(words)
    return result
