"""Optional per-segment LLM spell/word correction — faithful and reversible.

Pipeline position (added stage, everything else untouched)::

    ASR -> Persian normalization -> [THIS] LLM correction -> confidence -> output

Runs ONLY when ``llm_postprocess.enabled`` is true. It never participates in ASR
inference; it only touches text that already exists.

Design contract — enforced in *code*, not merely requested in the prompt, because
a 3B model will not always obey instructions:

  * **Per segment.** Each segment's ``text`` is sent to the model on its own and
    the reply is stored as ``text_corrected``. ``text``, ``words``, ``start``,
    ``end``, speaker labels and alignment are never modified, so every segment's
    timestamp mapping stays exactly intact. Correcting the whole transcript at
    once could shift sentence boundaries and break that mapping — so we don't.
  * **Original is never overwritten.** ``text`` keeps the original ASR/normalized
    text; the correction lives beside it in ``text_corrected``. Both are always
    available for comparison and rollback.
  * **Similarity guard.** If more than ``max_word_change_ratio`` of the words
    changed, the reply is *discarded* and ``text_corrected`` falls back to the
    original. This catches the model rewriting/summarizing instead of correcting.
  * **Output validation.** A reply with a preamble, explanation, markdown, or
    multiple lines is treated as invalid -> fall back to the original text.
  * **Fault tolerance.** Timeout / HTTP error / Ollama unavailable -> keep the
    original text and continue. This stage never raises and never blocks the
    pipeline.

Transport is the Python standard library (``urllib.request``) against the local
Ollama HTTP API, so no new third-party dependency is introduced.
"""
from __future__ import annotations

import difflib
import json
import urllib.request
from typing import Any, Dict, List, Optional

from core.config import LLMPostprocessSection
from core.utils import get_logger

logger = get_logger(__name__)


# --------------------------------------------------------------------------- #
# Prompt
# --------------------------------------------------------------------------- #
# Strict instruction: the model must return ONLY the corrected segment text.
# The code still validates and guards the reply — this prompt just makes the
# common case behave.
_SYSTEM_PROMPT = (
    "You are a Persian (Farsi) spelling corrector for speech-to-text output. "
    "You receive ONE short Persian segment that may contain spelling mistakes, "
    "homophone confusions (ص/س/ث, ز/ذ/ض/ظ, "
    "غ/ق, ع/ا/ح/ه), or broken/garbled words.\n\n"
    "Your ONLY task:\n"
    "- Fix Persian spelling errors and obviously broken words.\n"
    "- Fix homophone confusions (e.g. سنایه→"
    "صنایع، حولویت"
    "→اولویت، نیرگ"
    "اه→نیروگاه).\n"
    "- Add only minimal, obvious punctuation for readability.\n\n"
    "You MUST NOT summarize, rewrite, rephrase, translate, or change the meaning; "
    "MUST NOT add or remove information or words; MUST NOT change the speaker's "
    "style; and MUST NOT output any explanation, comment, preamble, quotes, or "
    "markdown.\n\n"
    "Output ONLY the corrected Persian sentence as plain text, nothing else. "
    "If the input is already correct, return it unchanged."
)


# --------------------------------------------------------------------------- #
# Ollama transport (stdlib only)
# --------------------------------------------------------------------------- #
def _ollama_generate(cfg: LLMPostprocessSection, prompt: str) -> Optional[str]:
    """One deterministic ``/api/generate`` call. Returns the raw reply text, or
    ``None`` on any failure (timeout, connection refused, bad JSON, ...)."""
    url = cfg.endpoint.rstrip("/") + "/api/generate"
    body = {
        "model": cfg.model,
        "system": _SYSTEM_PROMPT,
        "prompt": prompt,
        "stream": False,
        "options": {"temperature": float(cfg.temperature)},
    }
    data = json.dumps(body).encode("utf-8")
    req = urllib.request.Request(
        url, data=data, headers={"Content-Type": "application/json"}, method="POST"
    )
    try:
        with urllib.request.urlopen(req, timeout=cfg.timeout_seconds) as resp:
            payload = json.loads(resp.read().decode("utf-8"))
        return payload.get("response", "")
    except Exception as exc:  # timeout / URLError / JSON / anything — never raise
        logger.warning(
            "LLM correction call failed (%s: %s) — keeping original text.",
            type(exc).__name__,
            exc,
        )
        return None


# --------------------------------------------------------------------------- #
# Reply validation
# --------------------------------------------------------------------------- #
# One layer of surrounding quotes/backticks the model may wrap around the text.
_QUOTE_PAIRS = [
    ('"', '"'),
    ("'", "'"),
    ("«", "»"),  # « »
    ("“", "”"),  # “ ”
    ("‘", "’"),  # ‘ ’
    ("`", "`"),
]


def _strip_wrapping(text: str) -> str:
    """Strip surrounding quote/backtick wrappers the model likes to add.

    This is pure formatting removal (not a meaning change): a reply of
    ``"صنایع"`` becomes ``صنایع``.
    """
    text = text.strip()
    changed = True
    while changed and len(text) >= 2:
        changed = False
        for lq, rq in _QUOTE_PAIRS:
            if text.startswith(lq) and text.endswith(rq) and len(text) > len(lq):
                text = text[len(lq): len(text) - len(rq)].strip()
                changed = True
                break
    return text


def _sanitize_reply(raw: Optional[str]) -> Optional[str]:
    """Return a clean single-line corrected sentence, or ``None`` if the reply
    looks like anything other than that (preamble, explanation, markdown block,
    multiple lines, empty)."""
    if not raw:
        return None
    text = raw.strip()

    # Unwrap a fenced code block if the whole reply is one.
    if text.startswith("```") and text.endswith("```") and len(text) >= 6:
        inner = text[3:-3]
        inner = inner.split("\n", 1)[-1] if "\n" in inner else inner
        text = inner.strip()

    # A clean corrected sentence is exactly one non-empty line. More than one
    # means the model added a preamble/explanation -> invalid.
    lines = [ln.strip() for ln in text.splitlines() if ln.strip()]
    if len(lines) != 1:
        return None

    line = _strip_wrapping(lines[0])
    if not line or "```" in line:
        return None
    return line


# --------------------------------------------------------------------------- #
# Similarity guard (word-level change ratio, stdlib difflib)
# --------------------------------------------------------------------------- #
# Stripped from tokens before comparison so the guard measures *word* changes,
# not the punctuation the model is explicitly allowed to add.
_PUNCT_STRIP = "،؛؟!?.:«»\"'`()[]…-—–"


def _compare_tokens(text: str) -> List[str]:
    return [t for t in (tok.strip(_PUNCT_STRIP) for tok in text.split()) if t]


def _word_change_ratio(original: str, corrected: str) -> float:
    """Fraction of words that differ between the two texts, in ``[0, 1]``.

    Uses ``difflib`` (word-level edit distance): matched tokens are the longest
    common subsequence; anything not matched counts as changed. Denominator is
    the longer side, so both heavy additions and deletions push the ratio up.
    """
    a = _compare_tokens(original)
    b = _compare_tokens(corrected)
    if not a and not b:
        return 0.0
    matcher = difflib.SequenceMatcher(a=a, b=b, autojunk=False)
    matched = sum(block.size for block in matcher.get_matching_blocks())
    denom = max(len(a), len(b), 1)
    return (denom - matched) / denom


# --------------------------------------------------------------------------- #
# Public entrypoint
# --------------------------------------------------------------------------- #
def correct_segment(text: str, cfg: LLMPostprocessSection) -> Dict[str, Any]:
    """Correct a single segment's text. Returns a dict with the resulting text
    and the reason it was (or was not) applied. ``text`` is never mutated."""
    original = text or ""
    if not original.strip():
        return {"text_corrected": original, "status": "empty"}

    raw = _ollama_generate(cfg, original)
    if raw is None:
        return {"text_corrected": original, "status": "error"}

    clean = _sanitize_reply(raw)
    if clean is None:
        return {"text_corrected": original, "status": "rejected_format"}

    ratio = _word_change_ratio(original, clean)
    if ratio > cfg.max_word_change_ratio:
        return {
            "text_corrected": original,
            "status": "rejected_similarity",
            "change_ratio": ratio,
        }

    status = "unchanged" if clean == original else "corrected"
    return {"text_corrected": clean, "status": status, "change_ratio": ratio}


def correct_transcript(
    result: Dict[str, Any], cfg: LLMPostprocessSection
) -> Dict[str, Any]:
    """Add ``text_corrected`` to every segment of *result* in place.

    ``text`` (the original ASR/normalized text) is left untouched. Returns a
    stats dict for reporting. When disabled (or an unsupported provider) it is a
    no-op that adds nothing to the segments.
    """
    stats = {
        "enabled": bool(cfg.enabled),
        "total": 0,
        "corrected": 0,
        "unchanged": 0,
        "rejected_similarity": 0,
        "rejected_format": 0,
        "errors": 0,
        "empty": 0,
    }

    if not cfg.enabled:
        return stats
    if (cfg.provider or "ollama").lower() != "ollama":
        logger.warning(
            "llm_postprocess.provider=%r not supported; skipping LLM correction.",
            cfg.provider,
        )
        stats["enabled"] = False
        return stats

    segments = result.get("segments", [])
    for seg in segments:
        stats["total"] += 1
        original = seg.get("text", "") or ""
        outcome = correct_segment(original, cfg)
        # Always set the field (defaults to the original) so it exists uniformly.
        seg["text_corrected"] = outcome["text_corrected"]
        status = outcome["status"]
        stats[status] = stats.get(status, 0) + 1

    logger.info(
        "LLM correction: %d/%d corrected, %d unchanged, %d rejected(similarity), "
        "%d rejected(format), %d errors, %d empty.",
        stats["corrected"],
        stats["total"],
        stats["unchanged"],
        stats["rejected_similarity"],
        stats["rejected_format"],
        stats["errors"],
        stats["empty"],
    )
    return stats
