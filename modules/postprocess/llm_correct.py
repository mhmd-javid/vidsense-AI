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
import os
import urllib.request
from typing import Any, Dict, List, Optional

from core.config import LLMPostprocessSection
from core.utils import get_logger

logger = get_logger(__name__)


# --------------------------------------------------------------------------- #
# Prompt
# --------------------------------------------------------------------------- #
_SYSTEM_PROMPT = (
    "Fix Persian ASR errors. Change ONLY clear speech-recognition mistakes. "
    "If unsure, keep original. Never rewrite, rephrase, or improve style.\n\n"
    "Error patterns (wrong → correct):\n"
    "مدن→معدن | صنایعه→صنایع | شبکمون→شبکه | محققق→محقق | تنگهی→تنگه‌ی\n"
    "تبلیک→تبریک | هستهی→هسته‌ای | همهی→همه‌ی | میآید→می‌آید | بیشت→بیشتر\n"
    "منصجم→منسجم | مصوبات→مصوبات (fix تشدید) | قرضدانی→قدردانی\n\n"
    "NEVER change:\n"
    "- Valid words: طراز, عصاره, ملت, طور, عیار\n"
    "- Names: خامنه‌ای, پزشکیان, ترامپ\n"
    "- Numbers, dates, quantities, technical terms\n\n"
    "Output ONLY corrected text. No explanations, markdown, or quotes."
)


# --------------------------------------------------------------------------- #
# Ollama transport (stdlib only)
# --------------------------------------------------------------------------- #
def _ollama_generate(cfg: LLMPostprocessSection, prompt: str) -> Optional[str]:
    """One deterministic ``/api/generate`` call."""
    url = cfg.endpoint.rstrip("/") + "/api/generate"
    body = {
        "model": cfg.model,
        "system": _SYSTEM_PROMPT,
        "prompt": prompt,
        "stream": False,
        # Keep the model resident between per-segment calls so it is not
        # reloaded for every segment of a transcript.
        "keep_alive": cfg.keep_alive,
        "options": {
            "temperature": float(cfg.temperature),
            # Short ASR segments: a small context and a capped output are enough
            # and avoid a stray reply running away or a needlessly large window.
            "num_ctx": int(cfg.num_ctx),
            "num_predict": int(cfg.num_predict),
        },
    }
    data = json.dumps(body).encode("utf-8")
    req = urllib.request.Request(
        url, data=data, headers={"Content-Type": "application/json"}, method="POST"
    )
    try:
        with urllib.request.urlopen(req, timeout=cfg.timeout_seconds) as resp:
            payload = json.loads(resp.read().decode("utf-8"))
        return payload.get("response", "")
    except Exception as exc:
        logger.warning(
            "[llm] Ollama call failed (%s: %s) — keeping original text.",
            type(exc).__name__, exc,
        )
        return None


# --------------------------------------------------------------------------- #
# OpenRouter transport (stdlib only)
# --------------------------------------------------------------------------- #
_OPENROUTER_URL = "https://openrouter.ai/api/v1/chat/completions"


def _read_openrouter_key(cfg: LLMPostprocessSection) -> Optional[str]:
    """Read the OpenRouter API key from a Windows environment variable.

    The key value is read ONLY from the environment variable named by
    ``cfg.openrouter_key_env`` (default ``OPENROUTER_API_KEY``) — never from the
    config file or the repository, so it is never committed. Set it once, e.g.
    in PowerShell::

        setx OPENROUTER_API_KEY "sk-or-..."
    """
    env_name = (getattr(cfg, "openrouter_key_env", "") or "OPENROUTER_API_KEY").strip()
    key = (os.environ.get(env_name) or "").strip()
    return key or None


def _openrouter_generate(cfg: LLMPostprocessSection, prompt: str) -> Optional[str]:
    """One chat-completion call to OpenRouter."""
    key = _read_openrouter_key(cfg)
    if key is None:
        logger.error(
            "[llm] provider=openrouter but environment variable %r is not set — "
            "keeping original text.",
            (getattr(cfg, "openrouter_key_env", "") or "OPENROUTER_API_KEY"),
        )
        return None
    body = {
        "model": cfg.openrouter_model,
        "messages": [
            {"role": "system", "content": _SYSTEM_PROMPT},
            {"role": "user", "content": prompt},
        ],
        "temperature": float(cfg.temperature),
        "max_tokens": 256,
    }
    data = json.dumps(body).encode("utf-8")
    req = urllib.request.Request(
        _OPENROUTER_URL,
        data=data,
        headers={
            "Authorization": f"Bearer {key}",
            "Content-Type": "application/json",
            "HTTP-Referer": "http://localhost:8501",
            "X-Title": "VidSense",
        },
        method="POST",
    )
    try:
        with urllib.request.urlopen(req, timeout=cfg.timeout_seconds) as resp:
            payload = json.loads(resp.read().decode("utf-8"))
        return payload["choices"][0]["message"]["content"]
    except Exception as exc:
        logger.warning(
            "[llm] OpenRouter call failed (%s: %s) — keeping original text.",
            type(exc).__name__, exc,
        )
        return None


# --------------------------------------------------------------------------- #
# Reply validation
# --------------------------------------------------------------------------- #
_QUOTE_PAIRS = [
    ('"', '"'),
    ("'", "'"),
    ("«", "»"),
    ("“", "”"),
    ("‘", "’"),
    ("`", "`"),
]


def _strip_wrapping(text: str) -> str:
    """Strip surrounding quote/backtick wrappers."""
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


def _persian_score(text: str) -> int:
    """Count Persian/Arabic characters."""
    return sum(1 for c in text if "\u0600" <= c <= "\u06FF")


def _sanitize_reply(raw: Optional[str]) -> Optional[str]:
    """Return a clean corrected sentence, or ``None`` if invalid.

    Strips markdown, unwraps code blocks, and picks the line with the most
    Persian characters (so a preamble does not invalidate the reply).
    """
    if not raw:
        return None
    text = raw.strip()

    # Strip inline markdown emphasis before any other processing.
    text = text.replace("**", "").replace("__", "").replace("*", "").replace("_", "")

    # Unwrap a fenced code block if the whole reply is one.
    if text.startswith("```") and text.endswith("```") and len(text) >= 6:
        inner = text[3:-3]
        inner = inner.split("\n", 1)[-1] if "\n" in inner else inner
        text = inner.strip()

    # Collect non-empty lines.
    lines = [ln.strip() for ln in text.splitlines() if ln.strip()]
    if not lines:
        return None

    # Prefer the line with the most Persian/Arabic characters.
    best_line = max(lines, key=lambda ln: (_persian_score(ln), len(ln)))
    best_line = _strip_wrapping(best_line)

    if not best_line or "```" in best_line:
        return None
    return best_line


# --------------------------------------------------------------------------- #
# Similarity guard (word-level change ratio, stdlib difflib)
# --------------------------------------------------------------------------- #
_PUNCT_STRIP = '\u060c\u061b\u061f!?.:\u00ab\u00bb"\'`()[]\u2026-\u2014\u2013'


def _compare_tokens(text: str) -> List[str]:
    return [t for t in (tok.strip(_PUNCT_STRIP) for tok in text.split()) if t]


def _word_change_ratio(original: str, corrected: str) -> float:
    """Fraction of words that differ, in [0, 1].

    For very short segments (≤2 words) a 0.5x discount is applied so that
    single-word fixes like "مدن→معدن" are not rejected.
    """
    a = _compare_tokens(original)
    b = _compare_tokens(corrected)
    if not a and not b:
        return 0.0
    matcher = difflib.SequenceMatcher(a=a, b=b, autojunk=False)
    matched = sum(block.size for block in matcher.get_matching_blocks())
    denom = max(len(a), len(b), 1)
    ratio = (denom - matched) / denom
    if denom <= 2:
        ratio = ratio * 0.5
    return ratio


# --------------------------------------------------------------------------- #
# Public entrypoint
# --------------------------------------------------------------------------- #
def correct_segment(text: str, cfg: LLMPostprocessSection) -> Dict[str, Any]:
    """Correct a single segment's text. Returns a dict with the resulting text
    and the reason it was (or was not) applied. ``text`` is never mutated."""
    original = text or ""
    if not original.strip():
        return {"text_corrected": original, "status": "empty"}

    provider = (cfg.provider or "ollama").lower()
    logger.info("[llm] input segment (%s): %s", provider, original)

    if provider == "ollama":
        raw = _ollama_generate(cfg, original)
    elif provider == "openrouter":
        raw = _openrouter_generate(cfg, original)
    else:
        logger.warning("[llm] unknown provider=%r — skipping.", provider)
        return {"text_corrected": original, "status": "error"}

    if raw is None:
        logger.warning("[llm] %s returned None for: %s", provider, original)
        return {"text_corrected": original, "status": "error"}

    logger.info("[llm] raw output (%s): %s", provider, raw)
    clean = _sanitize_reply(raw)
    logger.info("[llm] sanitized: %s", clean)

    if clean is None:
        logger.info("[llm] status: rejected_format")
        return {"text_corrected": original, "status": "rejected_format"}

    ratio = _word_change_ratio(original, clean)
    logger.info(
        "[llm] change_ratio: %.2f (threshold: %.2f)",
        ratio, cfg.max_word_change_ratio,
    )

    if ratio > cfg.max_word_change_ratio:
        logger.info(
            "[llm] status: rejected_similarity (ratio %.2f > %.2f)",
            ratio, cfg.max_word_change_ratio,
        )
        return {
            "text_corrected": original,
            "status": "rejected_similarity",
            "change_ratio": ratio,
        }

    status = "unchanged" if clean == original else "corrected"
    logger.info("[llm] status: %s", status)
    return {"text_corrected": clean, "status": status, "change_ratio": ratio}


def correct_transcript(
    result: Dict[str, Any], cfg: LLMPostprocessSection
) -> Dict[str, Any]:
    """Add ``text_corrected`` to every segment of *result* in place."""
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

    provider = (cfg.provider or "ollama").lower()
    if provider not in ("ollama", "openrouter"):
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
        seg["text_corrected"] = outcome["text_corrected"]
        status = outcome["status"]
        stats[status] = stats.get(status, 0) + 1

    logger.info(
        "[llm] stats: total=%d, corrected=%d, unchanged=%d, rejected_similarity=%d, "
        "rejected_format=%d, errors=%d, empty=%d",
        stats["total"],
        stats["corrected"],
        stats["unchanged"],
        stats.get("rejected_similarity", 0),
        stats.get("rejected_format", 0),
        stats.get("errors", 0),
        stats.get("empty", 0),
    )
    return stats