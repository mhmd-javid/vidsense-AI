"""Unit tests for the semantic chunker (no models required)."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from modules.asr.chunker import chunk_segments  # noqa: E402


def _segs(*triples):
    return [{"start": s, "end": e, "text": t} for s, e, t in triples]


def test_empty_returns_empty():
    assert chunk_segments([]) == []


def test_merges_short_segments_into_target_window():
    # 10 x 5s segments = 50s -> should merge into ~30s chunks, not stay as 10.
    segs = _segs(*[(i * 5, i * 5 + 5, f"جمله شماره {i}.") for i in range(10)])
    chunks = chunk_segments(segs, target_seconds=30, max_seconds=45, max_chars=700)
    assert 1 <= len(chunks) < 10
    # Timestamps preserved & monotonic.
    assert chunks[0].start == 0
    assert chunks[-1].end == 50
    for c in chunks:
        assert c.end > c.start
        assert c.text


def test_hard_cap_on_chars():
    long_text = "x" * 400
    segs = _segs((0, 3, long_text), (3, 6, long_text))  # 800 chars > 700 cap
    chunks = chunk_segments(segs, target_seconds=60, max_seconds=120, max_chars=700)
    assert len(chunks) >= 2


def test_hard_cap_on_duration():
    segs = _segs((0, 40, "بدون نقطه پایان"), (40, 80, "همچنان ادامه دارد"))
    chunks = chunk_segments(segs, target_seconds=30, max_seconds=45, max_chars=9999)
    assert len(chunks) >= 2


def test_sentence_boundary_preference():
    # Reaches target at a sentence end -> should flush cleanly.
    segs = _segs((0, 20, "بخش اول"), (20, 32, "پایان جمله."), (32, 50, "بخش بعدی"))
    chunks = chunk_segments(segs, target_seconds=30, max_seconds=60, max_chars=9999)
    assert chunks[0].text.endswith(".")


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
