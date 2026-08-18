"""Unit tests for the NumPy vector store (no models required)."""
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from modules.vectorstore.numpy_store import NumpyVectorStore  # noqa: E402
from modules.vectorstore.base import VectorItem  # noqa: E402


def _store():
    s = NumpyVectorStore(normalize=True)
    s.upsert([
        VectorItem("a", np.array([1.0, 0.0, 0.0]), {"video_id": "v1", "text": "east"}),
        VectorItem("b", np.array([0.0, 1.0, 0.0]), {"video_id": "v1", "text": "north"}),
        VectorItem("c", np.array([0.0, 0.0, 1.0]), {"video_id": "v2", "text": "up"}),
    ])
    return s


def test_count():
    assert _store().count() == 3


def test_search_ranks_by_cosine():
    s = _store()
    hits = s.search(np.array([0.9, 0.1, 0.0]), top_k=3)
    assert hits[0].id == "a"  # closest to the x-axis vector
    assert hits[0].score >= hits[1].score >= hits[2].score


def test_metadata_filter_scopes_results():
    s = _store()
    hits = s.search(np.array([0.0, 0.0, 1.0]), top_k=5, where={"video_id": "v1"})
    assert all(h.metadata["video_id"] == "v1" for h in hits)
    assert "c" not in [h.id for h in hits]


def test_upsert_replaces_existing_id():
    s = _store()
    s.upsert([VectorItem("a", np.array([0.0, 0.0, 1.0]), {"video_id": "v1", "text": "changed"})])
    assert s.count() == 3
    hit = s.search(np.array([0.0, 0.0, 1.0]), top_k=1, where={"video_id": "v1"})[0]
    assert hit.id == "a" and hit.metadata["text"] == "changed"


def test_delete():
    s = _store()
    s.delete(["a"])
    assert s.count() == 2
    assert "a" not in [h.id for h in s.search(np.array([1.0, 0.0, 0.0]), top_k=5)]


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
