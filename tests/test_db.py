"""Unit tests for the SQLite storage layer (uses a temp DB, no models)."""
import sys
import tempfile
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from modules.storage.db import Database, STATUS_READY, STATUS_ERROR  # noqa: E402


def _fresh_db() -> Database:
    tmp = Path(tempfile.mkdtemp()) / "test.db"
    return Database(tmp)


def test_upsert_and_get_video():
    db = _fresh_db()
    db.upsert_video("v1", title="Hello", source="u", source_type="url")
    v = db.get_video("v1")
    assert v["title"] == "Hello" and v["status"] == "pending"
    # Update path preserves existing fields.
    db.upsert_video("v1", duration=42.0)
    v = db.get_video("v1")
    assert v["title"] == "Hello" and v["duration"] == 42.0


def test_status_transitions():
    db = _fresh_db()
    db.upsert_video("v1", title="x")
    db.set_status("v1", STATUS_READY)
    assert db.get_video("v1")["status"] == STATUS_READY
    db.set_status("v1", STATUS_ERROR, "boom")
    row = db.get_video("v1")
    assert row["status"] == STATUS_ERROR and row["error"] == "boom"


def test_chunks_roundtrip_with_embeddings():
    db = _fresh_db()
    db.upsert_video("v1", title="x")
    chunks = [
        {"index": 0, "start": 0.0, "end": 10.0, "text": "چانک اول"},
        {"index": 1, "start": 10.0, "end": 20.0, "text": "chunk two"},
    ]
    emb = np.random.rand(2, 8).astype(np.float32)
    db.replace_chunks("v1", chunks, embeddings=emb, embedding_model="e5")
    rows = db.get_chunks("v1")
    assert len(rows) == 2
    assert rows[0].text == "چانک اول"
    assert rows[0].embedding is not None and rows[0].embedding.shape == (8,)
    # Float32 blob survives the roundtrip.
    assert np.allclose(rows[0].embedding, emb[0], atol=1e-6)
    assert db.count_chunks("v1") == 2


def test_replace_chunks_is_idempotent():
    db = _fresh_db()
    db.upsert_video("v1", title="x")
    one = [{"start": 0.0, "end": 5.0, "text": "a"}]
    db.replace_chunks("v1", one)
    db.replace_chunks("v1", one)  # replace, not append
    assert db.count_chunks("v1") == 1


def test_delete_video_cascades_chunks():
    db = _fresh_db()
    db.upsert_video("v1", title="x")
    db.replace_chunks("v1", [{"start": 0, "end": 1, "text": "a"}])
    db.delete_video("v1")
    assert db.get_video("v1") is None
    assert db.count_chunks("v1") == 0


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
