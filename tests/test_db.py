"""Unit tests for the SQLite storage layer (temp DB, no models, no embeddings).

The chat/RAG schema (chunks + embedding blobs) is gone; the DB now holds only
video metadata, processing status, quality metrics, and pointers to the JSON /
SRT / VTT transcript files.
"""
import sys
import tempfile
from pathlib import Path

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
    # Update path preserves existing fields and only overwrites what's passed.
    db.upsert_video("v1", duration=42.0)
    v = db.get_video("v1")
    assert v["title"] == "Hello" and v["duration"] == 42.0


def test_transcript_metadata_roundtrip():
    db = _fresh_db()
    db.upsert_video("v1", title="x")
    db.upsert_video(
        "v1",
        language="fa",
        language_prob=0.98,
        num_speakers=2,
        quality_score=0.71,
        transcript_path="/data/transcripts/v1.json",
        srt_path="/data/transcripts/v1.srt",
        vtt_path="/data/transcripts/v1.vtt",
    )
    v = db.get_video("v1")
    assert v["language"] == "fa"
    assert v["num_speakers"] == 2
    assert abs(v["quality_score"] - 0.71) < 1e-9
    assert v["transcript_path"].endswith("v1.json")
    assert v["srt_path"].endswith("v1.srt")
    assert v["vtt_path"].endswith("v1.vtt")


def test_status_transitions():
    db = _fresh_db()
    db.upsert_video("v1", title="x")
    db.set_status("v1", STATUS_READY)
    assert db.get_video("v1")["status"] == STATUS_READY
    db.set_status("v1", STATUS_ERROR, "boom")
    row = db.get_video("v1")
    assert row["status"] == STATUS_ERROR and row["error"] == "boom"


def test_list_videos_ready_only():
    db = _fresh_db()
    db.upsert_video("v1", title="a")
    db.upsert_video("v2", title="b")
    db.set_status("v2", STATUS_READY)
    assert len(db.list_videos()) == 2
    ready = db.list_videos(ready_only=True)
    assert [r["video_id"] for r in ready] == ["v2"]


def test_delete_video():
    db = _fresh_db()
    db.upsert_video("v1", title="x")
    db.delete_video("v1")
    assert db.get_video("v1") is None


def test_no_embedding_api():
    """The embedding/chunk API must be fully removed."""
    db = _fresh_db()
    for gone in ("replace_chunks", "get_chunks", "count_chunks"):
        assert not hasattr(db, gone)


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
