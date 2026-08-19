"""Unit tests for video ingestion routing (no network, no downloads).

Covers the Aparat hash regex, the direct-API guard's fast return on non-Aparat
URLs, deterministic video ids, and local upload / file registration.
"""
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pytest  # noqa: E402

from core.utils import make_video_id  # noqa: E402
from modules.ingestion.downloader import (  # noqa: E402
    DownloadError,
    VideoDownloader,
    _APARAT_HASH_RE,
    resolve_aparat_direct_link,
)


def _downloader() -> VideoDownloader:
    return VideoDownloader(Path(tempfile.mkdtemp()) / "videos")


def test_aparat_hash_regex():
    m = _APARAT_HASH_RE.search("https://www.aparat.com/v/abC123?foo=bar")
    assert m and m.group(1) == "abC123"
    assert _APARAT_HASH_RE.search("https://www.youtube.com/watch?v=x") is None


def test_resolve_aparat_returns_none_for_non_aparat():
    # No hash match -> returns before any network call.
    assert resolve_aparat_direct_link("https://youtube.com/watch?v=x") is None
    assert resolve_aparat_direct_link("") is None


def test_make_video_id_is_deterministic():
    a = make_video_id("https://example.com/x")
    b = make_video_id("https://example.com/x")
    c = make_video_id("https://example.com/y")
    assert a == b and a != c
    assert len(a) == 12


def test_empty_url_raises():
    with pytest.raises(DownloadError):
        _downloader().download_url("   ")


def test_save_upload_routes_as_upload():
    dl = _downloader()
    res = dl.save_upload(b"fake-bytes", "clip.mp4")
    assert res.source_type == "upload"
    assert res.title == "clip.mp4"
    assert Path(res.filepath).exists()
    assert Path(res.filepath).suffix == ".mp4"


def test_save_upload_empty_raises():
    with pytest.raises(DownloadError):
        _downloader().save_upload(b"", "empty.mp4")


def test_ingest_local_copies_and_routes():
    dl = _downloader()
    src = Path(tempfile.mkdtemp()) / "local.wav"
    src.write_bytes(b"RIFFxxxxWAVE")
    res = dl.ingest_local(src)
    assert res.source_type == "upload"
    assert res.title == "local.wav"
    assert Path(res.filepath).exists()
    assert Path(res.filepath).parent == dl.videos_dir


def test_ingest_local_missing_raises():
    with pytest.raises(DownloadError):
        _downloader().ingest_local("does/not/exist.mp4")


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
