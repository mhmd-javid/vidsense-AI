"""Smoke tests: everything imports, config loads, environment tools are present.

These are cheap (no ML models loaded) and catch wiring/import regressions fast.
"""
import shutil
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


def test_core_imports():
    import core.config, core.utils, core.services  # noqa: F401
    import modules.ingestion.downloader  # noqa: F401
    import modules.audio.extractor  # noqa: F401
    import modules.asr.transcriber, modules.asr.chunker  # noqa: F401
    import modules.embedding.embedder  # noqa: F401
    import modules.vectorstore.numpy_store, modules.vectorstore.factory  # noqa: F401
    import modules.storage.db  # noqa: F401
    import modules.llm.ollama_client  # noqa: F401
    import modules.rag.pipeline  # noqa: F401
    import modules.workflow.pipeline  # noqa: F401


def test_config_loads_with_expected_defaults():
    from core.config import load_config
    cfg = load_config()
    assert cfg.asr.model_size in {"tiny", "base", "small", "medium"}
    assert "e5" in cfg.embedding.model_name.lower()  # multilingual, not English-only
    assert cfg.vectorstore.backend == "numpy"
    assert cfg.videos_dir_abs.exists()  # ensure_dirs ran


def test_timestamp_formatting():
    from core.utils import format_timestamp, format_range
    assert format_timestamp(0) == "00:00"
    assert format_timestamp(75) == "01:15"
    assert format_timestamp(3661) == "1:01:01"
    assert format_range(130, 155) == "02:10–02:35"


def test_ffmpeg_binary_available():
    import imageio_ffmpeg
    exe = imageio_ffmpeg.get_ffmpeg_exe()
    assert Path(exe).exists()


def test_vector_store_factory():
    from modules.vectorstore.factory import create_vector_store
    from modules.vectorstore.base import VectorStore
    assert isinstance(create_vector_store("numpy"), VectorStore)


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
