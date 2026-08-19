"""Smoke tests: everything imports, config loads, environment tools are present.

These are cheap (no ML models loaded — the heavy ASR/align/diarize deps are
lazy-imported inside methods) and catch wiring/import regressions fast.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


def test_core_imports():
    import core.config, core.utils, core.services  # noqa: F401
    import modules.ingestion.downloader  # noqa: F401
    import modules.audio.extractor, modules.audio.preprocess, modules.audio.loader  # noqa: F401
    import modules.vad.detector, modules.vad.segmenter  # noqa: F401
    import modules.asr.engine  # noqa: F401
    import modules.alignment.aligner  # noqa: F401
    import modules.diarization.diarizer  # noqa: F401
    import modules.postprocess.persian  # noqa: F401
    import modules.confidence.scorer  # noqa: F401
    import modules.storage.db, modules.storage.transcript  # noqa: F401
    import modules.workflow.pipeline  # noqa: F401


def test_services_container_shape():
    """Services must expose exactly cfg/db/processing — no RAG leftovers."""
    from core.services import Services
    fields = set(Services.__dataclass_fields__)
    assert fields == {"cfg", "db", "processing"}


def test_config_loads_with_expected_defaults():
    from core.config import load_config
    cfg = load_config()
    # Quality-first Persian ASR defaults.
    assert cfg.asr.model_size == "medium"
    assert cfg.asr.language == "fa"
    assert cfg.asr.compute_type == "int8"
    assert cfg.asr.device == "auto"
    # Alignment on, diarization off (graceful) by default.
    assert cfg.alignment.enabled is True
    assert cfg.diarization.enabled is False
    # Faithful post-processing on; LLM never involved.
    assert cfg.postprocess.enabled is True
    assert cfg.videos_dir_abs.exists()  # ensure_dirs ran


def test_no_rag_config_sections():
    """The chat/RAG stack is gone — its config sections must not resurface."""
    from core.config import Config
    fields = set(Config.__dataclass_fields__)
    for gone in ("embedding", "vectorstore", "llm", "rag"):
        assert gone not in fields


def test_pipeline_stage_order():
    from modules.workflow.pipeline import STAGES
    keys = [k for k, _ in STAGES]
    assert keys == [
        "download", "extract", "preprocess", "vad",
        "transcribe", "align", "diarize", "postprocess", "ready",
    ]


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
