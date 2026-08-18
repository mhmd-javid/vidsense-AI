"""Typed configuration loader.

Reads ``config/config.yaml`` into dataclasses so the rest of the codebase gets
attribute access + editor autocomplete instead of raw dict lookups. Paths are
resolved to absolute locations under the project root, and the data
directories are created on load.
"""
from __future__ import annotations

from dataclasses import dataclass, field, fields
from pathlib import Path
from typing import Any, Optional

import yaml

from core.utils import ensure_dir, resolve_path, PROJECT_ROOT

DEFAULT_CONFIG_PATH = PROJECT_ROOT / "config" / "config.yaml"


# --------------------------------------------------------------------------- #
# Section dataclasses
# --------------------------------------------------------------------------- #
@dataclass
class AppSection:
    db_path: str = "data/videoai.db"
    log_level: str = "INFO"


@dataclass
class PathsSection:
    videos_dir: str = "data/videos"
    audio_dir: str = "data/audio"
    transcripts_dir: str = "data/transcripts"
    models_dir: str = "models"


@dataclass
class ASRSection:
    model_size: str = "small"
    device: str = "auto"
    compute_type: str = "int8"
    language: Optional[str] = None
    beam_size: int = 5
    vad_filter: bool = True
    download_root: str = "models"


@dataclass
class ChunkingSection:
    target_seconds: float = 30.0
    max_seconds: float = 45.0
    max_chars: int = 700
    min_chars: int = 40


@dataclass
class EmbeddingSection:
    model_name: str = "intfloat/multilingual-e5-small"
    device: str = "cpu"
    batch_size: int = 16
    normalize: bool = True
    query_prefix: str = "query: "
    passage_prefix: str = "passage: "


@dataclass
class VectorStoreSection:
    backend: str = "numpy"
    top_k: int = 5


@dataclass
class LLMSection:
    backend: str = "ollama"
    model: str = "qwen2.5:3b-instruct"
    host: str = "http://localhost:11434"
    temperature: float = 0.2
    num_ctx: int = 4096
    max_tokens: int = 1024
    keep_alive: str = "5m"
    request_timeout: int = 180


@dataclass
class RAGSection:
    top_k: int = 5
    min_score: float = 0.10
    language: str = "auto"


@dataclass
class Config:
    app: AppSection = field(default_factory=AppSection)
    paths: PathsSection = field(default_factory=PathsSection)
    asr: ASRSection = field(default_factory=ASRSection)
    chunking: ChunkingSection = field(default_factory=ChunkingSection)
    embedding: EmbeddingSection = field(default_factory=EmbeddingSection)
    vectorstore: VectorStoreSection = field(default_factory=VectorStoreSection)
    llm: LLMSection = field(default_factory=LLMSection)
    rag: RAGSection = field(default_factory=RAGSection)

    # -- absolute, resolved paths (filled in __post_init__) -----------------
    @property
    def db_path_abs(self) -> Path:
        return resolve_path(self.app.db_path)

    @property
    def videos_dir_abs(self) -> Path:
        return resolve_path(self.paths.videos_dir)

    @property
    def audio_dir_abs(self) -> Path:
        return resolve_path(self.paths.audio_dir)

    @property
    def transcripts_dir_abs(self) -> Path:
        return resolve_path(self.paths.transcripts_dir)

    @property
    def models_dir_abs(self) -> Path:
        return resolve_path(self.paths.models_dir)

    def ensure_dirs(self) -> None:
        for p in (
            self.db_path_abs.parent,
            self.videos_dir_abs,
            self.audio_dir_abs,
            self.transcripts_dir_abs,
            self.models_dir_abs,
        ):
            ensure_dir(p)


def _build_section(cls, data: dict[str, Any]):
    """Instantiate a dataclass from a dict, ignoring unknown keys."""
    if not data:
        return cls()
    known = {f.name for f in fields(cls)}
    filtered = {k: v for k, v in data.items() if k in known}
    return cls(**filtered)


def load_config(path: str | Path | None = None) -> Config:
    """Load configuration from YAML into a :class:`Config` object.

    Falls back to dataclass defaults for any missing section/key so the app
    still runs if the file is partial. Data directories are created on load.
    """
    cfg_path = Path(path) if path else DEFAULT_CONFIG_PATH
    raw: dict[str, Any] = {}
    if cfg_path.exists():
        with open(cfg_path, "r", encoding="utf-8") as fh:
            raw = yaml.safe_load(fh) or {}

    cfg = Config(
        app=_build_section(AppSection, raw.get("app", {})),
        paths=_build_section(PathsSection, raw.get("paths", {})),
        asr=_build_section(ASRSection, raw.get("asr", {})),
        chunking=_build_section(ChunkingSection, raw.get("chunking", {})),
        embedding=_build_section(EmbeddingSection, raw.get("embedding", {})),
        vectorstore=_build_section(VectorStoreSection, raw.get("vectorstore", {})),
        llm=_build_section(LLMSection, raw.get("llm", {})),
        rag=_build_section(RAGSection, raw.get("rag", {})),
    )
    cfg.ensure_dirs()
    return cfg


# Convenience singleton for simple imports: ``from core.config import CONFIG``
CONFIG = load_config()
