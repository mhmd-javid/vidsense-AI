"""Typed configuration loader.

Reads ``config/config.yaml`` into dataclasses so the rest of the codebase gets
attribute access + editor autocomplete instead of raw dict lookups. Paths are
resolved to absolute locations under the project root, and the data
directories are created on load.

The sections mirror the pipeline stages: download -> audio -> preprocess -> vad
-> asr -> alignment -> diarization -> postprocess -> confidence.
"""
from __future__ import annotations

from dataclasses import dataclass, field, fields
from pathlib import Path
from typing import Any, List, Optional

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
class DownloadSection:
    max_height: int = 720
    retries: int = 3
    socket_timeout: int = 30
    aparat_direct_api: bool = True
    insecure_ssl: bool = True


@dataclass
class AudioSection:
    sample_rate: int = 16000


@dataclass
class PreprocessSection:
    enabled: bool = True
    loudnorm: bool = True
    highpass_hz: int = 0
    denoise: bool = False


@dataclass
class VADSection:
    method: str = "pyannote"
    onset: float = 0.500
    offset: float = 0.363
    chunk_size: int = 30
    model_fp: Optional[str] = None


@dataclass
class ASRSection:
    model_size: str = "medium"
    device: str = "auto"
    compute_type: str = "int8"
    language: Optional[str] = "fa"
    batch_size: int = 8
    beam_size: int = 5
    best_of: int = 5
    patience: float = 1.0
    temperatures: List[float] = field(
        default_factory=lambda: [0.0, 0.2, 0.4, 0.6, 0.8, 1.0]
    )
    compression_ratio_threshold: float = 2.4
    log_prob_threshold: float = -1.0
    no_speech_threshold: float = 0.6
    condition_on_previous_text: bool = False
    initial_prompt: Optional[str] = None
    suppress_numerals: bool = False
    download_root: str = "models"


@dataclass
class AlignmentSection:
    enabled: bool = True
    model_name: Optional[str] = None
    interpolate_method: str = "nearest"
    return_char_alignments: bool = False


@dataclass
class DiarizationSection:
    enabled: bool = False
    model_name: str = "pyannote/speaker-diarization-community-1"
    hf_token: Optional[str] = None
    num_speakers: Optional[int] = None
    min_speakers: Optional[int] = None
    max_speakers: Optional[int] = None


@dataclass
class PostprocessSection:
    enabled: bool = True
    arabic_to_persian: bool = True
    fix_zwnj: bool = True
    normalize_punctuation: bool = True
    collapse_repeats: bool = True
    persian_digits: bool = False


@dataclass
class ConfidenceSection:
    low_threshold: float = 0.5
    weight_logprob: float = 0.6
    weight_word_score: float = 0.4


@dataclass
class Config:
    app: AppSection = field(default_factory=AppSection)
    paths: PathsSection = field(default_factory=PathsSection)
    download: DownloadSection = field(default_factory=DownloadSection)
    audio: AudioSection = field(default_factory=AudioSection)
    preprocess: PreprocessSection = field(default_factory=PreprocessSection)
    vad: VADSection = field(default_factory=VADSection)
    asr: ASRSection = field(default_factory=ASRSection)
    alignment: AlignmentSection = field(default_factory=AlignmentSection)
    diarization: DiarizationSection = field(default_factory=DiarizationSection)
    postprocess: PostprocessSection = field(default_factory=PostprocessSection)
    confidence: ConfidenceSection = field(default_factory=ConfidenceSection)

    # -- absolute, resolved paths ------------------------------------------
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
        download=_build_section(DownloadSection, raw.get("download", {})),
        audio=_build_section(AudioSection, raw.get("audio", {})),
        preprocess=_build_section(PreprocessSection, raw.get("preprocess", {})),
        vad=_build_section(VADSection, raw.get("vad", {})),
        asr=_build_section(ASRSection, raw.get("asr", {})),
        alignment=_build_section(AlignmentSection, raw.get("alignment", {})),
        diarization=_build_section(DiarizationSection, raw.get("diarization", {})),
        postprocess=_build_section(PostprocessSection, raw.get("postprocess", {})),
        confidence=_build_section(ConfidenceSection, raw.get("confidence", {})),
    )
    cfg.ensure_dirs()
    return cfg


# Convenience singleton for simple imports: ``from core.config import CONFIG``
CONFIG = load_config()
