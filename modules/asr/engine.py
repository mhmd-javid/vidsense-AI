"""Persian ASR engine — batched faster-whisper via the vendored WhisperX.

This is the project's **single** ASR path (the old ``transcriber.py`` is gone).
It wraps ``whisperx.load_model`` + ``FasterWhisperPipeline.transcribe`` and adds
the operational behaviour the target box (Windows, T1000, 4 GB VRAM) needs:

  * ``device: auto`` probes CUDA and **transparently falls back to CPU** — both
    at load time (missing cuDNN/cublas) and at inference time (CUDA libs load
    lazily, so GPU errors can surface mid-transcribe).
  * On CPU it forces ``compute_type=int8`` (WhisperX's ``"default"`` is float32
    on CPU, which is far slower and heavier).
  * Explicit ``load()`` / ``unload()`` so the workflow frees the model before
    the next heavy stage (alignment / diarization) — models are never
    co-resident.
  * Accepts **pre-computed VAD windows** from the VAD stage (via
    ``segmenter.build_precomputed_vad``) so VAD runs exactly once; if none are
    given it builds VAD itself from config (``vad_method``).
  * Recovers ``language_probability`` (WhisperX skips detection when a language
    is forced) for the confidence layer.

Heavy imports (torch, ctranslate2, faster_whisper, whisperx) are deferred to
``load()``/``transcribe()`` so importing this module never requires the stack.
"""
from __future__ import annotations

import gc
import os
from pathlib import Path
from typing import Any, Dict, List, Optional

from core.config import ASRSection, VADSection
from core.utils import get_logger

logger = get_logger(__name__)


def _add_nvidia_dll_dirs() -> None:
    """Best-effort: expose CUDA DLLs shipped by ``nvidia-*`` pip packages.

    Their DLLs land in ``site-packages/nvidia/*/bin`` but are not on PATH;
    registering those dirs lets CTranslate2 find cublas/cudnn. No-op when the
    packages aren't installed or off Windows.
    """
    if os.name != "nt":
        return
    try:
        import importlib.util

        spec = importlib.util.find_spec("nvidia")
        if not spec or not spec.submodule_search_locations:
            return
        nvidia_root = Path(list(spec.submodule_search_locations)[0])
        for bin_dir in nvidia_root.glob("*/bin"):
            if bin_dir.is_dir():
                os.add_dll_directory(str(bin_dir))
    except Exception as exc:  # pragma: no cover
        logger.debug("Could not register NVIDIA DLL dirs: %s", exc)


def _is_gpu_library_error(exc: Exception) -> bool:
    msg = str(exc).lower()
    return any(k in msg for k in ("cublas", "cudnn", "cuda", "cudart", "library", "gpu"))


class ASREngine:
    def __init__(
        self,
        cfg: ASRSection,
        vad_cfg: VADSection,
        download_root: Optional[str] = None,
    ):
        self.cfg = cfg
        self.vad_cfg = vad_cfg
        self.download_root = download_root or cfg.download_root
        self._pipeline = None
        self._vad_windows: Optional[List[Dict[str, Any]]] = None
        self.active_device: Optional[str] = None
        self.active_compute: Optional[str] = None

    # ------------------------------------------------------------- device ---
    def _compute_for(self, device: str) -> str:
        ct = (self.cfg.compute_type or "default").lower()
        # float16 / default are GPU-oriented; int8 is the right CPU choice.
        if device == "cpu" and ct in ("default", "float16", "int8_float16"):
            return "int8"
        return ct

    def _resolve_device(self) -> tuple[str, str]:
        dev = (self.cfg.device or "auto").lower()
        if dev != "auto":
            return dev, self._compute_for(dev)
        try:
            import ctranslate2

            if ctranslate2.get_cuda_device_count() > 0:
                return "cuda", self._compute_for("cuda")
        except Exception as exc:  # pragma: no cover
            logger.debug("CUDA probe failed: %s", exc)
        return "cpu", "int8"

    # ----------------------------------------------------------- asr options -
    def _asr_options(self) -> Dict[str, Any]:
        return {
            "beam_size": int(self.cfg.beam_size),
            "best_of": int(self.cfg.best_of),
            "patience": float(self.cfg.patience),
            "temperatures": list(self.cfg.temperatures),
            "compression_ratio_threshold": float(self.cfg.compression_ratio_threshold),
            "log_prob_threshold": float(self.cfg.log_prob_threshold),
            "no_speech_threshold": float(self.cfg.no_speech_threshold),
            "condition_on_previous_text": bool(self.cfg.condition_on_previous_text),
            "initial_prompt": self.cfg.initial_prompt,
            "suppress_numerals": bool(self.cfg.suppress_numerals),
        }

    # -------------------------------------------------------------- lifecycle
    def load(self, vad_windows: Optional[List[Dict[str, Any]]] = None) -> None:
        if self._pipeline is not None:
            return
        from modules.whisperx.asr import load_model

        self._vad_windows = vad_windows
        device, compute = self._resolve_device()
        if device == "cuda":
            _add_nvidia_dll_dirs()

        def _build(dev: str, comp: str):
            kwargs: Dict[str, Any] = dict(
                whisper_arch=self.cfg.model_size,
                device=dev,
                compute_type=comp,
                language=self.cfg.language,
                asr_options=self._asr_options(),
                download_root=self.download_root,
            )
            if vad_windows is not None:
                from modules.vad.segmenter import build_precomputed_vad

                kwargs["vad_model"] = build_precomputed_vad(vad_windows)
            else:
                kwargs["vad_method"] = self.vad_cfg.method
                kwargs["vad_options"] = {
                    "vad_onset": float(self.vad_cfg.onset),
                    "vad_offset": float(self.vad_cfg.offset),
                    "chunk_size": int(self.vad_cfg.chunk_size),
                }
            return load_model(**kwargs)

        logger.info("Loading ASR '%s' on %s (%s)…", self.cfg.model_size, device, compute)
        try:
            self._pipeline = _build(device, compute)
            self.active_device, self.active_compute = device, compute
        except Exception as exc:
            if device != "cpu":
                logger.warning("GPU ASR load failed (%s). Falling back to CPU int8.", exc)
                self._pipeline = _build("cpu", "int8")
                self.active_device, self.active_compute = "cpu", "int8"
            else:
                raise

    def unload(self) -> None:
        if self._pipeline is not None:
            logger.info("Unloading ASR model to free memory.")
        self._pipeline = None
        self.active_device = None
        gc.collect()
        try:
            import torch

            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except Exception:
            pass

    def __enter__(self) -> "ASREngine":
        return self

    def __exit__(self, *exc) -> None:
        self.unload()

    # -------------------------------------------------------------- transcribe
    def transcribe(
        self, audio, vad_windows: Optional[List[Dict[str, Any]]] = None
    ) -> Dict[str, Any]:
        """Transcribe a mono 16 kHz float32 ``audio`` ndarray.

        Returns ``{segments:[{start,end,text,avg_logprob}], language,
        language_probability}``. ``segments`` and ``language`` come straight from
        WhisperX; ``language_probability`` is recovered here.
        """
        self.load(vad_windows)
        try:
            return self._run(audio)
        except Exception as exc:
            # CUDA runtime libs load lazily at inference, so GPU errors can
            # surface here rather than at load(). Fall back to CPU once.
            if self.active_device != "cpu" and _is_gpu_library_error(exc):
                logger.warning("GPU ASR inference failed (%s). Retrying on CPU int8.", exc)
                windows = self._vad_windows
                self.unload()
                self.cfg.device = "cpu"
                self.load(windows)
                return self._run(audio)
            raise

    def _run(self, audio) -> Dict[str, Any]:
        result = self._pipeline.transcribe(
            audio,
            batch_size=int(self.cfg.batch_size),
            language=self.cfg.language,
            chunk_size=int(self.vad_cfg.chunk_size),
        )
        segments = result.get("segments", [])
        language = result.get("language") or self.cfg.language or "fa"
        _, language_probability = self._detect_language(audio)
        logger.info(
            "ASR done (%s): %d segments, language=%s (p=%s)",
            self.active_device,
            len(segments),
            language,
            f"{language_probability:.2f}" if language_probability is not None else "n/a",
        )
        return {
            "segments": segments,
            "language": language,
            "language_probability": language_probability,
        }

    def _detect_language(self, audio) -> tuple[str, Optional[float]]:
        """Return ``(language, probability)`` from the first 30 s of audio.

        WhisperX only detects when no language is forced; we always want the
        probability for the confidence layer, so we replicate its logic without
        mutating the (forced) tokenizer.
        """
        try:
            from modules.whisperx.audio import N_SAMPLES, log_mel_spectrogram

            model = self._pipeline.model
            n_mels = model.feat_kwargs.get("feature_size")
            segment = log_mel_spectrogram(
                audio[:N_SAMPLES],
                n_mels=n_mels if n_mels is not None else 80,
                padding=0 if audio.shape[0] >= N_SAMPLES else N_SAMPLES - audio.shape[0],
            )
            encoder_output = model.encode(segment)
            results = model.model.detect_language(encoder_output)
            language_token, probability = results[0][0]
            return language_token[2:-2], float(probability)
        except Exception as exc:  # never fail transcription over a metric
            logger.debug("Language-probability detection failed: %s", exc)
            return (self.cfg.language or "fa"), None
