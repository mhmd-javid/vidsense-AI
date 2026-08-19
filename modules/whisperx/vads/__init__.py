"""VAD backends for the vendored WhisperX engine.

Re-exports the three VAD classes so ``from modules.whisperx.vads import Vad,
Silero, Pyannote`` (used by ``asr.py``) resolves. ``Vad`` must be imported first
because ``Silero``/``Pyannote`` subclass it.
"""
from modules.whisperx.vads.vad import Vad
from modules.whisperx.vads.silero import Silero
from modules.whisperx.vads.pyannote import Pyannote

__all__ = ["Vad", "Silero", "Pyannote"]
