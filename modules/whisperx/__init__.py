"""Vendored WhisperX inference engine (adapted from m-bain/whisperX).

This package is an *internal engine*, not a public dependency: VidSense's thin
staged wrappers (``modules/asr/engine.py``, ``modules/alignment/aligner.py``,
``modules/vad/detector.py``, ``modules/diarization/diarizer.py``) import the
specific functions they need from the submodules here and expose a clean,
resource-disciplined API to the rest of the app.

Import policy
-------------
This ``__init__`` is intentionally *empty of heavy imports*. Importing the
submodules (``asr``, ``alignment``, ``diarize``, ``vads``) pulls in torch,
ctranslate2, faster-whisper and pyannote.audio, so the wrappers import those
submodules lazily (inside their ``load()`` methods) — never at module top level.
That keeps ``import modules.whisperx`` cheap and lets the rest of the codebase be
imported (and unit-tested) without the heavy speech stack installed.

Adaptations from upstream (mechanical, no logic changes):
  * absolute ``whisperx.*`` imports rewritten to ``modules.whisperx.*``
  * added ``log_utils.py`` shim forwarding to ``core.utils.get_logger``
  * added this + ``vads/__init__.py`` to make the tree an importable package
"""

__all__: list[str] = []
