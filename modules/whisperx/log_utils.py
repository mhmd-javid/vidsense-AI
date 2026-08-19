"""Logging shim for the vendored WhisperX engine.

Upstream WhisperX imports ``from whisperx.log_utils import get_logger`` in every
module. Rather than vendor a second logging setup, this shim forwards to the
project's own logger factory so WhisperX logs flow through the same handlers and
formatting as the rest of VidSense.
"""
from __future__ import annotations

from core.utils import get_logger

__all__ = ["get_logger"]
