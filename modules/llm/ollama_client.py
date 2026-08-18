"""Local LLM client for Ollama (spoken to over its HTTP API via httpx).

We use httpx (already a dependency) instead of the ollama SDK to avoid an extra
package. The model is fully configurable; nothing is hardcoded. ``unload()``
asks Ollama to evict the model from VRAM (keep_alive=0) so the GPU is free for
the next pipeline stage.
"""
from __future__ import annotations

from typing import Dict, List, Optional

import httpx

from core.utils import get_logger

logger = get_logger(__name__)


class LLMError(Exception):
    """Raised for any LLM/Ollama failure, with a user-friendly message."""


class OllamaClient:
    def __init__(
        self,
        model: str = "qwen2.5:3b-instruct",
        host: str = "http://localhost:11434",
        temperature: float = 0.2,
        num_ctx: int = 4096,
        max_tokens: int = 1024,
        keep_alive: str = "5m",
        request_timeout: int = 180,
    ):
        self.model = model
        self.host = host.rstrip("/")
        self.temperature = temperature
        self.num_ctx = num_ctx
        self.max_tokens = max_tokens
        self.keep_alive = keep_alive
        self.request_timeout = request_timeout
        self._http: Optional[httpx.Client] = None

    def _client(self) -> httpx.Client:
        # trust_env=False -> ignore HTTP(S)_PROXY env vars. Critical: on this
        # machine a system proxy (WARP) otherwise intercepts localhost:11434.
        if self._http is None:
            self._http = httpx.Client(trust_env=False, timeout=self.request_timeout)
        return self._http

    # ------------------------------------------------------------- health --
    def is_available(self) -> bool:
        try:
            r = self._client().get(f"{self.host}/api/tags", timeout=5)
            return r.status_code == 200
        except Exception:
            return False

    def list_models(self) -> List[str]:
        try:
            r = self._client().get(f"{self.host}/api/tags", timeout=5)
            r.raise_for_status()
            return [m.get("name", "") for m in r.json().get("models", [])]
        except Exception as exc:
            raise LLMError(self._conn_hint(exc)) from exc

    def model_available(self) -> bool:
        try:
            models = self.list_models()
        except LLMError:
            return False
        # Match exact tag or the base name (e.g. "qwen2.5:3b-instruct").
        return any(m == self.model or m.split(":")[0] == self.model.split(":")[0]
                   for m in models)

    # -------------------------------------------------------------- chat ---
    def chat(
        self,
        messages: List[Dict[str, str]],
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
    ) -> str:
        payload = {
            "model": self.model,
            "messages": messages,
            "stream": False,
            "keep_alive": self.keep_alive,
            "options": {
                "temperature": self.temperature if temperature is None else temperature,
                "num_ctx": self.num_ctx,
                "num_predict": self.max_tokens if max_tokens is None else max_tokens,
            },
        }
        try:
            r = self._client().post(
                f"{self.host}/api/chat", json=payload, timeout=self.request_timeout
            )
            r.raise_for_status()
            data = r.json()
        except httpx.HTTPStatusError as exc:
            detail = exc.response.text[:300] if exc.response is not None else ""
            if exc.response is not None and exc.response.status_code == 404:
                raise LLMError(
                    f"Model '{self.model}' not found in Ollama. "
                    f"Pull it with:  ollama pull {self.model}"
                ) from exc
            raise LLMError(f"Ollama returned an error: {detail}") from exc
        except Exception as exc:
            raise LLMError(self._conn_hint(exc)) from exc

        return (data.get("message") or {}).get("content", "").strip()

    def generate(self, prompt: str, system: Optional[str] = None) -> str:
        messages = []
        if system:
            messages.append({"role": "system", "content": system})
        messages.append({"role": "user", "content": prompt})
        return self.chat(messages)

    # ------------------------------------------------------------ unload ---
    def unload(self) -> None:
        """Evict the model from memory (VRAM) by setting keep_alive=0."""
        try:
            self._client().post(
                f"{self.host}/api/generate",
                json={"model": self.model, "keep_alive": 0},
                timeout=15,
            )
            logger.info("Requested Ollama to unload '%s'.", self.model)
        except Exception as exc:  # pragma: no cover
            logger.debug("Ollama unload request failed: %s", exc)

    @staticmethod
    def _conn_hint(exc: Exception) -> str:
        return (
            "Could not reach Ollama. Make sure it is installed and running "
            "(run `ollama serve`, or launch the Ollama app). "
            f"Details: {exc}"
        )
