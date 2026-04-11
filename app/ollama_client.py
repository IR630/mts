"""Async HTTP client for the local Ollama API (chat endpoint)."""

import logging
from typing import Any

import httpx

log = logging.getLogger(__name__)

# Type alias for the chat message list passed to /api/chat
Message = dict[str, str]

BASE_OPTIONS: dict[str, Any] = {
    "num_ctx": 4096,
    "num_predict": 1024,
}


class OllamaClient:
    def __init__(self, host: str, model: str):
        self.host = host.rstrip("/")
        self.model = model
        self._client = httpx.AsyncClient(base_url=self.host, timeout=300.0)

    async def chat(
        self,
        messages: list[Message],
        temperature: float = 0.1,
        extra_options: dict[str, Any] | None = None,
    ) -> str:
        """Call /api/chat with a full message history and return assistant text."""
        options: dict[str, Any] = {**BASE_OPTIONS, "temperature": temperature}
        if extra_options:
            options.update(extra_options)

        payload = {
            "model": self.model,
            "messages": messages,
            "stream": False,
            "options": options,
        }
        log.debug(
            "Ollama chat: model=%s messages=%d temp=%.2f",
            self.model,
            len(messages),
            temperature,
        )
        resp = await self._client.post("/api/chat", json=payload)
        resp.raise_for_status()
        data = resp.json()
        text: str = data["message"]["content"]
        log.debug("Ollama response length: %d", len(text))
        return text

    async def generate(
        self,
        prompt: str,
        system: str,
        temperature: float = 0.1,
    ) -> str:
        """Convenience wrapper: single-turn chat from (system, user) pair."""
        messages: list[Message] = [
            {"role": "system", "content": system},
            {"role": "user", "content": prompt},
        ]
        return await self.chat(messages, temperature=temperature)

    async def check_health(self) -> bool:
        """Return True if Ollama is reachable and the model is available."""
        try:
            resp = await self._client.get("/api/tags")
            resp.raise_for_status()
            models = [m["name"] for m in resp.json().get("models", [])]
            return any(self.model in m for m in models)
        except Exception:
            return False

    async def close(self) -> None:
        await self._client.aclose()
