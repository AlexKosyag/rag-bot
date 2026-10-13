"""Клиент Ollama (/api/chat): потоковая и пакетная генерация."""
from __future__ import annotations

import json
from collections.abc import Iterator

import requests


class LLMError(Exception):
    """Ошибка обращения к LLM (недоступна, таймаут, некорректный ответ)."""


class OllamaClient:
    def __init__(self, base_url: str, model: str, timeout_s: int = 90):
        self.base_url = base_url.rstrip("/")
        self.model = model
        self.timeout_s = timeout_s

    # ------------------------------------------------------------------ #
    def health(self) -> tuple[bool, str]:
        """(доступен ли сервер, сообщение). Проверяет наличие модели."""
        try:
            r = requests.get(f"{self.base_url}/api/tags", timeout=5)
            r.raise_for_status()
            names = [m.get("name", "") for m in r.json().get("models", [])]
        except Exception as exc:
            return False, f"Ollama недоступна ({self.base_url}): {exc}"
        if not any(n == self.model or n.split(":")[0] == self.model for n in names):
            return False, (
                f"Модель «{self.model}» не найдена. Доступно: "
                + (", ".join(names) or "нет моделей")
                + f". Скачайте: ollama pull {self.model}"
            )
        return True, f"Ollama: {self.model}"

    # ------------------------------------------------------------------ #
    def _payload(self, messages: list[dict], stream: bool) -> dict:
        return {
            "model": self.model,
            "messages": messages,
            "stream": stream,
            "options": {"temperature": 0.2},
            # держим модель в памяти между запросами чата (иначе Ollama
            # выгружает её через 5 минут простоя и первый ответ платит за загрузку)
            "keep_alive": "30m",
        }

    def stream(self, messages: list[dict]) -> Iterator[str]:
        """Потоковая генерация: отдаёт куски текста по мере появления."""
        try:
            resp = requests.post(
                f"{self.base_url}/api/chat",
                json=self._payload(messages, stream=True),
                stream=True,
                timeout=(10, self.timeout_s),
            )
            resp.raise_for_status()
        except requests.RequestException as exc:
            raise LLMError(f"Не удалось вызвать LLM ({self.base_url}): {exc}") from exc

        for line in resp.iter_lines(decode_unicode=True):
            if not line:
                continue
            try:
                data = json.loads(line)
            except json.JSONDecodeError:
                continue
            if "error" in data:
                raise LLMError(str(data["error"]))
            content = data.get("message", {}).get("content", "")
            if content:
                yield content
            if data.get("done"):
                break

    def complete(self, messages: list[dict]) -> str:
        """Пакетная генерация (eval, judge)."""
        try:
            resp = requests.post(
                f"{self.base_url}/api/chat",
                json=self._payload(messages, stream=False),
                timeout=(10, self.timeout_s),
            )
            resp.raise_for_status()
            data = resp.json()
        except requests.RequestException as exc:
            raise LLMError(f"Не удалось вызвать LLM ({self.base_url}): {exc}") from exc
        if "error" in data:
            raise LLMError(str(data["error"]))
        return data.get("message", {}).get("content", "")
