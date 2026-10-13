"""Эмбеддинги: два бэкенда под одной абстракцией.

- SentenceTransformerEmbedder — sentence-transformers (Hugging Face);
  основной бэкенд, когда HF доступен (напр. intfloat/multilingual-e5-base).
- OllamaEmbedder — модели через Ollama (/api/embed); fallback, когда
  Hugging Face недоступен из среды (обрывы CDN) или не хочется второй
  стек загрузки (напр. qwen3-embedding:0.6b, bge-m3).

Оба бэкенда: L2-нормализация -> cosine similarity = 1 - distance в Chroma.
"""
from __future__ import annotations

import math
from typing import TYPE_CHECKING

import numpy as np
import requests

if TYPE_CHECKING:
    from src.config import Config, ModelsCfg

# Символов на BPE-токен (консервативно для русского): fallback подсчёта,
# если токенизатор недоступен. Завышение -> чанки чуть меньше (безопаснее).
CHARS_PER_TOKEN = 2.0


def estimate_tokens(text: str) -> int:
    return max(1, math.ceil(len(text) / CHARS_PER_TOKEN))


def _l2_normalize(arr: np.ndarray) -> np.ndarray:
    arr = np.asarray(arr, dtype=np.float32)
    norms = np.linalg.norm(arr, axis=-1, keepdims=True)
    norms[norms == 0] = 1.0
    return arr / norms


class SentenceTransformerEmbedder:
    """Ленивая загрузка модели HF, префиксы E5, нормализация."""

    def __init__(self, model_name: str, device: str | None = "cpu"):
        self.model_name = model_name
        self.device = device
        self.backend = "sentence-transformers"
        self._model = None
        if "e5" in model_name.lower():
            self.query_prefix = "query: "
            self.passage_prefix = "passage: "
        else:
            self.query_prefix = ""
            self.passage_prefix = ""

    @property
    def model(self):
        if self._model is None:
            from sentence_transformers import SentenceTransformer

            self._model = SentenceTransformer(self.model_name, device=self.device or None)
        return self._model

    @property
    def dim(self) -> int:
        return int(self.model.get_sentence_embedding_dimension())

    def encode_passages(self, texts: list[str]) -> np.ndarray:
        emb = self.model.encode(
            [self.passage_prefix + t for t in texts],
            batch_size=32,
            normalize_embeddings=True,
            show_progress_bar=False,
            convert_to_numpy=True,
        )
        return _l2_normalize(emb)

    def encode_query(self, text: str) -> np.ndarray:
        emb = self.model.encode(
            [self.query_prefix + text],
            normalize_embeddings=True,
            show_progress_bar=False,
            convert_to_numpy=True,
        )
        return _l2_normalize(emb)[0]

    def count_tokens(self, text: str) -> int:
        try:
            return len(self.model.tokenizer.encode(text, add_special_tokens=False))
        except Exception:
            return estimate_tokens(text)


class OllamaEmbedder:
    """Эмбеддинги через Ollama: POST /api/embed, /api/tokenize."""

    def __init__(self, model_name: str, base_url: str = "http://localhost:11434",
                 timeout_s: int = 180):
        self.model_name = model_name
        self.base_url = base_url.rstrip("/")
        self.timeout_s = timeout_s
        self.backend = "ollama"
        self._dim: int | None = None

    def _post(self, path: str, payload: dict) -> dict:
        try:
            r = requests.post(
                f"{self.base_url}{path}",
                json=payload,
                timeout=(10, self.timeout_s),
            )
            r.raise_for_status()
        except requests.RequestException as exc:
            raise RuntimeError(
                f"Ollama-эмбеддинги недоступны ({self.base_url}{path}): {exc}"
            ) from exc
        data = r.json()
        if "error" in data:
            raise RuntimeError(f"Ollama: {data['error']}")
        return data

    def _embed(self, texts: list[str]) -> np.ndarray:
        data = self._post("/api/embed", {
            "model": self.model_name,
            "input": texts,
            "keep_alive": "30m",  # не выгружать модель между запросами
        })
        embs = data.get("embeddings", [])
        if not embs:
            raise RuntimeError(f"Ollama вернул пустые эмбеддинги для {self.model_name}")
        return _l2_normalize(np.asarray(embs, dtype=np.float32))

    @property
    def dim(self) -> int:
        if self._dim is None:
            self._dim = int(self._embed(["dimension probe"]).shape[1])
        return self._dim

    def encode_passages(self, texts: list[str]) -> np.ndarray:
        out = []
        for i in range(0, len(texts), 32):  # батчи, чтобы не раздувать запрос
            out.append(self._embed(texts[i:i + 32]))
        return np.vstack(out)

    def encode_query(self, text: str) -> np.ndarray:
        return self._embed([text])[0]

    def count_tokens(self, text: str) -> int:
        try:
            data = self._post("/api/tokenize", {"model": self.model_name, "text": text})
            return len(data.get("tokens", []))
        except Exception:
            return estimate_tokens(text)


def create_embedder(cfg: "Config") -> SentenceTransformerEmbedder | OllamaEmbedder:
    """Фабрика по config.yaml: models.embedding_backend = ollama | sentence_transformers."""
    models: "ModelsCfg" = cfg.models
    if models.embedding_backend == "ollama":
        return OllamaEmbedder(models.embedding, cfg.generation.base_url)
    return SentenceTransformerEmbedder(models.embedding, models.device)


# Оба бэкенда имеют единый интерфейс (encode_passages/encode_query/
# count_tokens/dim) — тип для аннотаций в подсистемах:
AnyEmbedder = SentenceTransformerEmbedder | OllamaEmbedder
