"""Обёртка над sentence-transformers: эмбеддинги запросов и чанков."""
from __future__ import annotations

import math

import numpy as np

# Сколько символов текста приходится примерно на один BPE-токен (консервативно
# для русского языка; используется только как fallback, если токенизатор
# модели недоступен). Завышение -> чанки чуть меньше, что безопаснее для
# окна внимания модели (риск молчаливого усечения).
CHARS_PER_TOKEN = 2.0


def estimate_tokens(text: str) -> int:
    return max(1, math.ceil(len(text) / CHARS_PER_TOKEN))


class Embedder:
    """Ленивая загрузка модели, префиксы E5, L2-нормализация векторов."""

    def __init__(self, model_name: str, device: str | None = "cpu"):
        self.model_name = model_name
        self.device = device
        self._model = None
        # e5 требует префиксов "query: "/"passage: ", остальные модели — нет.
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

            self._model = SentenceTransformer(
                self.model_name, device=self.device or None
            )
        return self._model

    @property
    def dim(self) -> int:
        return int(self.model.get_sentence_embedding_dimension())

    def encode_passages(self, texts: list[str]) -> np.ndarray:
        prefixed = [self.passage_prefix + t for t in texts]
        emb = self.model.encode(
            prefixed,
            batch_size=32,
            normalize_embeddings=True,
            show_progress_bar=False,
            convert_to_numpy=True,
        )
        return np.asarray(emb, dtype=np.float32)

    def encode_query(self, text: str) -> np.ndarray:
        emb = self.model.encode(
            [self.query_prefix + text],
            normalize_embeddings=True,
            show_progress_bar=False,
            convert_to_numpy=True,
        )
        return np.asarray(emb[0], dtype=np.float32)

    def count_tokens(self, text: str) -> int:
        """Реальный подсчёт токенов токенизатором модели (fallback — оценка)."""
        try:
            tok = self.model.tokenizer
            return len(tok.encode(text, add_special_tokens=False))
        except Exception:
            return estimate_tokens(text)
