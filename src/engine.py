"""RAG-движок: подготовка ответа (ретрив -> порог -> промпт) и стриминг."""
from __future__ import annotations

from collections.abc import Iterator
from dataclasses import dataclass, field

from src.config import Config
from src.embedder import Embedder
from src.generate.answer import (
    build_messages,
    format_sources,
    generate_answer,
    looks_like_refusal,
    stream_answer,
)
from src.llm.client import OllamaClient
from src.retrieve.gate import Decision, decide
from src.retrieve.search import RetrievalResult, retrieve
from src.store import VectorStore


@dataclass
class Prepared:
    """Решение принято: либо отказ, либо готовые сообщения для LLM."""

    decision: Decision
    retrieval: RetrievalResult
    messages: list[dict] = field(default_factory=list)

    @property
    def sources(self) -> list[dict]:
        return format_sources(self.retrieval.hits)


class RagEngine:
    def __init__(self, cfg: Config, embedder: Embedder, store: VectorStore,
                 llm: OllamaClient):
        self.cfg = cfg
        self.embedder = embedder
        self.store = store
        self.llm = llm

    @classmethod
    def from_config(cls, cfg: Config) -> "RagEngine":
        embedder = Embedder(cfg.models.embedding, cfg.models.device)
        store = VectorStore(cfg.paths.chroma_dir)
        llm = OllamaClient(cfg.generation.base_url, cfg.models.llm,
                           cfg.generation.timeout_s)
        return cls(cfg, embedder, store, llm)

    # ------------------------------------------------------------------ #
    def prepare(self, query: str, top_k: int | None = None,
                tau: float | None = None) -> Prepared:
        """Ретривер + порог отказа. Без вызова LLM (для отказов)."""
        r = retrieve(
            query,
            self.embedder,
            self.store,
            top_k=top_k or self.cfg.retrieval.top_k,
        )
        decision = decide(
            r,
            tau if tau is not None else self.cfg.retrieval.threshold,
            self.cfg.retrieval.delta,
        )
        messages = build_messages(query, r.hits) if decision.answer else []
        return Prepared(decision=decision, retrieval=r, messages=messages)

    def stream(self, prepared: Prepared) -> Iterator[str]:
        """Потоковая генерация ответа (вызывать только при prepared.messages)."""
        yield from stream_answer(prepared.messages, self.llm)

    def answer(self, prepared: Prepared) -> str:
        """Пакетная генерация (eval/CLI)."""
        return generate_answer(prepared.messages, self.llm)

    def answer_is_refusal(self, text: str) -> bool:
        return looks_like_refusal(text)

    # ------------------------------------------------------------------ #
    def health(self) -> tuple[bool, str]:
        return self.llm.health()
