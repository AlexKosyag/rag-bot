"""RAG-движок: агентный цикл (seed-ретрив -> tool calling -> ответ).

Принудительный первый поиск (seed) гарантирует контекст даже если LLM не
зовёт инструмент; дальше модель сама решает, достаточно ли фрагментов или
нужен дополнительный поиск (docs/pipeline.md).
"""
from __future__ import annotations

from collections.abc import Iterator
from dataclasses import dataclass, field

from src.config import Config
from src.embedder import create_embedder
from src.generate.answer import (
    build_agent_messages,
    build_messages,
    format_sources,
    generate_answer,
    looks_like_refusal,
    stream_answer,
)
from src.generate.tools import (
    SEARCH_TOOL,
    SEARCH_TOOL_NAME,
    ToolCall,
    execute_search,
    merge_hits,
)
from src.llm.client import OllamaClient
from src.retrieve.gate import Decision, decide
from src.retrieve.search import RetrievalResult, retrieve
from src.store import Hit, Registry, VectorStore


@dataclass
class Prepared:
    """Решение принято: либо отказ, либо готовые сообщения для LLM."""

    decision: Decision
    retrieval: RetrievalResult
    messages: list[dict] = field(default_factory=list)

    @property
    def sources(self) -> list[dict]:
        return format_sources(self.retrieval.hits)


@dataclass
class AgentStep:
    """Один шаг поиска агента (seed или вызов инструмента)."""

    query: str
    source: str                 # seed | tool
    max_score: float
    tau: float
    n_hits: int
    answer: bool                # прошёл ли порог
    hits: list[Hit] = field(default_factory=list)


@dataclass
class AgentPrepared:
    """Агентный цикл завершён: финальные сообщения готовы к генерации."""

    query: str
    steps: list[AgentStep]
    hits: list[Hit]             # объединённые hits всех шагов
    messages: list[dict]        # диалог со всеми tool-шагами
    tool_calls: int = 0
    limit_hit: bool = False
    direct_answer: str | None = None  # модель ответила без финальной генерации

    @property
    def sources(self) -> list[dict]:
        return format_sources(self.hits)

    @property
    def max_score(self) -> float:
        return max((s.max_score for s in self.steps), default=0.0)


class RagEngine:
    def __init__(self, cfg: Config, embedder, store: VectorStore,
                 llm: OllamaClient, registry: Registry | None = None):
        self.cfg = cfg
        self.embedder = embedder
        self.store = store
        self.llm = llm
        self.registry = registry

    @classmethod
    def from_config(cls, cfg: Config) -> "RagEngine":
        embedder = create_embedder(cfg)
        store = VectorStore(cfg.paths.chroma_dir)
        registry = Registry(cfg.paths.registry_db)
        llm = OllamaClient(cfg.generation.base_url, cfg.models.llm,
                           cfg.generation.timeout_s)
        return cls(cfg, embedder, store, llm, registry)

    # ------------------------------------------------------------------ #
    def prepare(self, query: str, top_k: int | None = None,
                tau: float | None = None) -> Prepared:
        """Одношаговый ретривер + порог отказа (калибровка tau, юнит-тесты)."""
        r = retrieve(
            query,
            self.embedder,
            self.store,
            top_k=top_k or self.cfg.retrieval.top_k,
            registry=self.registry,
        )
        decision = decide(
            r,
            tau if tau is not None else self.cfg.retrieval.threshold,
            self.cfg.retrieval.delta,
        )
        messages = build_messages(query, r.hits) if decision.answer else []
        return Prepared(decision=decision, retrieval=r, messages=messages)

    # ------------------------------------------------------------------ #
    def prepare_agent(self, query: str, top_k: int | None = None,
                      tau: float | None = None) -> AgentPrepared:
        """Принудительный seed-ретрив -> цикл tool calling -> готовый диалог.

        Первый поиск выполняется всегда (даже если LLM не зовёт инструмент).
        Далее модель вызывает search_documents до max_tool_calls раз либо
        отвечает сразу. messages содержат всё, финальная генерация — отдельно
        (stream_agent/answer_agent).
        """
        k = top_k or self.cfg.retrieval.top_k
        t = tau if tau is not None else self.cfg.retrieval.threshold
        delta = self.cfg.retrieval.delta
        max_calls = self.cfg.agentic.max_tool_calls

        # 1. Принудительный seed-поиск
        seed = retrieve(query, self.embedder, self.store, top_k=k,
                        registry=self.registry)
        seed_decision = decide(seed, t, delta)
        steps = [AgentStep(
            query=query, source="seed", max_score=seed.max_score, tau=t,
            n_hits=len(seed.hits), answer=seed_decision.answer,
            hits=list(seed.hits),
        )]
        messages = build_agent_messages(query, seed.hits, seed.max_score, t)
        tool_calls = 0
        limit_hit = False
        all_hits: list[Hit] = list(seed.hits)

        if not self.cfg.agentic.enabled:
            # Агентный режим выключен: работаем как раньше (один ретрив + промпт)
            if seed_decision.answer:
                return AgentPrepared(query=query, steps=steps, hits=all_hits,
                                     messages=build_messages(query, seed.hits))
            return AgentPrepared(query=query, steps=steps, hits=all_hits,
                                 messages=[])

        # 2. Цикл tool calling
        refusal_retries = 0
        max_refusal_retries = 1
        for _ in range(max_calls + max_refusal_retries):
            if tool_calls >= max_calls:
                break  # бюджет вызовов исчерпан — финал без инструментов
            content, calls = self.llm.complete_with_tools(messages, [SEARCH_TOOL])
            if not calls:
                # Отказ без вызова инструмента при оставшемся бюджете — retry:
                # модель должна сначала поискать, потом отказываться
                if (looks_like_refusal(content or "")
                        and refusal_retries < max_refusal_retries
                        and tool_calls < max_calls):
                    refusal_retries += 1
                    messages.append({"role": "assistant", "content": content or ""})
                    messages.append({
                        "role": "user",
                        "content": "Прежде чем отказывать — вызови search_documents "
                                   "с перефразированным запросом (другая формулировка).",
                    })
                    continue
                # Модель решила, что контекста хватает — ответ уже сгенерирован
                messages.append({"role": "assistant", "content": content})
                return AgentPrepared(query=query, steps=steps,
                                     hits=merge_hits(*[s.hits for s in steps]),
                                     messages=messages,
                                     tool_calls=tool_calls,
                                     direct_answer=content)

            # Подтверждаем вызов в диалоге (assistant с tool_calls)
            messages.append({
                "role": "assistant",
                "content": content or "",
                "tool_calls": [
                    {"function": {"name": c["name"], "arguments": c["args"]}}
                    for c in calls
                ],
            })

            for call_dict in calls:
                name = call_dict.get("name", "")
                if name != SEARCH_TOOL_NAME:
                    messages.append({
                        "role": "tool",
                        "name": name,
                        "content": f"Неизвестный инструмент «{name}». "
                                   f"Доступен только search_documents.",
                    })
                    continue
                call = ToolCall(name=name, args=call_dict.get("args", {}))
                outcome = execute_search(call, self.embedder, self.store,
                                         self.registry, k, t, delta)
                tool_calls += 1
                steps.append(AgentStep(
                    query=outcome.call.args.get("query", ""),
                    source="tool",
                    max_score=outcome.decision.max_score,
                    tau=t,
                    n_hits=len(outcome.result.hits),
                    answer=outcome.decision.answer,
                    hits=list(outcome.result.hits),
                ))
                all_hits.extend(outcome.result.hits)
                messages.append(outcome.as_message())

        # 3. Лимит исчерпан — форсируем финал без инструментов
        limit_hit = True
        messages.append({
            "role": "user",
            "content": "Инструменты больше недоступны. Ответь строго по "
                       "уже найденному контексту.",
        })
        return AgentPrepared(query=query, steps=steps,
                             hits=merge_hits(*[s.hits for s in steps]),
                             messages=messages, tool_calls=tool_calls,
                             limit_hit=limit_hit)

    def stream_agent(self, prepared: AgentPrepared) -> Iterator[str]:
        """Потоковая финальная генерация агентного ответа."""
        if prepared.direct_answer is not None:
            yield prepared.direct_answer
            return
        yield from stream_answer(prepared.messages, self.llm)

    def answer_agent(self, prepared: AgentPrepared) -> str:
        """Пакетная финальная генерация агентного ответа (eval/CLI)."""
        if prepared.direct_answer is not None:
            return prepared.direct_answer
        return generate_answer(prepared.messages, self.llm)

    # ------------------------------------------------------------------ #
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
