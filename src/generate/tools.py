"""Инструменты агентного RAG: search_documents для tool calling LLM.

Диспетчер поверх существующего retrieve()+decide(): LLM сама решает, когда
и с каким запросом искать дополнительные документы. Порог tau остаётся
частью данных инструмента (лучший score / порог) — решение об отказе
делегировано модели, но калибруемый сигнал сохраняется.
"""
from __future__ import annotations

from dataclasses import dataclass

from src.embedder import AnyEmbedder
from src.retrieve.gate import Decision, decide
from src.retrieve.search import RetrievalResult, retrieve
from src.store import Hit, Registry, VectorStore

SEARCH_TOOL_NAME = "search_documents"

SEARCH_TOOL: dict = {
    "type": "function",
    "function": {
        "name": SEARCH_TOOL_NAME,
        "description": (
            "Поиск по базе внутренних документов организации. "
            "Возвращает релевантные фрагменты с оценкой близости и порогом. "
            "Вызывай с перефразированным запросом, если контекста недостаточно, "
            "вопрос составной (несколько документов) или нужен другой раздел."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "Поисковый запрос: суть нужной информации",
                },
                "top_k": {
                    "type": "integer",
                    "description": "Число фрагментов (1..10, по умолчанию 5)",
                },
            },
            "required": ["query"],
        },
    },
}


@dataclass
class ToolCall:
    """Разобранный вызов инструмента из ответа LLM."""

    name: str
    args: dict


@dataclass
class ToolOutcome:
    """Результат исполненного вызова: ретрив + решение порога."""

    call: ToolCall
    result: RetrievalResult
    decision: Decision

    def as_message(self) -> dict:
        """Сообщение role=tool для диалога с LLM."""
        return {
            "role": "tool",
            "name": self.call.name,
            "content": self.render(),
        }

    def render(self) -> str:
        d = self.decision
        n = len(self.result.hits)
        if n == 0:
            return (
                f"Фрагментов по запросу «{self.call.args.get('query', '')}» "
                f"в базе нет."
            )
        head = (
            f"Найдено фрагментов: {n}. Лучший score {d.max_score:.2f} "
            f"при пороге {d.tau:.2f} — "
            + ("релевантно." if d.answer else
               "ниже порога: как основание для ответа не подходи, "
               "поищи иначе либо ответь «Нет в документах.»")
        )
        blocks = [head]
        for i, h in enumerate(self.result.hits, 1):
            page = f", стр. {h.page}" if h.page else ""
            blocks.append(
                f"### [{i}] {h.doc_name} — «{h.heading}»{page} "
                f"(score={h.score:.2f})\n{h.text}"
            )
        return "\n\n".join(blocks)


def execute_search(
    call: ToolCall,
    embedder: AnyEmbedder,
    store: VectorStore,
    registry: Registry | None,
    top_k_default: int,
    tau: float,
    delta: float = 0.0,
) -> ToolOutcome:
    """Исполняет search_documents: retrieve() + decide() с аргументами LLM."""
    query = str(call.args.get("query") or "").strip()
    if not query:
        empty = ToolCall(name=call.name, args={"query": ""})
        res = RetrievalResult(query="", hits=[])
        return ToolOutcome(call=empty, result=res,
                           decision=decide(res, tau, delta))
    try:
        top_k = int(call.args.get("top_k") or top_k_default)
    except (TypeError, ValueError):
        top_k = top_k_default
    top_k = max(1, min(top_k, 10))
    call = ToolCall(name=call.name, args={"query": query, "top_k": top_k})
    res = retrieve(query, embedder, store, top_k=top_k,
                   registry=registry)
    return ToolOutcome(call=call, result=res, decision=decide(res, tau, delta))


def merge_hits(*hit_lists: list[Hit]) -> list[Hit]:
    """Объединение hits всех поисков: дедуп по chunk_id, сортировка по score."""
    best: dict[str, Hit] = {}
    for hits in hit_lists:
        for h in hits:
            cur = best.get(h.chunk_id)
            if cur is None or h.score > cur.score:
                best[h.chunk_id] = h
    merged = sorted(best.values(), key=lambda h: -h.score)
    return [
        Hit(chunk_id=h.chunk_id, doc_id=h.doc_id, doc_name=h.doc_name,
            text=h.text, heading=h.heading, page=h.page, score=h.score,
            rank=i)
        for i, h in enumerate(merged)
    ]
