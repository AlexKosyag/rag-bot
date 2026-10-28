"""Тесты агентного цикла: seed-ретрив, tool calling, лимит итераций."""
import numpy as np

from src.config import AgenticCfg, Config, RetrievalCfg
from src.engine import RagEngine
from src.store import Registry, VectorStore


def _vec(seed: int) -> list[float]:
    v = np.zeros(8, dtype=np.float32)
    v[seed] = 1.0
    return v


class _FakeEmbedder:
    def encode_query(self, query: str):
        return _vec(0)


class _FakeLLM:
    """Фейковый LLM: очередь (content, tool_calls) для complete_with_tools."""

    def __init__(self, script: list[tuple[str, list[dict]]]):
        self._script = script
        self._calls = 0

    def complete_with_tools(self, messages, tools):
        if self._calls < len(self._script):
            result = self._script[self._calls]
        else:
            result = ("Готово.", [])
        self._calls += 1
        return result

    def complete(self, messages):
        return "Финальный ответ без инструментов."


def _make_engine(tmp_path, script, max_tool_calls=3, enabled=True):
    cfg = Config(
        retrieval=RetrievalCfg(top_k=5, threshold=0.50, delta=0.05),
        agentic=AgenticCfg(enabled=enabled, max_tool_calls=max_tool_calls),
    )
    store = VectorStore(tmp_path / "chroma")
    reg = Registry(tmp_path / "registry.db")
    store.upsert(
        ids=["d1:0:0"],
        texts=["отпуск 28 дней"],
        embeddings=[_vec(0)],
        metadatas=[{"chunk_id": "d1:0", "doc_id": "d1", "doc_name": "ustav.md",
                    "heading": "Отпуск", "page": 0}],
    )
    reg.save_chunks([
        {"id": "d1:0", "doc_id": "d1", "chunk_index": 0,
         "heading": "Отпуск", "page": 0, "text": "Ежегодный отпуск 28 дней"},
    ])
    llm = _FakeLLM(script)
    return RagEngine(cfg, _FakeEmbedder(), store, llm, reg), llm


def test_seed_always_runs(tmp_path):
    engine, llm = _make_engine(tmp_path, script=[("Ответ без вызовов.", [])])
    prep = engine.prepare_agent("сколько отпуск?")
    assert len(prep.steps) == 1
    assert prep.steps[0].source == "seed"
    assert prep.steps[0].n_hits >= 1
    assert prep.tool_calls == 0
    assert prep.direct_answer == "Ответ без вызовов."


def test_tool_call_extends_search(tmp_path):
    tool_call = {
        "name": "search_documents",
        "args": {"query": "уточнение по командировкам"},
    }
    engine, llm = _make_engine(
        tmp_path,
        script=[("", [tool_call]), ("Итоговый ответ.", [])],
    )
    prep = engine.prepare_agent("вопрос")
    # seed + tool
    assert len(prep.steps) == 2
    assert prep.steps[1].source == "tool"
    assert "командировкам" in prep.steps[1].query
    assert prep.tool_calls == 1
    assert prep.direct_answer == "Итоговый ответ."
    # hits агрегированы (дедуп)
    assert prep.hits
    assert prep.max_score >= prep.steps[0].max_score


def test_tool_loop_stops_at_limit(tmp_path):
    # LLM упорно зовёт инструмент — ограничиваем max_tool_calls=1
    tc = {"name": "search_documents", "args": {"query": "ещё"}}
    engine, llm = _make_engine(
        tmp_path,
        script=[("", [tc]), ("", [tc]), ("", [tc]), ("", [tc])],
        max_tool_calls=1,
    )
    prep = engine.prepare_agent("вопрос")
    # seed + ровно 1 tool-шаг
    assert len(prep.steps) == 2
    assert prep.tool_calls == 1
    assert prep.limit_hit
    # финальная генерация — отдельный вызов complete()
    answer = engine.answer_agent(prep)
    assert answer == "Финальный ответ без инструментов."


def test_refusal_without_tool_call_triggers_retry(tmp_path):
    # Модель отвечает отказом без вызова инструмента -> retry с hint,
    # затем вызов инструмента, затем финальный ответ
    tc = {"name": "search_documents", "args": {"query": "уточнение"}}
    engine, llm = _make_engine(
        tmp_path,
        script=[
            ("Нет в документах.", []),   # отказ без вызова -> retry
            ("", [tc]),                  # вызов после hint
            ("Итог.", []),               # финальный ответ
        ],
        max_tool_calls=2,
    )
    prep = engine.prepare_agent("вопрос")
    # seed + 1 tool-шаг (retry не создаёт шаг)
    assert len(prep.steps) == 2
    assert prep.steps[1].source == "tool"
    assert prep.tool_calls == 1
    assert prep.direct_answer == "Итог."
    # hint-сообщение в диалоге
    roles = [m["role"] for m in prep.messages]
    assert "tool" in roles


def test_unknown_tool_ignored(tmp_path):
    bad = {"name": "unknown_tool", "args": {}}
    engine, llm = _make_engine(
        tmp_path,
        script=[("", [bad]), ("Ответ.", [])],
        max_tool_calls=2,
    )
    prep = engine.prepare_agent("вопрос")
    # seed + никакого tool-шага (unknown отброшен)
    assert len(prep.steps) == 1
    assert prep.tool_calls == 0
    assert prep.direct_answer == "Ответ."


def test_agentic_disabled_uses_one_shot(tmp_path):
    engine, llm = _make_engine(
        tmp_path,
        script=[("Не должно вызваться.", [])],
        enabled=False,
    )
    prep = engine.prepare_agent("сколько отпуск?")
    # только seed, без вызовов LLM
    assert len(prep.steps) == 1
    assert llm._calls == 0
    # при этом hits заполнены и messages готовы (one-shot)
    assert prep.hits
    assert prep.messages


def test_answer_agent_uses_direct_answer(tmp_path):
    engine, _ = _make_engine(tmp_path, script=[("Прямой ответ.", [])])
    prep = engine.prepare_agent("вопрос")
    assert engine.answer_agent(prep) == "Прямой ответ."


def test_stream_agent_yields_direct_answer(tmp_path):
    engine, _ = _make_engine(tmp_path, script=[("Прямой ответ.", [])])
    prep = engine.prepare_agent("вопрос")
    chunks = list(engine.stream_agent(prep))
    assert "".join(chunks) == "Прямой ответ."
