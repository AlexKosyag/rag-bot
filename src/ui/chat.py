"""Чат: стриминг агентного ответа, поиск, источники, отказ «Нет в документах»."""
from __future__ import annotations

import streamlit as st

from src.engine import AgentPrepared, RagEngine
from src.llm.client import LLMError


def _sources_block(sources: list[dict]) -> None:
    with st.expander(f"📎 Источники ({len(sources)})"):
        for s in sources:
            page = f", стр. {s['page']}" if s.get("page") else ""
            st.markdown(
                f"**{s['doc_name']}** — «{s['heading']}»{page} · "
                f"score `{s['score']}`"
            )
            st.caption(s["preview"] + ("…" if len(s["preview"]) >= 240 else ""))


def _fragments_block(sources: list[dict], max_score: float) -> None:
    if not sources:
        st.caption("В базе нет ни одного близкого фрагмента.")
        return
    with st.expander("🔍 Показать найденные фрагменты"):
        st.caption(f"Лучший score {max_score:.3f} — ниже порога.")
        for s in sources:
            st.markdown(f"**{s['doc_name']}** — «{s['heading']}» · score `{s['score']}`")
            st.caption(s["preview"] + "…")


def _steps_block(steps: list[dict]) -> None:
    if len(steps) <= 1:
        return
    with st.expander(f"🔍 Поиски агента ({len(steps)})"):
        for s in steps:
            icon = "🌱" if s["source"] == "seed" else "🔎"
            st.markdown(
                f"{icon} **{s['source']}** · query=`{s['query']}` · "
                f"score `{s['max_score']:.3f}` · фрагментов `{s['n_hits']}`"
            )


def _render_assistant_message(msg: dict) -> None:
    st.markdown(msg["content"])
    _steps_block(msg.get("steps", []))
    if msg.get("refused"):
        _fragments_block(msg.get("sources", []), msg.get("max_score", 0.0))
    elif msg.get("sources"):
        _sources_block(msg["sources"])
    if msg.get("note"):
        st.caption(f"⚠️ {msg['note']}")


def render_chat(engine: RagEngine) -> None:
    st.title("💬 Чат по документам")

    if "messages" not in st.session_state:
        st.session_state.messages = []

    cols = st.columns([1, 4])
    with cols[0]:
        if st.button("🗑 Новый диалог", disabled=not st.session_state.messages):
            st.session_state.messages = []
            st.rerun()

    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            if msg["role"] == "user":
                st.markdown(msg["content"])
            else:
                _render_assistant_message(msg)

    if query := st.chat_input("Вопрос по внутренним документам…"):
        st.session_state.messages.append({"role": "user", "content": query})
        with st.chat_message("user"):
            st.markdown(query)

        with st.chat_message("assistant"):
            prepared: AgentPrepared = engine.prepare_agent(query)
            steps_data = [
                {
                    "source": s.source,
                    "query": s.query,
                    "max_score": round(s.max_score, 3),
                    "n_hits": s.n_hits,
                }
                for s in prepared.steps
            ]
            _steps_block(steps_data)

            if not prepared.hits:
                st.session_state.messages.append({
                    "role": "assistant",
                    "content": "Нет в документах.",
                    "refused": True,
                    "sources": [],
                    "max_score": prepared.max_score,
                    "steps": steps_data,
                    "note": "база пуста или нет фрагментов",
                })
                _fragments_block([], prepared.max_score)
                st.caption("ℹ️ база пуста или нет фрагментов")
            else:
                try:
                    answer = st.write_stream(engine.stream_agent(prepared))
                except LLMError as exc:
                    answer = ""
                    st.error(str(exc))

                refused = engine.answer_is_refusal(answer or "")
                sources = prepared.sources
                note = ""
                if refused:
                    note = f"модель отказалась, лучший score {prepared.max_score:.3f}"
                elif prepared.limit_hit:
                    note = "лимит вызовов инструмента исчерпан"

                st.session_state.messages.append({
                    "role": "assistant",
                    "content": answer or "_(пустой ответ)_",
                    "refused": refused,
                    "sources": [] if refused else sources,
                    "max_score": prepared.max_score,
                    "steps": steps_data,
                    "note": note,
                })
                if answer:
                    if refused:
                        _fragments_block(sources, prepared.max_score)
                    else:
                        _sources_block(sources)
                    if note:
                        st.caption(f"⚠️ {note}")
