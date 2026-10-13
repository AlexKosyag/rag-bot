"""Чат: стриминг ответа, источники, отказ «Нет в документах»."""
from __future__ import annotations

import streamlit as st

from src.engine import Prepared, RagEngine
from src.llm.client import LLMError

ROLE_LABEL = {"ok": "ответ по контексту", "low_margin": "низкий разрыв top1-top2"}


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


def _render_assistant_message(msg: dict) -> None:
    st.markdown(msg["content"])
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
            prepared: Prepared = engine.prepare(query)
            d = prepared.decision

            if not d.answer:
                note = {
                    "empty": "база пуста",
                    "below_threshold": f"score {d.max_score:.3f} < порога {d.tau:.2f}",
                }.get(d.reason, d.reason)
                st.session_state.messages.append({
                    "role": "assistant",
                    "content": "Нет в документах.",
                    "refused": True,
                    "sources": prepared.sources,
                    "max_score": d.max_score,
                    "note": note,
                })
                _fragments_block(prepared.sources, d.max_score)
                st.caption(f"ℹ️ {note}")
            else:
                try:
                    answer = st.write_stream(engine.stream(prepared))
                except LLMError as exc:
                    answer = ""
                    st.error(str(exc))
                note = ROLE_LABEL.get(d.reason, "")
                sources = prepared.sources
                st.session_state.messages.append({
                    "role": "assistant",
                    "content": answer or "_(пустой ответ)_",
                    "refused": False,
                    "sources": sources,
                    "note": note,
                })
                if answer:
                    _sources_block(sources)
                    if note:
                        st.caption(f"⚠️ {note}")
