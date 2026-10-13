"""Streamlit-приложение: чат + админка. Запуск: streamlit run app.py"""
from __future__ import annotations

import streamlit as st

st.set_page_config(page_title="RAG-бот", page_icon="💬", layout="wide")

from src.config import load_config  # noqa: E402
from src.engine import RagEngine  # noqa: E402
from src.ui.admin import render_admin  # noqa: E402
from src.ui.chat import render_chat  # noqa: E402


@st.cache_resource(show_spinner="Загрузка конфигурации и хранилища…")
def get_engine() -> RagEngine:
    """Один движок на процесс: Chroma и кэш моделей переиспользуются."""
    return RagEngine.from_config(load_config())


def main() -> None:
    engine = get_engine()
    cfg = engine.cfg

    with st.sidebar:
        st.title("💬 RAG-бот")
        st.caption("Ответы **только** по загруженным документам")
        page = st.radio("Разделы", ["Чат", "Админка"], horizontal=True)
        st.divider()
        ok, msg = engine.health()
        if ok:
            st.success(msg)
        else:
            st.warning(msg)
        st.caption(f"Чанков в базе: {engine.store.count()}")
        st.caption(f"Модель эмбеддингов: `{cfg.models.embedding}`")
        with st.expander("Подсказки"):
            st.markdown(
                "- Документы → загрузка и переиндексация\n"
                "- Параметры → top_k и порог отказа τ\n"
                "- Оценка: `python -m src.evaluate`"
            )

    if page == "Чат":
        render_chat(engine)
    else:
        render_admin(engine, cfg)


if __name__ == "__main__":
    main()
