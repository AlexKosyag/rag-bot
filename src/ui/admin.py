"""Админка: документы (загрузка/переиндексация), параметры, метрики."""
from __future__ import annotations

import json

import pandas as pd
import streamlit as st

from src.config import Config, save_retrieval
from src.engine import RagEngine
from src.ingest.indexer import index_file
from src.store import Registry


def _tab_documents(engine: RagEngine, cfg: Config) -> None:
    st.subheader("Загрузка документов")
    uploaded = st.file_uploader(
        "PDF / DOCX / MD (текстовый документ, без сканов)",
        type=["pdf", "docx", "md"],
        accept_multiple_files=True,
    )
    if st.button("⬆ Загрузить и проиндексировать", type="primary",
                 disabled=not uploaded, key="btn_upload"):
        registry = Registry(cfg.paths.registry_db)
        for f in uploaded:
            target = cfg.paths.raw_dir / f.name
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(f.getbuffer())
            with st.spinner(f"Индексация {f.name}…"):
                res = index_file(cfg, target, engine.embedder, engine.store, registry)
            if res.status == "error":
                st.error(f"{f.name}: {res.detail}")
            else:
                st.success(f"{f.name}: {res.chunks} чанков ({res.status})")
        st.rerun()

    st.subheader("Документы в базе")
    registry = Registry(cfg.paths.registry_db)
    docs = registry.list()
    if not docs:
        st.info("Документов пока нет — загрузите первый или запустите "
                "`python -m src.ingest`.")
        return

    df = pd.DataFrame(docs)[
        ["name", "status", "chunks_count", "error", "uploaded_at"]
    ].rename(columns={
        "name": "Имя", "status": "Статус", "chunks_count": "Чанков",
        "error": "Ошибка", "uploaded_at": "Загружен (UTC)",
    })
    st.dataframe(df, use_container_width=True, hide_index=True)

    names = [d["name"] for d in docs]
    col1, col2, col3 = st.columns([2, 1, 1])
    with col1:
        selected = st.selectbox("Документ", names, key="sel_doc")
    doc = next((d for d in docs if d["name"] == selected), None)
    if doc:
        with col2:
            if st.button("🔄 Переиндексировать", key="btn_reindex"):
                target = cfg.paths.raw_dir / doc["name"]
                if not target.exists():
                    st.error(f"Файл {target} не найден в {cfg.paths.raw_dir}")
                else:
                    with st.spinner(f"Переиндексация {doc['name']}…"):
                        res = index_file(cfg, target, engine.embedder,
                                         engine.store, registry, force=True)
                    st.success(f"{doc['name']}: {res.chunks} чанков")
                    st.rerun()
        with col3:
            if st.button("🗑 Удалить", key="btn_delete"):
                engine.store.delete_doc(doc["id"])
                registry.delete(doc["id"])
                st.rerun()


def _tab_params(engine: RagEngine, cfg: Config) -> None:
    st.subheader("Параметры поиска")
    st.caption("Изменения действуют в текущей сессии; "
               "«Сохранить в конфиг» фиксирует их в config.yaml.")

    r = cfg.retrieval
    r.top_k = st.slider("top_k (число чанков в контексте)", 3, 10, r.top_k)
    r.threshold = st.slider("Порог отказа τ", 0.0, 1.0, r.threshold, 0.01)
    r.delta = st.slider("Разрыв top1–top2 (delta)", 0.0, 0.3, r.delta, 0.01)

    col1, col2 = st.columns(2)
    with col1:
        if st.button("💾 Сохранить в конфиг", key="btn_save_cfg"):
            save_retrieval(cfg)
            st.success("config.yaml обновлён")
    with col2:
        st.caption(f"τ={r.threshold:.2f} · top_k={r.top_k} · delta={r.delta:.2f}")
        st.caption("⚠ τ нужно калибровать на dev-наборе — см. docs/evaluation.md")

    st.subheader("Модели")
    st.code(
        f"embedding: {cfg.models.embedding}\n"
        f"llm:       {cfg.models.llm}  ({cfg.generation.base_url})\n"
        f"chunk:     {cfg.chunking.size_tokens}/{cfg.chunking.overlap_tokens} токенов",
        language="text",
    )


def _tab_metrics(engine: RagEngine, cfg: Config) -> None:
    st.subheader("Метрики")
    report_path = cfg.paths.reports_dir / "metrics.json"
    if not report_path.exists():
        st.info(
            "Отчёта нет. Запустите из корня проекта:\n\n"
            "`python -m src.evaluate --qa data/qa/test.json` "
            "(+ `--llm --judge` для Faithfulness и latency)"
        )
        return
    report = json.loads(report_path.read_text(encoding="utf-8"))
    st.caption(f"Параметры запуска: {report.get('params', {})}")
    for section in ("retrieval", "refusal", "generation"):
        if section in report:
            st.markdown(f"**{section}**")
            st.json(report[section])


def render_admin(engine: RagEngine, cfg: Config) -> None:
    st.title("🛠 Админка")
    tab_docs, tab_params, tab_metrics = st.tabs(
        ["Документы", "Параметры", "Метрики"]
    )
    with tab_docs:
        _tab_documents(engine, cfg)
    with tab_params:
        _tab_params(engine, cfg)
    with tab_metrics:
        _tab_metrics(engine, cfg)
