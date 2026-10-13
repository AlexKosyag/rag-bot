# Запуск, структура репозитория, инфраструктура

## 1. Целевая структура репозитория

```
rag-bot/
├── app.py                     # точка входа Streamlit (чат + админка)
├── config.yaml                # параметры (модели, top_k, порог, чанкинг)
├── .env.example               # OLLAMA_BASE_URL, ключи API (опц.)
├── requirements.txt           # зафиксированные версии зависимостей
├── Dockerfile
├── docker-compose.yml         # app + ollama
├── README.md                  # быстрый старт, скриншоты, метрики
├── data/
│   ├── raw/                   # загруженные документы (не коммитится)
│   ├── chroma/                # векторное хранилище (не коммитится)
│   ├── registry.db            # SQLite: реестр документов
│   └── qa/
│       ├── dev.json           # 20 вопросов (калибровка порога)
│       └── test.json          # 40+ вопросов (финальная оценка)
├── src/
│   ├── ingest/
│   │   ├── parser.py          # PDF/DOCX/MD → текст + метаданные
│   │   ├── chunker.py         # структурный чанкинг 500/50
│   │   └── indexer.py         # эмбеддинг + запись в Chroma/SQLite
│   ├── retrieve/
│   │   ├── search.py          # top-k exact cosine
│   │   └── gate.py            # порог отказа (tau, delta)
│   ├── generate/
│   │   └── answer.py          # промпт, генерация, парсинг ссылок
│   ├── llm/
│   │   └── client.py          # Ollama / OpenAI-совместимый клиент
│   ├── eval/
│   │   ├── metrics.py         # MRR, Hit@k, Faithfulness, refusal, latency
│   │   └── judge.py           # LLM-as-judge
│   ├── ui/
│   │   ├── chat.py
│   │   └── admin.py
│   └── config.py              # загрузка config.yaml + .env
├── tests/
│   ├── test_chunker.py        # границы чанков, перекрытие, заголовки
│   ├── test_gate.py           # порог/отказ на синтетических score
│   ├── test_parser.py         # PDF/DOCX/MD fixtures
│   └── test_retrieval_smoke.py# индекс из 3 чанков → точный ответ в top-1
├── reports/                   # metrics.json, графики (коммитятся)
└── docs/                      # настоящая документация
```

## 2. Быстрый старт

```bash
# 1. Python-окружение
python -m venv .venv && .venv\Scripts\activate     # Windows
pip install -r requirements.txt

# 2. LLM (Ollama)
ollama pull qwen2.5:7b

# 3. Конфиг
cp .env.example .env                               # при необходимости правьте endpoint

# 4. Индексация демо-корпуса
python -m src.ingest --path data/raw

# 5. Приложение
streamlit run app.py

# 6. Оценка (по завершении пайплайна)
python -m src.evaluate --qa data/qa/test.json --out reports/metrics.json
```

Тесты и линт: `pytest -q` · `ruff check src tests`.

## 3. Docker Compose

```yaml
# docker-compose.yml (фрагмент)
services:
  app:
    build: .
    ports: ["8501:8501"]
    volumes:
      - ./data:/app/data
      - ./config.yaml:/app/config.yaml
    environment:
      - OLLAMA_BASE_URL=http://ollama:11434
  ollama:
    image: ollama/ollama:latest
    volumes:
      - ollama:/root/.ollama
    # для GPU: deploy.resources.reservations.devices (nvidia)
volumes:
  ollama:
```

ChromaDB **embedded** — отдельного сервиса не требуется. Без Docker всё то же самое
работает локально (Ollama на хосте) — это основной режим разработки.

## 4. Конфигурация и секреты

| Источник | Что хранит |
|---|---|
| `config.yaml` | модели, `top_k`, `threshold`, размеры чанков, temperature |
| `.env` | `OLLAMA_BASE_URL`, `OPENAI_API_KEY` (опц.), `RAG_EMBED_DEVICE`, `ADMIN_PASSWORD` (опц.) |
| Админка (session) | переопределяет `top_k`/`threshold`; «Сохранить в конфиг» → запись в yaml |

Секреты и `data/` — в `.gitignore`. В репозиторий коммитятся: код, конфиг, QA-наборы,
`reports/`, `docs/`.

## 5. Воспроизводимость (чек-лист)

- [x] `requirements.txt` с пиннингом версий
- [x] зафиксированные теги моделей (`intfloat/multilingual-e5-base`, `qwen2.5:7b`)
- [x] `seed=42`, `temperature ≤ 0.2`
- [x] QA-набор и параметры `tau` в репозитории
- [x] каждый запуск eval пишет полный набор параметров в `reports/metrics.json`
- [ ] README со скриншотами, инструкцией запуска и таблицей метрик

## 6. CI (GitHub Actions, минимальный)

На каждый push/PR: `ruff check` → `pytest` → smoke-индекс тестового корпуса
(3 чанка, light-эмбеддер) и проверка, что retrieval возвращает top-1.
Полный eval с LLM — вручную перед чекпоинтами (CI без GPU).

## 7. Требования к окружению

| Режим | Минимум |
|---|---|
| Разработка (CPU, light) | 8 GB RAM, `qwen2.5:3b` + MiniLM |
| Полный локальный | 16 GB RAM / 8 GB VRAM, `qwen2.5:7b` + e5-base |
| Docker | Docker 24+, Compose v2; GPU — NVIDIA Container Toolkit |
