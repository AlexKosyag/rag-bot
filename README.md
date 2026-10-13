# RAG-чат-бот по внутренним документам

Чат-бот на базе RAG: отвечает на вопросы сотрудников **только** по загруженной базе
документов (регламенты, инструкции, FAQ), со ссылками на источники и корректным отказом
«Нет в документах», если ответа в базе нет.

Полная документация — в [`docs/`](docs/README.md):
[обзор](docs/overview.md) · [архитектура](docs/architecture.md) ·
[стек и модели](docs/stack.md) · [конвейер](docs/pipeline.md) ·
[оценка качества](docs/evaluation.md) · [интерфейс](docs/interface.md) ·
[запуск](docs/launch.md)

## Стек

| Слой | Решение |
|---|---|
| Эмбеддинги | `intfloat/multilingual-e5-base` (768-dim, окно 512 токенов) |
| LLM | `Qwen2.5-7B-Instruct` через Ollama (fallback: `qwen2.5:3b`, API) |
| Векторное хранилище | ChromaDB (embedded, cosine) + SQLite-реестр |
| Интерфейс | Streamlit: чат с источниками + админка |
| Оценка | Hit@k / MRR / Context Recall + refusal-метрики + LLM-as-judge |

## Быстрый старт

```bash
# 1. Зависимости
pip install -r requirements.txt

# 2. LLM (Ollama: https://ollama.com)
ollama pull qwen2.5:7b

# 3. Конфиг (при необходимости)
cp .env.example .env

# 4. Индексация демо-корпуса (data/raw/*.md)
python -m src.ingest

# 5. Приложение
streamlit run app.py
```

## Оценка качества

```bash
# retrieval + метрики отказов (без LLM, быстро)
python -m src.evaluate --qa data/qa/test.json

# + генерация ответов, latency и Faithfulness
python -m src.evaluate --qa data/qa/test.json --llm --judge
```

Отчёт: `reports/metrics.json` (его же показывает вкладка «Метрики» в админке).

## Тесты и линт

```bash
pytest -q        # 29 тестов: чанкер, порог отказа, парсер, метрики, хранилище
ruff check src tests app.py
```

## Структура

```
app.py                  # Streamlit: чат + админка
config.yaml             # модели, top_k, порог, чанкинг
src/
  config.py             # конфиг (yaml + .env)
  embedder.py           # sentence-transformers, префиксы E5, нормализация
  store.py              # Chroma (чанки) + SQLite (реестр документов)
  engine.py             # ретрив -> порог -> промпт -> LLM
  ingest/               # парсер, структурный чанкер, индексатор (+ CLI)
  retrieve/             # top-k поиск, порог отказа
  generate/             # промпт, сборка контекста, ссылки
  llm/                  # клиент Ollama (стриминг)
  evaluate/             # метрики, judge, CLI
  ui/                   # чат и админка
data/
  raw/                  # демо-корпус (4 синтетических регламента)
  qa/                   # dev.json (10) + test.json (14) QA-покрытие
tests/                  # pytest
docs/                   # документация проекта
```

## Статус MVP

Работает end-to-end: парсинг → чанкинг → эмбеддинги → Chroma → порог отказа →
стриминговый ответ с источниками → админка → eval-скрипт.

Дemo-корпус — 4 синтетических документа и 24 QA (ТЗ требует 50+ документов и 60+
вопросов: масштабирование — положить файлы в `data/raw/` и дополнить `data/qa/*.json`).
Калибровка порога `tau`, эксперименты с размером чанка и baseline «Qwen без RAG» —
по плану из [docs/evaluation.md](docs/evaluation.md).
