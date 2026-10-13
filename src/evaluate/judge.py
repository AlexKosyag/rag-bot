"""LLM-as-judge: Faithfulness ответа контексту (0/1/2)."""
from __future__ import annotations

import re

from src.llm.client import OllamaClient

JUDGE_PROMPT = """Оцени, насколько ответ подтверждён контекстом из документов.

Вопрос: {question}

Контекст:
{context}

Ответ ассистента: {answer}

Критерии:
2 — ответ полностью следует из контекста, факты подтверждены, ничего не выдумано;
1 — ответ частично подтверждён, есть несущественные добавления;
0 — ответ противоречит контексту или выдуман.

Ответь одним числом: 0, 1 или 2."""


def _parse_score(text: str) -> int:
    m = re.search(r"\b([012])\b", text.strip())
    return int(m.group(1)) if m else 0


def faithfulness_score(question: str, context: str, answer: str,
                       llm: OllamaClient) -> int:
    """0/1/2; при ошибке LLM — 0 (консервативно)."""
    prompt = JUDGE_PROMPT.format(question=question, context=context[:6000],
                                 answer=answer[:2000])
    try:
        raw = llm.complete([{"role": "user", "content": prompt}])
    except Exception:
        return 0
    return _parse_score(raw)
