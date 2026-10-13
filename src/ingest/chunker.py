"""Структурный чанкинг: секции по заголовкам -> блоки ~size_tokens с перекрытием."""
from __future__ import annotations

import re
from dataclasses import dataclass

from src.embedder import CHARS_PER_TOKEN, estimate_tokens

HEADING_MD = re.compile(r"^(#{1,6})\s+(.+)$")
# Нумерованный заголовок раздела: "1. Общие положения", "2.3 Порядок".
# Списки-предложения (заканчиваются на '.') — не заголовки.
HEADING_NUM = re.compile(r"^(\d+(?:\.\d+)*)[.)]?\s+\S")


@dataclass
class Unit:
    """Абзац (или набор абзацев) с текущим заголовком раздела."""

    heading: str | None
    text: str
    start: int  # смещение в исходном тексте документа


@dataclass
class Chunk:
    index: int
    text: str
    heading: str
    start: int  # смещение основного содержимого (для номера страницы)


def _is_heading(line: str) -> str | None:
    """Возвращает текст заголовка или None."""
    if m := HEADING_MD.match(line):
        return m.group(2).strip()
    if len(line) < 80 and not line.endswith((".", ":", ";", "—", "-")):
        if HEADING_NUM.match(line):
            return line.strip()
    return None


def _scan(text: str) -> list[Unit]:
    """Разбирает текст на абзацы, отслеживая ближайший заголовок."""
    units: list[Unit] = []
    heading: str | None = None
    offset = 0
    para_start: int | None = None
    para_lines: list[str] = []

    def flush() -> None:
        nonlocal para_start, para_lines
        if para_lines:
            body = "\n".join(para_lines).strip()
            if body:
                units.append(Unit(heading, body, para_start if para_start is not None else 0))
        para_start = None
        para_lines = []

    for line in text.split("\n"):
        stripped = line.strip()
        if h := _is_heading(stripped):
            flush()
            heading = h
        elif not stripped:
            flush()
        else:
            if not para_lines:
                para_start = offset
            para_lines.append(line)
        offset += len(line) + 1
    flush()
    return units


def _overlap_tail(body: str, overlap_tokens: int) -> str:
    """Хвост тела чанка длиной ~overlap_tokens (со сдвигом до границы слова)."""
    if overlap_tokens <= 0 or not body:
        return ""
    overlap_chars = int(overlap_tokens * CHARS_PER_TOKEN)
    tail = body[-overlap_chars:]
    if (space := tail.find(" ")) != -1 and space < 60:
        tail = tail[space + 1:]
    return tail.strip()


def chunk_document(
    doc_text: str,
    size_tokens: int = 500,
    overlap_tokens: int = 50,
    min_tokens: int = 50,
    token_counter=estimate_tokens,
    page_starts: list[int] | None = None,
) -> list[Chunk]:
    """Пакует абзацы в чанки до size_tokens; длинные секции режет с перекрытием.

    min_tokens зарезервирован под эксперименты (см. docs/pipeline.md).
    """
    del min_tokens  # используется в экспериментах, не в базовом алгоритме
    units = _scan(doc_text)
    chunks: list[Chunk] = []
    tail = ""  # перекрытие, префиксующее следующий чанк

    def emit(body: str, start: int, heading: str) -> None:
        nonlocal tail
        body = body.strip()
        if not body:
            return
        text = f"{tail}\n\n{body}" if tail else body
        chunks.append(Chunk(index=len(chunks), text=text, heading=heading, start=start))
        tail = _overlap_tail(body, overlap_tokens)

    buf: list[Unit] = []
    buf_tokens = 0

    def flush() -> None:
        nonlocal buf, buf_tokens
        if buf:
            emit(
                "\n\n".join(u.text for u in buf),
                start=buf[0].start,
                heading=buf[0].heading or "—",
            )
        buf, buf_tokens = [], 0

    for unit in units:
        unit_tokens = token_counter(unit.text)
        if unit_tokens > size_tokens:
            # Длинная секция: режем по абзацам с перекрытием.
            flush()
            piece: list[str] = []
            piece_tokens = 0
            piece_start = unit.start

            def emit_piece() -> None:
                nonlocal piece, piece_tokens, piece_start
                if piece:
                    emit("\n\n".join(piece), start=piece_start,
                         heading=unit.heading or "—")
                    piece, piece_tokens = [], 0
                    piece_start = unit.start  # сдвиг уточняется ниже

            for para in unit.text.split("\n\n"):
                if not para.strip():
                    continue
                p_tokens = token_counter(para)
                if piece and piece_tokens + p_tokens > size_tokens:
                    emit_piece()
                if not piece:
                    # позиция абзаца внутри секции
                    rel = unit.text.find(para[:40])
                    piece_start = unit.start + (rel if rel != -1 else 0)
                piece.append(para)
                piece_tokens += p_tokens
            emit_piece()
            continue

        if buf and buf_tokens + unit_tokens > size_tokens:
            flush()
        buf.append(unit)
        buf_tokens += unit_tokens

    flush()
    return [c for c in chunks if c.text]
