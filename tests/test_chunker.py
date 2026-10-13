"""Тесты структурного чанкинга: заголовки, размер, перекрытие."""
from src.ingest.chunker import _scan, chunk_document

DOC = """# Устав

Общие сведения об организации.

## 1. Общие положения

Абзац один первого раздела.

Абзац два первого раздела.

## 2. Цели

Абзац раздела про цели.
"""


def test_scan_tracks_headings():
    units = _scan(DOC)
    headings = [u.heading for u in units]
    assert headings[0] == "Устав"
    assert "1. Общие положения" in headings
    assert "2. Цели" in headings
    # последний абзац относится ко второму разделу
    assert units[-1].heading == "2. Цели"


def test_chunk_carries_heading_and_index():
    chunks = chunk_document(DOC, size_tokens=500, overlap_tokens=0)
    assert chunks
    assert all(c.heading for c in chunks)
    assert [c.index for c in chunks] == list(range(len(chunks)))


def test_chunk_children_reconstruct_body():
    # child-абзацы (для поиска) собираются в тело parent-чанка (для контекста)
    chunks = chunk_document(DOC, size_tokens=500, overlap_tokens=0)
    assert all(c.children for c in chunks)
    for c in chunks:
        joined = "\n\n".join(u.text for u in c.children)
        assert c.text == joined
        # у child сохраняется заголовок его раздела (для контекстуализации)
        assert all(u.heading for u in chunks[0].children)


def test_chunks_respect_size_limit():
    text = "\n\n".join(f"Абзац номер {i} с достаточно длинным текстом." for i in range(40))
    # консервативный счётчик токенов + запас на склейку абзацев
    chunks = chunk_document(text, size_tokens=40, overlap_tokens=0)
    assert len(chunks) > 1
    for c in chunks:
        assert len(c.text) <= (40 + 100) * 2  # ~40 токенов (+ запас на абзац) * CHARS_PER_TOKEN


def test_overlap_between_chunks():
    text = "\n\n".join(f"Абзац номер {i} с достаточно длинным текстом." for i in range(40))
    chunks = chunk_document(text, size_tokens=40, overlap_tokens=10)
    assert len(chunks) >= 2
    # хвост предыдущего чанка присутствует в начале следующего
    tail = chunks[0].text[-30:]
    assert tail[:20] in chunks[1].text


def test_oversized_section_is_split():
    long_section = "## Большой раздел\n\n" + "\n\n".join(
        f"Абзац {i}: " + "подробный текст регламента. " * 5 for i in range(30)
    )
    chunks = chunk_document(long_section, size_tokens=60, overlap_tokens=5)
    assert len(chunks) > 1
    assert all(c.heading == "Большой раздел" for c in chunks)


def test_empty_text():
    assert chunk_document("") == []
    assert chunk_document("   \n\n  ") == []


def test_numbered_heading_detected():
    # PDF/DOCX без разметки: включён эвристический детектор заголовков
    text = "1. Общие положения\n\nТекст раздела о общих положениях."
    units = _scan(text, detect_numbered_headings=True)
    # строка-заголовок не становится unit, но «прилипает» к следующему абзацу
    assert len(units) == 1
    assert units[0].heading == "1. Общие положения"


def test_numbered_heading_disabled_by_default():
    # markdown: нумерованная строка — обычный текст (иначе теряются пункты)
    text = "1. Общие положения\n\nТекст раздела о общих положениях."
    units = _scan(text)
    assert len(units) == 2
    assert units[0].heading is None
    assert units[0].text == "1. Общие положения"


def test_list_item_is_not_heading():
    # пункт-предложение даже с включённым детектором — не заголовок
    text = "## Раздел\n\n1. Первый пункт списка предложений.\n"
    units = _scan(text, detect_numbered_headings=True)
    assert units[0].heading == "Раздел"


def test_long_numbered_sentence_is_not_heading():
    text = (
        "3.3. Каждое подразделение возглавляет руководитель, "
        "назначаемый Генеральным директором и подотчётный ему.\n"
    )
    units = _scan(text, detect_numbered_headings=True)
    assert units[0].heading is None
    assert units[0].text.startswith("3.3.")
