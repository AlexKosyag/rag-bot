"""Загрузка конфигурации: config.yaml + переменные окружения (.env)."""
from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path

import yaml
from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = ROOT / "config.yaml"


@dataclass
class ModelsCfg:
    embedding: str = "intfloat/multilingual-e5-base"
    llm: str = "qwen2.5:7b"
    device: str = "cpu"


@dataclass
class RetrievalCfg:
    top_k: int = 5
    threshold: float = 0.55
    delta: float = 0.05


@dataclass
class ChunkingCfg:
    size_tokens: int = 500
    overlap_tokens: int = 50
    min_tokens: int = 50


@dataclass
class GenerationCfg:
    temperature: float = 0.2
    max_tokens: int = 700
    timeout_s: int = 90
    base_url: str = "http://localhost:11434"


@dataclass
class PathsCfg:
    raw_dir: Path = ROOT / "data" / "raw"
    chroma_dir: Path = ROOT / "data" / "chroma"
    registry_db: Path = ROOT / "data" / "registry.db"
    reports_dir: Path = ROOT / "reports"
    qa_dir: Path = ROOT / "data" / "qa"


@dataclass
class Config:
    models: ModelsCfg = field(default_factory=ModelsCfg)
    retrieval: RetrievalCfg = field(default_factory=RetrievalCfg)
    chunking: ChunkingCfg = field(default_factory=ChunkingCfg)
    generation: GenerationCfg = field(default_factory=GenerationCfg)
    paths: PathsCfg = field(default_factory=PathsCfg)


def _resolve(value: str) -> Path:
    p = Path(value)
    return p if p.is_absolute() else ROOT / p


def load_config(path: Path | str = CONFIG_PATH) -> Config:
    """Читает config.yaml, применяет переопределения из окружения (.env)."""
    load_dotenv(ROOT / ".env")
    raw: dict = {}
    cfg_path = Path(path)
    if cfg_path.exists():
        raw = yaml.safe_load(cfg_path.read_text(encoding="utf-8")) or {}

    models = ModelsCfg(**raw.get("models", {}))
    retrieval = RetrievalCfg(**raw.get("retrieval", {}))
    chunking = ChunkingCfg(**raw.get("chunking", {}))
    generation = GenerationCfg(**raw.get("generation", {}))
    paths_raw = raw.get("paths", {})
    paths = PathsCfg(
        raw_dir=_resolve(paths_raw.get("raw_dir", "data/raw")),
        chroma_dir=_resolve(paths_raw.get("chroma_dir", "data/chroma")),
        registry_db=_resolve(paths_raw.get("registry_db", "data/registry.db")),
        reports_dir=_resolve(paths_raw.get("reports_dir", "reports")),
        qa_dir=_resolve(paths_raw.get("qa_dir", "data/qa")),
    )

    if os.getenv("OLLAMA_BASE_URL"):
        generation.base_url = os.environ["OLLAMA_BASE_URL"]
    if os.getenv("RAG_EMBED_DEVICE"):
        models.device = os.environ["RAG_EMBED_DEVICE"]

    return Config(models=models, retrieval=retrieval, chunking=chunking,
                  generation=generation, paths=paths)


def save_retrieval(cfg: Config, path: Path | str = CONFIG_PATH) -> None:
    """Сохраняет текущие top_k / threshold / delta в config.yaml (кнопка в админке)."""
    cfg_path = Path(path)
    raw: dict = {}
    if cfg_path.exists():
        raw = yaml.safe_load(cfg_path.read_text(encoding="utf-8")) or {}
    raw.setdefault("retrieval", {})
    raw["retrieval"].update(
        top_k=cfg.retrieval.top_k,
        threshold=cfg.retrieval.threshold,
        delta=cfg.retrieval.delta,
    )
    cfg_path.write_text(
        yaml.safe_dump(raw, allow_unicode=True, sort_keys=False),
        encoding="utf-8",
    )
