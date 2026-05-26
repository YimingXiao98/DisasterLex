"""Build a LightRAG index over the TDIS source corpus.

This is the Phase-2 counterpart to ``index`` step in CHESS. The TRIAGE
Text-RAG baseline (``src/agent/text_rag.py``) chunks the TDIS corpus into
flat chunks indexed by sentence-transformer embeddings; LightRAG runs over
the same corpus but additionally extracts entities and relationships into a
knowledge graph that supports multi-level (low + high) retrieval.

Same input → different retrieval substrate. That's the contract that makes
the ladder-of-structure comparison clean (Text-RAG → LightRAG → Full).

Usage:
    PYTHONPATH=. conda run -n disaster_graph_rag --no-capture-output python -u \\
        scripts/baselines/index_lightrag.py
"""
from __future__ import annotations

import argparse
import asyncio
import json
import logging
import os
from pathlib import Path
from typing import Iterable

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
DEFAULT_CHUNKS = PROJECT_ROOT / "config/text_rag/chunks.jsonl"
DEFAULT_WORKING_DIR = PROJECT_ROOT / "data/lightrag_index"

# Add LightRAG vendored repo to import path. Done at module load so the
# `from lightrag import ...` below resolves.
import sys
sys.path.insert(0, str(PROJECT_ROOT / "external/lightrag"))


logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger("lightrag-index")


def load_chunks(chunks_path: Path) -> list[dict]:
    """Load TDIS chunks from JSONL. Each entry has at minimum a ``text``
    field plus a ``source_file`` or ``chunk_id`` for citation."""
    chunks: list[dict] = []
    with open(chunks_path) as f:
        for line in f:
            line = line.strip()
            if line:
                chunks.append(json.loads(line))
    return chunks


def group_chunks_by_source(chunks: list[dict]) -> dict[str, list[str]]:
    """Group chunks by source file so LightRAG sees one document per source.

    The TDIS index has 2,577 chunks across two source corpora; LightRAG's
    indexer wants document-level text and produces its own chunking, so we
    re-concatenate first.
    """
    grouped: dict[str, list[str]] = {}
    for c in chunks:
        src = c.get("source_file") or c.get("chunk_id", "unknown")
        # Normalise the source to a filename stem.
        from pathlib import Path as _P
        stem = _P(str(src)).stem or "unknown"
        grouped.setdefault(stem, []).append(c["text"])
    return grouped


async def build_llm_func():
    """Return a LightRAG-compatible async LLM completion function backed by
    OpenRouter (OpenAI-compatible endpoint)."""
    from lightrag.llm.openai import openai_complete_if_cache

    api_key = os.getenv("OPENROUTER_API_KEY_HELDOUT") or os.getenv("OPENROUTER_API_KEY")
    if not api_key:
        raise RuntimeError(
            "OPENROUTER_API_KEY_HELDOUT or OPENROUTER_API_KEY must be set "
            "for the LightRAG indexer to call the LLM."
        )

    async def llm_func(prompt, system_prompt=None, history_messages=None, **kwargs):
        return await openai_complete_if_cache(
            "google/gemini-3.1-flash-lite-preview",
            prompt,
            system_prompt=system_prompt,
            history_messages=history_messages or [],
            api_key=api_key,
            base_url="https://openrouter.ai/api/v1",
            **kwargs,
        )

    return llm_func


async def build_embedding_func():
    """Return a LightRAG-compatible async embedding function backed by a
    local sentence-transformers model (matches our text_rag.py choice)."""
    from lightrag.utils import EmbeddingFunc
    from sentence_transformers import SentenceTransformer

    model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")

    async def _embed(texts: list[str]) -> np.ndarray:
        # SentenceTransformer.encode is synchronous; offload to a thread so
        # LightRAG's async pipeline doesn't block the event loop.
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(
            None,
            lambda: model.encode(
                texts, normalize_embeddings=True, show_progress_bar=False
            ),
        )

    return EmbeddingFunc(
        embedding_dim=384,
        max_token_size=8192,
        func=_embed,
    )


async def index_corpus(chunks_path: Path, working_dir: Path, force_rebuild: bool) -> None:
    from lightrag import LightRAG  # type: ignore

    if working_dir.exists() and force_rebuild:
        log.info("Removing existing LightRAG working dir at %s", working_dir)
        import shutil
        shutil.rmtree(working_dir)
    working_dir.mkdir(parents=True, exist_ok=True)

    log.info("Loading chunks from %s", chunks_path)
    chunks = load_chunks(chunks_path)
    log.info("Loaded %d chunks", len(chunks))

    grouped = group_chunks_by_source(chunks)
    log.info("Grouped into %d source documents", len(grouped))

    llm_func = await build_llm_func()
    embedding_func = await build_embedding_func()

    rag = LightRAG(
        working_dir=str(working_dir),
        llm_model_func=llm_func,
        embedding_func=embedding_func,
        chunk_token_size=1200,
        chunk_overlap_token_size=200,
    )
    await rag.initialize_storages()

    try:
        for stem, parts in grouped.items():
            full_text = "\n\n".join(parts)
            log.info("Inserting %s (%d chunks, %d chars)", stem, len(parts), len(full_text))
            await rag.ainsert(full_text, ids=[stem], file_paths=[f"{stem}.txt"])
    finally:
        await rag.finalize_storages()
    log.info("Done. Index lives at %s", working_dir)


def main(argv: Iterable[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--chunks", type=Path, default=DEFAULT_CHUNKS,
                        help="TDIS chunks JSONL")
    parser.add_argument("--working-dir", type=Path, default=DEFAULT_WORKING_DIR,
                        help="LightRAG index working directory")
    parser.add_argument("--force", action="store_true",
                        help="Wipe existing index and rebuild")
    args = parser.parse_args(argv)
    asyncio.run(index_corpus(args.chunks, args.working_dir, args.force))


if __name__ == "__main__":
    main()
