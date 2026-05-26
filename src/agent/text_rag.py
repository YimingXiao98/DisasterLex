"""
Text-RAG retrieval over the TDIS source corpus.

Baseline for the ``text-rag`` ablation: instead of the EKG (structured
causal edges) or the DDCG (structured schema metadata), the agent gets a
commodity text retriever over the raw document chunks the EKG was
extracted from. Paired with ``query_database`` this tests whether the
graph-structured knowledge beats a chunk retriever over the same sources.

Loads the index built by ``scripts/build_text_rag_index.py`` lazily on
first call; returns top-K chunks formatted for LLM consumption.
"""
from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any, List

import numpy as np

logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
INDEX_DIR = PROJECT_ROOT / "configs" / "text_rag"

_STATE: dict[str, Any] = {"loaded": False}


def _ensure_loaded() -> None:
    if _STATE["loaded"]:
        return
    emb_path = INDEX_DIR / "embeddings.npz"
    meta_path = INDEX_DIR / "chunks.jsonl"
    info_path = INDEX_DIR / "info.json"
    if not (emb_path.exists() and meta_path.exists()):
        raise RuntimeError(
            f"Text-RAG index missing at {INDEX_DIR}. Run "
            f"`python scripts/build_text_rag_index.py` first."
        )
    info = json.loads(info_path.read_text())
    embeddings = np.load(emb_path)["embeddings"]
    chunks = []
    with open(meta_path) as f:
        for line in f:
            line = line.strip()
            if line:
                chunks.append(json.loads(line))
    assert len(chunks) == embeddings.shape[0], \
        f"chunk count mismatch: {len(chunks)} meta vs {embeddings.shape[0]} emb"

    from sentence_transformers import SentenceTransformer
    model = SentenceTransformer(info["model"])

    _STATE.update(
        loaded=True,
        info=info,
        embeddings=embeddings,   # already L2-normalised
        chunks=chunks,
        model=model,
    )
    logger.info(f"Text-RAG index loaded: {len(chunks)} chunks from {len(info['sources'])} corpora.")


def retrieve(question: str, k: int = 5) -> List[dict]:
    """Return top-``k`` chunks ranked by cosine similarity to ``question``."""
    _ensure_loaded()
    q_emb = _STATE["model"].encode(
        [question],
        convert_to_numpy=True,
        normalize_embeddings=True,
    ).astype(np.float32)
    sims = _STATE["embeddings"] @ q_emb[0]   # dot product == cosine (both normalised)
    top_idx = np.argsort(-sims)[:k]
    return [
        {**_STATE["chunks"][int(i)], "score": float(sims[int(i)])}
        for i in top_idx
    ]


def retrieve_formatted(question: str, k: int = 5, max_chars: int = 1200) -> str:
    """Return top-``k`` chunks as a single prompt-ready text block.

    Each chunk is truncated to ``max_chars`` and prefixed with its source.
    """
    hits = retrieve(question, k=k)
    lines: list[str] = [f"Top-{len(hits)} document excerpts retrieved for the question:"]
    for i, h in enumerate(hits, 1):
        src = h.get("source_file") or h.get("chunk_id") or "unknown"
        text = h["text"]
        if len(text) > max_chars:
            text = text[:max_chars].rstrip() + " [...]"
        lines.append(f"\n[{i}] Source: {src}  (sim={h['score']:.3f})\n{text}")
    return "\n".join(lines)
