"""Compute sentence-transformer embeddings for the Text-RAG ablation.

Reads the chunked source corpus shipped at ``configs/text_rag/chunks.jsonl``
(n = 2,577 chunks from the TDIS source documents that backed the EKG
construction), embeds each chunk with sentence-transformers/all-MiniLM-L6-v2,
and writes ``configs/text_rag/embeddings.npz``.

Used as the retrieval substrate for the ``text-rag`` ablation. The chunk file
itself is checked into the repo so reviewers do not need access to the source
PDFs; this script just produces the matching embedding matrix.

Usage:
    python scripts/build_text_rag_index.py
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from sentence_transformers import SentenceTransformer

PROJECT_ROOT = Path(__file__).resolve().parent.parent
CHUNKS_PATH = PROJECT_ROOT / "configs" / "text_rag" / "chunks.jsonl"
OUT_DIR = PROJECT_ROOT / "configs" / "text_rag"
OUT_DIR.mkdir(parents=True, exist_ok=True)

MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"


def load_chunks() -> list[dict]:
    out: list[dict] = []
    with open(CHUNKS_PATH) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                out.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    return out


def main() -> None:
    print(f"Loading chunks from {CHUNKS_PATH} ...")
    chunks = load_chunks()
    print(f"  loaded {len(chunks)} chunks")

    print(f"Loading embedding model: {MODEL_NAME}")
    model = SentenceTransformer(MODEL_NAME)

    print("Encoding (about one minute for ~2,500 chunks) ...")
    texts = [c["text"] for c in chunks]
    embeddings = model.encode(
        texts,
        batch_size=32,
        show_progress_bar=True,
        convert_to_numpy=True,
        normalize_embeddings=True,   # so cosine == dot product
    ).astype(np.float32)
    print(f"  embeddings shape: {embeddings.shape}")

    np.savez_compressed(OUT_DIR / "embeddings.npz", embeddings=embeddings)
    info = {
        "model": MODEL_NAME,
        "count": len(chunks),
        "dim": int(embeddings.shape[1]),
        "sources": ["configs/text_rag/chunks.jsonl"],
    }
    (OUT_DIR / "info.json").write_text(json.dumps(info, indent=2))
    print(f"\nSaved {OUT_DIR}/embeddings.npz and updated info.json")


if __name__ == "__main__":
    main()
