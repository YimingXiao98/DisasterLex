"""Build the LightRAG index over the TDIS source corpus.

Reads `configs/text_rag/chunks.jsonl` and ingests it into a fresh LightRAG
working directory (`data/lightrag_index/`). The index contains entity,
relation, and chunk vector stores. Wall-clock is roughly 20-40 minutes on a
laptop, dominated by LLM entity-extraction calls.

Reviewers running the LightRAG ablation must execute this once before
`scripts/run_benchmark.py --ablation lightrag`.

Usage:
    python scripts/build_lightrag_index.py [--working-dir data/lightrag_index]

Requires `OPENROUTER_API_KEY` (for entity extraction) and the
`sentence-transformers/all-MiniLM-L6-v2` embedding model (downloaded
automatically from HuggingFace on first run).
"""
from __future__ import annotations

import argparse
import asyncio
import json
import os
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
CHUNKS_PATH = PROJECT_ROOT / "configs" / "text_rag" / "chunks.jsonl"
DEFAULT_WORKING_DIR = PROJECT_ROOT / "data" / "lightrag_index"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--working-dir",
        type=Path,
        default=DEFAULT_WORKING_DIR,
        help="Output directory for the LightRAG index.",
    )
    parser.add_argument(
        "--llm-model",
        default="google/gemini-3.1-flash-lite-preview",
        help="LLM used for LightRAG entity extraction (via OpenRouter).",
    )
    parser.add_argument(
        "--embed-model",
        default="sentence-transformers/all-MiniLM-L6-v2",
        help="Embedding model for chunk and entity vectors.",
    )
    return parser.parse_args()


async def main_async(args: argparse.Namespace) -> None:
    try:
        from lightrag import LightRAG, QueryParam
    except ImportError:
        sys.stderr.write(
            "lightrag package not found. Install with: pip install lightrag-hku\n"
        )
        sys.exit(1)

    args.working_dir.mkdir(parents=True, exist_ok=True)

    chunks = []
    with open(CHUNKS_PATH) as f:
        for line in f:
            chunks.append(json.loads(line)["text"])
    print(f"Loaded {len(chunks)} chunks from {CHUNKS_PATH}.")

    rag = LightRAG(
        working_dir=str(args.working_dir),
        llm_model_name=args.llm_model,
        embedding_func_max_async=4,
    )
    print(f"Ingesting {len(chunks)} chunks into {args.working_dir} ...")
    await rag.ainsert(chunks)
    print("Done. Index files written to:", args.working_dir)


if __name__ == "__main__":
    asyncio.run(main_async(parse_args()))
