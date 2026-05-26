"""Build a HippoRAG 2 index over the TDIS source corpus.

HippoRAG 2 (Gutiérrez et al., 2025) does open-relation extraction over passages,
constructs a memory-style KG, and retrieves via personalised PageRank. We use
the OSU NLP Group's open-source implementation.

Pre-reqs:
  - hipporag installed (pip install git+https://github.com/OSU-NLP-Group/HippoRAG@main)
  - OPENROUTER_API_KEY in env (LLM-driven entity/relation extraction)
  - OPENAI_API_KEY in env (embeddings via text-embedding-3-small)

Usage:
    python scripts/baselines/index_hipporag.py
"""
from __future__ import annotations

import argparse
import json
import logging
import os
import shutil
import sys
from pathlib import Path

# Stub vllm + outlines BEFORE importing hipporag. HippoRAG's __init__.py
# eagerly imports offline-transformers and offline-vllm paths that we don't
# use (we route LLM through OpenRouter and embeddings through OpenAI).
class _Stub:
    def __init__(self, *a, **k): pass
    def __getattr__(self, n): return _Stub()
    def __call__(self, *a, **k): return _Stub()

for _n in ("vllm", "vllm.engine", "vllm.engine.async_llm_engine",
          "outlines", "outlines.generate", "outlines.models",
          "outlines.types", "outlines.fsm"):
    if _n not in sys.modules:
        _m = type(sys)(_n)
        _m.LLM = _m.SamplingParams = _m.AsyncEngineArgs = _m.AsyncLLMEngine = _Stub
        _m.generate = _m.models = _m.fsm = _m.regex = _m.json = _m.choice = _Stub
        _m.transformers = _m.openai = _Stub
        sys.modules[_n] = _m

# Monkey-patch openai.OpenAI to route api_key by base_url. HippoRAG constructs
# two clients (LLM via OpenRouter; embeddings via OpenAI direct) but reads only
# OPENAI_API_KEY from env. We dispatch by URL.
import openai as _openai_mod  # noqa: E402

_real_openai_init = _openai_mod.OpenAI.__init__

def _patched_openai_init(self, *args, **kw):  # type: ignore
    base_url = kw.get("base_url") or (args[1] if len(args) > 1 else None)
    if kw.get("api_key") is None and base_url and "openrouter" in str(base_url).lower():
        kw["api_key"] = os.environ.get("OPENROUTER_API_KEY_HELDOUT") or os.environ.get("OPENROUTER_API_KEY")
    return _real_openai_init(self, *args, **kw)

_openai_mod.OpenAI.__init__ = _patched_openai_init

# Cap HippoRAG's OpenIE ThreadPoolExecutor concurrency. The default
# ThreadPoolExecutor() with no max_workers spawns 32+ threads, which floods
# OpenRouter's rate limit. Patch concurrent.futures.ThreadPoolExecutor used
# by openie_openai to a small max_workers.
import concurrent.futures as _cfutures  # noqa: E402

_real_tpe = _cfutures.ThreadPoolExecutor

class _CappedTPE(_real_tpe):
    def __init__(self, max_workers=None, *a, **kw):
        if max_workers is None or max_workers > 4:
            max_workers = 4
        super().__init__(max_workers=max_workers, *a, **kw)

_cfutures.ThreadPoolExecutor = _CappedTPE

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
DEFAULT_CHUNKS = PROJECT_ROOT / "configs/text_rag/chunks.jsonl"
DEFAULT_SAVE = PROJECT_ROOT / "data/hipporag_index"

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger("hipporag-index")


def load_chunks(chunks_path: Path) -> list[dict]:
    chunks = []
    with open(chunks_path) as f:
        for line in f:
            line = line.strip()
            if line:
                chunks.append(json.loads(line))
    return chunks


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--chunks", type=Path, default=DEFAULT_CHUNKS)
    parser.add_argument("--save-dir", type=Path, default=DEFAULT_SAVE)
    parser.add_argument("--llm-model", type=str,
                        default="google/gemini-3.1-flash-lite-preview")
    parser.add_argument("--llm-base-url", type=str,
                        default="https://openrouter.ai/api/v1")
    parser.add_argument("--embedding-model", type=str, default="text-embedding-3-small")
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    if args.force and args.save_dir.exists():
        log.info("Removing existing HippoRAG index at %s", args.save_dir)
        shutil.rmtree(args.save_dir)
    args.save_dir.mkdir(parents=True, exist_ok=True)

    # Lazy import so the file at least parses without hipporag installed.
    from hipporag import HippoRAG  # type: ignore

    # HippoRAG uses OPENAI_API_KEY for both LLM (we pass llm_base_url to point at
    # OpenRouter) and embeddings (which MUST go to api.openai.com, not OpenRouter).
    # If we set OPENAI_BASE_URL=openrouter, embeddings break. So we explicitly
    # clear it and pass per-call base URLs via HippoRAG kwargs.
    os.environ.pop("OPENAI_BASE_URL", None)
    if not os.environ.get("OPENAI_API_KEY"):
        raise RuntimeError("OPENAI_API_KEY required (real OpenAI key, not OpenRouter)")

    chunks = load_chunks(args.chunks)
    log.info("Loaded %d TDIS chunks", len(chunks))
    docs = [c["text"] for c in chunks]

    hipporag = HippoRAG(
        save_dir=str(args.save_dir),
        llm_model_name=args.llm_model,
        llm_base_url=args.llm_base_url,
        embedding_model_name=args.embedding_model,
    )
    log.info("Indexing %d docs via HippoRAG...", len(docs))
    hipporag.index(docs=docs)
    log.info("Done. Index at %s", args.save_dir)


if __name__ == "__main__":
    main()
