"""LightRAG-backed retrieval for the ``lightrag`` ablation.

Mirrors ``src/agent/text_rag.py`` but instead of returning flat top-k chunks,
queries a pre-built LightRAG knowledge-graph index that was extracted from the
same TDIS source corpus. Used by the ``query_lightrag`` ReAct tool and by the
``--ablation lightrag`` benchmark mode to test whether *automatically extracted*
graph structure is enough, or whether the curated EKG's typed concepts and
\\textsc{Maps\\_To} bridges are load-bearing.

The TDIS chunks (n=2,577) feed both Text-RAG and LightRAG so the only thing
that differs between those two baselines is the retrieval substrate (flat
similarity vs. entity-relation-graph traversal).
"""
from __future__ import annotations

import asyncio
import logging
import os
import sys
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
INDEX_DIR = PROJECT_ROOT / "data/lightrag_index"
# Optional: if a vendored copy of LightRAG exists at external/lightrag, prefer
# it; otherwise fall back to the pip-installed `lightrag-hku` package.
LIGHTRAG_REPO = PROJECT_ROOT / "external/lightrag"

# Cache the initialised RAG instance across calls. LightRAG init is expensive
# (loads graphs, embeddings, KV stores) and there's no benefit to repeating it
# per question.
#
# _RAG_TLS / _LOOP_TLS hold thread-local references when --parallel > 1; when
# the benchmark harness spawns N workers each thread has its own LightRAG
# handle and event loop, avoiding "event loop is already running" errors.
_RAG_TLS = None
_LOOP_TLS = None


def _get_loop() -> asyncio.AbstractEventLoop:
    """Reuse a single event loop per *thread*. ReAct calls into this
    synchronously, so we can't rely on an outer loop being present.

    ThreadPoolExecutor (used when --parallel > 1) gives each worker its own
    thread, but module-level singletons would alias across threads and trigger
    "event loop is already running" because asyncio.run / loop.run_until_complete
    forbids re-entering an already-running loop. Storing the loop in a
    threading.local() keeps each thread's loop independent.
    """
    global _LOOP_TLS
    if _LOOP_TLS is None:
        import threading
        _LOOP_TLS = threading.local()
    loop = getattr(_LOOP_TLS, "loop", None)
    if loop is None or loop.is_closed():
        loop = asyncio.new_event_loop()
        _LOOP_TLS.loop = loop
    return loop


def _ensure_lightrag_on_path() -> None:
    # Prefer a vendored copy under external/lightrag if present; otherwise rely
    # on the pip-installed package (`pip install lightrag-hku`).
    if LIGHTRAG_REPO.exists() and str(LIGHTRAG_REPO) not in sys.path:
        sys.path.insert(0, str(LIGHTRAG_REPO))


async def _build_rag() -> Any:
    _ensure_lightrag_on_path()
    from lightrag import LightRAG  # type: ignore
    from lightrag.utils import EmbeddingFunc  # type: ignore
    from lightrag.llm.openai import openai_complete_if_cache  # type: ignore
    from sentence_transformers import SentenceTransformer

    api_key = os.getenv("OPENROUTER_API_KEY_HELDOUT") or os.getenv("OPENROUTER_API_KEY")
    if not api_key:
        raise RuntimeError(
            "Set OPENROUTER_API_KEY_HELDOUT (or OPENROUTER_API_KEY) before "
            "calling LightRAG retrieval — query-time generation needs an LLM."
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

    embed_model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")

    async def _embed(texts):
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(
            None,
            lambda: embed_model.encode(
                texts, normalize_embeddings=True, show_progress_bar=False
            ),
        )

    embedding_func = EmbeddingFunc(
        embedding_dim=384,
        max_token_size=8192,
        func=_embed,
    )

    rag = LightRAG(
        working_dir=str(INDEX_DIR),
        llm_model_func=llm_func,
        embedding_func=embedding_func,
    )
    await rag.initialize_storages()
    return rag


def _ensure_rag() -> Any:
    global _RAG_TLS
    if _RAG_TLS is None:
        import threading
        _RAG_TLS = threading.local()
    rag = getattr(_RAG_TLS, "rag", None)
    if rag is not None:
        return rag
    if not INDEX_DIR.exists():
        raise RuntimeError(
            f"LightRAG index missing at {INDEX_DIR}. Run "
            f"`python scripts/baselines/index_lightrag.py` first."
        )
    loop = _get_loop()
    rag = loop.run_until_complete(_build_rag())
    _RAG_TLS.rag = rag
    logger.info("LightRAG index loaded into thread %s", __import__("threading").get_ident())
    return rag


_KG_SECTION_END_MARKERS = (
    "Document Chunks",
    "Reference Document List",
)
_MAX_CONTEXT_CHARS = 4000


def _trim_to_graph_sections(context: str, max_chars: int = _MAX_CONTEXT_CHARS) -> str:
    """Strip the verbose Document-Chunks / Reference-Document-List sections
    from LightRAG's raw output. Those sections dump ~100KB of source-text
    spans which (a) overwhelm the downstream agent's prompt, and (b) are
    redundant with the Knowledge-Graph entity/relation summaries above them.
    Hard-cap the survivor at ``max_chars`` so a single LightRAG call cannot
    blow up the agent's context budget.
    """
    end = len(context)
    for marker in _KG_SECTION_END_MARKERS:
        idx = context.find(marker)
        if idx > 0 and idx < end:
            end = idx
    trimmed = context[:end].rstrip()
    if len(trimmed) <= max_chars:
        return trimmed
    return trimmed[:max_chars].rstrip() + "\n[...LightRAG context truncated to keep agent prompt within budget...]"


def retrieve_formatted(question: str, mode: str = "local") -> str:
    """Retrieve LightRAG context for ``question`` and return a prompt-ready
    text block. ``mode`` is one of ``naive``, ``local``, ``global``, ``hybrid``,
    ``mix`` (see LightRAG docs). Defaults to ``local`` (entity-focused) for
    speed; the returned block is trimmed to the entity/relation graph
    sections only (the Document-Chunks section bloats the agent's prompt
    and is redundant with the curated EKG/text-rag baselines).
    """
    _ensure_lightrag_on_path()
    from lightrag import QueryParam  # type: ignore

    rag = _ensure_rag()
    loop = _get_loop()
    param = QueryParam(mode=mode, only_need_context=True, top_k=5)
    context = loop.run_until_complete(rag.aquery(question, param=param))
    if not context:
        return "LightRAG returned no context for this query."
    context = _trim_to_graph_sections(context)
    header = (
        f"Top entity+relation context retrieved from LightRAG (mode={mode}) "
        f"over the same source corpus that feeds the Expert Knowledge Graph:"
    )
    return f"{header}\n\n{context}"


def shutdown() -> None:
    """Tear down the per-thread LightRAG instance and event loop cleanly.

    LightRAG has long-lived background tasks (Postgres pool, async queues) that
    raise ``Event loop is closed`` warnings if the loop ends mid-flight. Call
    this in any short-lived script or test before exiting; the benchmark
    harness invokes it via atexit (registered below). Only shuts down the
    current thread's instance — other workers handle their own teardown.
    """
    if _RAG_TLS is None or _LOOP_TLS is None:
        return
    rag = getattr(_RAG_TLS, "rag", None)
    loop = getattr(_LOOP_TLS, "loop", None)
    if rag is not None and loop is not None and not loop.is_closed():
        try:
            loop.run_until_complete(rag.finalize_storages())
        except Exception as e:  # noqa: BLE001
            logger.warning("LightRAG finalize_storages failed: %s", e)
    if loop is not None and not loop.is_closed():
        loop.close()
    if hasattr(_RAG_TLS, "rag"):
        _RAG_TLS.rag = None
    if hasattr(_LOOP_TLS, "loop"):
        _LOOP_TLS.loop = None


import atexit
atexit.register(shutdown)
