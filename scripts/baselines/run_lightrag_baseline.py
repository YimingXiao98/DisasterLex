"""Run LightRAG end-to-end on the TRIAGE 75-case heldout split.

This is the END-TO-END LightRAG baseline (not the in-pipeline ablation that
swaps the EKG retrieval call). For each benchmark case, we:

  1. Issue a LightRAG hybrid-mode query against the pre-built index, taking
     LightRAG's own answer as the system response (no DB access, no routing).
  2. Feed that answer through the same TRIAGE claim extractor + reasoning
     judge that scores every other ablation.

Mirrors scripts/baselines/run_hipporag_baseline.py: external systems
evaluated in their authors' native end-to-end mode, then scored on the
shared harness.

Pre-reqs:
  - LightRAG index built at data/lightrag_index/ (already exists)
  - OPENROUTER_API_KEY in env (LLM)
"""
from __future__ import annotations

import argparse
import asyncio
import json
import logging
import os
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "scripts"))
sys.path.insert(0, str(PROJECT_ROOT / "external" / "lightrag"))
import run_benchmark  # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger("lightrag-bench")

INDEX_DIR = Path(os.environ.get("LIGHTRAG_WORKING_DIR", PROJECT_ROOT / "data/lightrag_index"))


async def _build_rag(llm_model: str, llm_base_url: str, api_key: str):
    """Initialise a LightRAG instance for end-to-end query mode."""
    from lightrag import LightRAG  # type: ignore
    from lightrag.utils import EmbeddingFunc  # type: ignore
    from lightrag.llm.openai import openai_complete_if_cache  # type: ignore
    from sentence_transformers import SentenceTransformer

    async def llm_func(prompt, system_prompt=None, history_messages=None, **kwargs):
        return await openai_complete_if_cache(
            llm_model,
            prompt,
            system_prompt=system_prompt,
            history_messages=history_messages or [],
            api_key=api_key,
            base_url=llm_base_url,
            **kwargs,
        )

    embed_model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")

    async def _embed(texts):
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(
            None,
            lambda: embed_model.encode(texts, normalize_embeddings=True, show_progress_bar=False),
        )

    embedding_func = EmbeddingFunc(embedding_dim=384, max_token_size=8192, func=_embed)

    rag = LightRAG(
        working_dir=str(INDEX_DIR),
        llm_model_func=llm_func,
        embedding_func=embedding_func,
    )
    await rag.initialize_storages()
    return rag


async def _ask(rag, question: str, mode: str = "hybrid") -> str:
    from lightrag import QueryParam  # type: ignore
    try:
        result = await rag.aquery(question, param=QueryParam(mode=mode))
        return str(result) if result is not None else ""
    except Exception as e:  # noqa: BLE001
        log.warning("LightRAG aquery failed: %s", e)
        return ""


def render_answer(question: str, raw: str) -> str:
    return (
        f"The LightRAG baseline produced the following answer:\n\n"
        f"{raw.strip()}\n\n"
        f"STRUCTURED FACTS:\n"
    )


def synthesise_routing_state(case: dict) -> dict:
    return {"criticality": 0, "cluster": "", "query_type": "", "area_of_interest": "",
            "hazard_type": "", "data_availability_warnings": []}


def score_one(case, system_answer, routing_state, extractor_llm, reasoning_llm):
    deterministic_checks = [c for c in case.get("checks", []) if c.get("evaluator") == "deterministic"]
    reasoning_checks = [c for c in case.get("checks", []) if c.get("evaluator") == "reasoning_llm"]
    extracted_claims = run_benchmark._extract_case_claims(extractor_llm, case, system_answer)
    extracted_claims["pipeline_state"] = routing_state
    deterministic_results = [run_benchmark._evaluate_deterministic_check(c, extracted_claims) for c in deterministic_checks]
    reasoning_results = run_benchmark._score_reasoning_checks(reasoning_llm, case, system_answer, reasoning_checks)
    fact_score = round(sum(r["weighted_score"] for r in deterministic_results), 3)
    reasoning_score = round(sum(r["weighted_score"] for r in reasoning_results), 3)
    judge_score_raw = round(fact_score + reasoning_score, 3)
    judge_score = run_benchmark._compat_score(judge_score_raw)
    return {
        "id": case["id"], "tier": case.get("tier", ""), "category": case.get("category", ""),
        "question": case["question"], "system_answer": system_answer,
        "judge_score": judge_score, "judge_score_raw": judge_score_raw,
        "fact_score": fact_score, "reasoning_score": reasoning_score,
        "extracted_claims": extracted_claims,
        "fact_checks_passed": [r for r in deterministic_results if r["passed"]],
        "fact_checks_failed": [r for r in deterministic_results if not r["passed"]],
        "reasoning_checks": reasoning_results, "routing_state": routing_state,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--benchmark", type=Path,
                        default=PROJECT_ROOT / "configs/benchmark/benchmark_incident_heldout.json")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--pipeline-model", type=str, default="google/gemini-3.1-flash-lite-preview")
    parser.add_argument("--llm-base-url", type=str, default="https://openrouter.ai/api/v1")
    parser.add_argument("--extractor-model", type=str, default="google/gemini-2.5-flash")
    parser.add_argument("--reasoning-model", type=str, default="google/gemini-2.5-flash")
    parser.add_argument("--mode", choices=["naive", "local", "global", "hybrid", "mix"], default="hybrid")
    parser.add_argument("--cases", nargs="+", default=None)
    args = parser.parse_args()

    api_key = os.environ.get("OPENROUTER_API_KEY_HELDOUT") or os.environ.get("OPENROUTER_API_KEY", "")
    if not api_key:
        raise RuntimeError("OPENROUTER_API_KEY required")

    benchmark = json.loads(args.benchmark.read_text())
    cases = benchmark.get("cases", [])
    if args.cases:
        cases = [c for c in cases if c["id"] in args.cases]
    log.info("Running end-to-end LightRAG on %d cases (model=%s, mode=%s)",
             len(cases), args.pipeline_model, args.mode)

    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    rag = loop.run_until_complete(_build_rag(args.pipeline_model, args.llm_base_url, api_key))

    extractor_llm = run_benchmark._make_llm(args.extractor_model)
    reasoning_llm = run_benchmark._make_llm(args.reasoning_model)

    results = []
    for i, case in enumerate(cases, 1):
        raw = loop.run_until_complete(_ask(rag, case["question"], mode=args.mode))
        routing = synthesise_routing_state(case)
        if not raw:
            results.append({
                "id": case["id"], "tier": case.get("tier",""), "category": case.get("category",""),
                "question": case["question"], "system_answer": "", "judge_score": 1,
                "judge_score_raw": 0.0, "fact_score": 0.0, "reasoning_score": 0.0,
                "extracted_claims": {}, "fact_checks_passed": [], "fact_checks_failed": [],
                "reasoning_checks": [], "routing_state": routing,
                "skipped_reason": "LightRAG returned empty",
            })
            log.info("[%d/%d] %s: SKIPPED", i, len(cases), case["id"])
            continue
        system_answer = render_answer(case["question"], raw)
        result = score_one(case, system_answer, routing, extractor_llm, reasoning_llm)
        result["lightrag_raw_answer"] = raw
        results.append(result)
        log.info("[%d/%d] %s judge=%d fact=%.2f reasoning=%.2f", i, len(cases),
                 case["id"], result["judge_score"], result["fact_score"], result["reasoning_score"])

    scores = [r["judge_score"] for r in results]
    fact_scores = [r["fact_score"] for r in results]
    reasoning_scores = [r["reasoning_score"] for r in results]
    output = {
        "ablation": "lightrag-e2e", "pipeline_model": args.pipeline_model,
        "claim_extractor_model": args.extractor_model, "reasoning_judge_model": args.reasoning_model,
        "lightrag_mode": args.mode,
        "mean_judge_score": round(sum(scores)/max(len(scores),1), 3),
        "mean_fact_score": round(sum(fact_scores)/max(len(fact_scores),1), 3),
        "mean_reasoning_score": round(sum(reasoning_scores)/max(len(reasoning_scores),1), 3),
        "score_distribution": {str(i): scores.count(i) for i in range(1, 6)},
        "total": len(results), "cases": results,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(output, indent=2, default=str))
    log.info("Wrote %s (mean=%.3f / %d cases)", args.output, output["mean_judge_score"], len(results))


if __name__ == "__main__":
    main()
