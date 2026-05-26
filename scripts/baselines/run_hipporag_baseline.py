"""Run HippoRAG 2 end-to-end on the TRIAGE 75-case heldout split.

Mirror of run_graphrag_baseline.py: HippoRAG produces an answer per query;
we feed it through the same TRIAGE claim extractor + judge harness as every
other ablation.

Pre-reqs:
  - HippoRAG index built at ``data/hipporag_index/`` (see index_hipporag.py)
  - OPENROUTER_API_KEY in env (LLM)
  - OPENAI_API_KEY in env (embeddings)
"""
from __future__ import annotations

import argparse
import json
import logging
import os
import sys
from pathlib import Path

# Stub vllm + outlines BEFORE importing hipporag (see index_hipporag.py).
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

# Monkey-patch openai.OpenAI to route api_key by base_url (see index_hipporag.py).
import openai as _openai_mod  # noqa: E402

_real_openai_init = _openai_mod.OpenAI.__init__

def _patched_openai_init(self, *args, **kw):  # type: ignore
    base_url = kw.get("base_url") or (args[1] if len(args) > 1 else None)
    if kw.get("api_key") is None and base_url and "openrouter" in str(base_url).lower():
        kw["api_key"] = os.environ.get("OPENROUTER_API_KEY_HELDOUT") or os.environ.get("OPENROUTER_API_KEY")
    return _real_openai_init(self, *args, **kw)

_openai_mod.OpenAI.__init__ = _patched_openai_init

# Cap HippoRAG's OpenIE ThreadPoolExecutor concurrency (see index_hipporag.py).
import concurrent.futures as _cfutures  # noqa: E402
_real_tpe = _cfutures.ThreadPoolExecutor

class _CappedTPE(_real_tpe):
    def __init__(self, max_workers=None, *a, **kw):
        if max_workers is None or max_workers > 4:
            max_workers = 4
        super().__init__(max_workers=max_workers, *a, **kw)

_cfutures.ThreadPoolExecutor = _CappedTPE

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "scripts"))
import run_benchmark  # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger("hipporag-bench")


def render_answer(question: str, raw_answer: str) -> str:
    return (
        f"The HippoRAG 2 baseline produced the following answer:\n\n"
        f"{raw_answer.strip()}\n\n"
        f"STRUCTURED FACTS:\n"
    )


def synthesise_routing_state(case: dict) -> dict:
    return {
        "criticality": 0, "cluster": "", "query_type": "",
        "area_of_interest": "", "hazard_type": "", "data_availability_warnings": [],
    }


def score_one(case, system_answer, routing_state, extractor_llm, reasoning_llm):
    deterministic_checks = [c for c in case.get("checks", []) if c.get("evaluator") == "deterministic"]
    reasoning_checks = [c for c in case.get("checks", []) if c.get("evaluator") == "reasoning_llm"]
    extracted_claims = run_benchmark._extract_case_claims(extractor_llm, case, system_answer)
    extracted_claims["pipeline_state"] = routing_state
    deterministic_results = [run_benchmark._evaluate_deterministic_check(check, extracted_claims)
                             for check in deterministic_checks]
    reasoning_results = run_benchmark._score_reasoning_checks(
        reasoning_llm, case, system_answer, reasoning_checks)
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
    parser.add_argument("--save-dir", type=Path, default=PROJECT_ROOT / "data/hipporag_index")
    parser.add_argument("--benchmark", type=Path,
                        default=PROJECT_ROOT / "configs/benchmark/benchmark_incident_heldout.json")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--pipeline-model", type=str,
                        default="google/gemini-3.1-flash-lite-preview")
    parser.add_argument("--llm-base-url", type=str,
                        default="https://openrouter.ai/api/v1")
    parser.add_argument("--embedding-model", type=str, default="text-embedding-3-small")
    parser.add_argument("--extractor-model", type=str, default="google/gemini-2.5-flash")
    parser.add_argument("--reasoning-model", type=str, default="google/gemini-2.5-flash")
    parser.add_argument("--cases", nargs="+", default=None)
    args = parser.parse_args()

    from hipporag import HippoRAG  # type: ignore

    os.environ.setdefault("OPENAI_API_KEY", os.environ.get("OPENROUTER_API_KEY", ""))

    benchmark = json.loads(args.benchmark.read_text())
    cases = benchmark.get("cases", [])
    if args.cases:
        cases = [c for c in cases if c["id"] in args.cases]
    log.info("Running HippoRAG on %d cases (model=%s)", len(cases), args.pipeline_model)

    hipporag = HippoRAG(
        save_dir=str(args.save_dir),
        llm_model_name=args.pipeline_model,
        llm_base_url=args.llm_base_url,
        embedding_model_name=args.embedding_model,
    )

    extractor_llm = run_benchmark._make_llm(args.extractor_model)
    reasoning_llm = run_benchmark._make_llm(args.reasoning_model)

    results = []
    queries = [c["question"] for c in cases]
    # rag_qa returns a 3- or 5-element tuple where index 0 is the list of
    # QuerySolution objects and index 1 is the list of answer strings.
    try:
        rqa_out = hipporag.rag_qa(queries=queries)
        answers = rqa_out[1] if len(rqa_out) >= 2 else []
    except Exception as e:  # noqa: BLE001
        log.error("HippoRAG batch query failed: %s; falling back to per-case", e)
        answers = []
        for q in queries:
            try:
                rqa_out = hipporag.rag_qa(queries=[q])
                answers.append(rqa_out[1][0] if len(rqa_out) >= 2 and rqa_out[1] else "")
            except Exception as e2:  # noqa: BLE001
                log.warning("Per-case HippoRAG failed: %s", e2)
                answers.append("")

    for i, case in enumerate(cases):
        raw = str(answers[i]) if i < len(answers) and answers[i] else ""

        system_answer = render_answer(case["question"], raw) if raw else ""
        routing = synthesise_routing_state(case)
        if not raw:
            results.append({
                "id": case["id"], "tier": case.get("tier",""), "category": case.get("category",""),
                "question": case["question"], "system_answer": "", "judge_score": 1,
                "judge_score_raw": 0.0, "fact_score": 0.0, "reasoning_score": 0.0,
                "extracted_claims": {}, "fact_checks_passed": [], "fact_checks_failed": [],
                "reasoning_checks": [], "routing_state": routing,
                "skipped_reason": "HippoRAG produced no answer",
            })
            log.info("[%d/%d] %s: SKIPPED", i+1, len(cases), case["id"])
            continue
        result = score_one(case, system_answer, routing, extractor_llm, reasoning_llm)
        result["hipporag_raw_answer"] = raw
        results.append(result)
        log.info("[%d/%d] %s judge=%d fact=%.2f reasoning=%.2f", i+1, len(cases),
                 case["id"], result["judge_score"], result["fact_score"], result["reasoning_score"])

    scores = [r["judge_score"] for r in results]
    fact_scores = [r["fact_score"] for r in results]
    reasoning_scores = [r["reasoning_score"] for r in results]
    output = {
        "ablation": "hipporag",
        "pipeline_model": args.pipeline_model,
        "claim_extractor_model": args.extractor_model,
        "reasoning_judge_model": args.reasoning_model,
        "embedding_model": args.embedding_model,
        "mean_judge_score": round(sum(scores)/max(len(scores),1), 3),
        "mean_fact_score": round(sum(fact_scores)/max(len(fact_scores),1), 3),
        "mean_reasoning_score": round(sum(reasoning_scores)/max(len(reasoning_scores),1), 3),
        "score_distribution": {str(i): scores.count(i) for i in range(1, 6)},
        "total": len(results),
        "cases": results,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(output, indent=2, default=str))
    log.info("Wrote %s (mean=%.3f / %d cases)", args.output, output["mean_judge_score"], len(results))


if __name__ == "__main__":
    main()
