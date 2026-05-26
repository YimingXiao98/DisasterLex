"""
DisasterLex — Rescore Existing Benchmark Results

Re-scores a saved benchmark result file WITHOUT re-running the pipeline.
For each case it re-extracts claims from the saved ``system_answer`` using a
(potentially different) extractor model, then re-runs deterministic and
reasoning checks against the gold facts from the benchmark JSON.

Usage::

    python scripts/rescore_results.py \\
        --input  results/gemini-3.1-flash-lite-preview/full_seed1.json \\
        --output results/full_seed1_rescored.json \\
        --extractor-model google/gemini-2.5-flash

    # Rescore specific cases only
    python scripts/rescore_results.py \\
        --input  results/full_seed1.json \\
        --output results/full_seed1_rescored.json \\
        --extractor-model google/gemini-2.5-flash \\
        --cases draft_k13 draft_m18
"""
from __future__ import annotations

import argparse
import concurrent.futures
import json
import logging
import os
import sys
from datetime import datetime
from pathlib import Path
from typing import Any

from dotenv import load_dotenv

load_dotenv()
logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

_DEFAULT_BENCHMARK = PROJECT_ROOT / "configs" / "benchmark" / "dev.json"

# ---------------------------------------------------------------------------
# Import scoring helpers from run_benchmark — single source of truth
# ---------------------------------------------------------------------------
from scripts.run_benchmark import (  # noqa: E402
    _collect_extraction_targets,
    _empty_claims,
    _evaluate_deterministic_check,
    _extract_case_claims,
    _find_boolean_claim,
    _find_numeric_claim,
    _load_json_response,
    _make_llm,
    _normalize_text,
    _score_boolean_check,
    _score_entity_check,
    _score_numeric_check,
    _score_ordered_check,
    _score_reasoning_checks,
    _score_recommendation_count,
    _compat_score,
)


# ---------------------------------------------------------------------------
# Core per-case rescoring
# ---------------------------------------------------------------------------

def _rescore_single_case(
    result_case: dict,
    benchmark_case: dict,
    extractor_llm: Any,
    reasoning_llm: Any,
    verbose: bool,
) -> dict:
    """Re-extract claims and re-score one case from saved pipeline output.

    Args:
        result_case:    The case dict as stored in the original result file.
        benchmark_case: The matching case dict from the benchmark JSON (gold
                        facts + check specs).
        extractor_llm:  LLM instance used for claim extraction.
        reasoning_llm:  LLM instance used for reasoning_llm checks.
        verbose:        Log extra detail per case.

    Returns:
        Updated result dict with fresh scores.  Preserves all original fields
        that are not re-computed (e.g., ``system_answer``, ``routing_state``,
        ``tool_calls``).
    """
    cid = result_case.get("id", benchmark_case.get("id", "unknown"))
    system_answer: str = result_case.get("system_answer", "")

    # routing_state from the original run — used for pipeline_state_match checks
    routing_state: dict = result_case.get("routing_state", {})

    deterministic_checks = [
        c for c in benchmark_case.get("checks", []) if c.get("evaluator") == "deterministic"
    ]
    reasoning_checks = [
        c for c in benchmark_case.get("checks", []) if c.get("evaluator") == "reasoning_llm"
    ]

    try:
        # Re-extract claims from the saved answer
        extracted_claims = _extract_case_claims(extractor_llm, benchmark_case, system_answer)
        # Inject routing state so pipeline_state_match checks can read it
        extracted_claims["__routing_state__"] = routing_state

        # Deterministic checks
        deterministic_results = [
            _evaluate_deterministic_check(check, extracted_claims)
            for check in deterministic_checks
        ]

        # Reasoning checks (LLM-based)
        reasoning_results = _score_reasoning_checks(
            reasoning_llm, benchmark_case, system_answer, reasoning_checks
        )

        all_check_results = deterministic_results + reasoning_results
        fact_score = round(sum(r["weighted_score"] for r in deterministic_results), 3)
        reasoning_score = round(sum(r["weighted_score"] for r in reasoning_results), 3)
        judge_score_raw = round(fact_score + reasoning_score, 3)
        judge_score = _compat_score(judge_score_raw)

        failed_check_ids = [r["id"] for r in all_check_results if r["score_fraction"] < 1.0]
        judge_reason = (
            "All grounded checks passed."
            if not failed_check_ids
            else f"Incomplete: {', '.join(failed_check_ids)}"
        )

        updated = dict(result_case)
        updated.update(
            {
                "judge_score": judge_score,
                "judge_score_raw": judge_score_raw,
                "fact_score": fact_score,
                "reasoning_score": reasoning_score,
                "judge_reason": judge_reason,
                "fact_checks_passed": [r for r in deterministic_results if r["passed"]],
                "fact_checks_failed": [r for r in deterministic_results if not r["passed"]],
                "reasoning_checks": reasoning_results,
                "extracted_claims": {
                    k: v for k, v in extracted_claims.items() if k != "__routing_state__"
                },
            }
        )

        if verbose or judge_score <= 2:
            logger.info(
                "[rescore] %s: score=%.2f/5 -> %s/5 | fact=%.2f reason=%.2f | %s",
                cid,
                judge_score_raw,
                judge_score,
                fact_score,
                reasoning_score,
                judge_reason,
            )
        return updated

    except Exception as exc:
        logger.error("[rescore] %s: EXCEPTION — %s", cid, exc)
        updated = dict(result_case)
        updated["rescore_error"] = str(exc)
        return updated


# ---------------------------------------------------------------------------
# Main rescoring loop
# ---------------------------------------------------------------------------

def rescore_results(
    result_path: str,
    benchmark_path: str,
    extractor_model: str,
    reasoning_model: str,
    output_path: str,
    case_filter: list[str] | None = None,
    parallel: int = 1,
    verbose: bool = False,
) -> dict:
    """Load an existing result file, re-score all cases, and write output.

    Args:
        result_path:     Path to the original benchmark result JSON.
        benchmark_path:  Path to the benchmark JSON (gold checks + facts).
        extractor_model: OpenRouter model id for claim extraction.
        reasoning_model: OpenRouter model id for reasoning_llm checks.
        output_path:     Where to write the updated result JSON.
        case_filter:     Optional list of case ids to rescore (others pass
                         through unchanged).
        parallel:        Concurrent extraction threads.
        verbose:         Log per-case detail.

    Returns:
        The updated result dict (also written to ``output_path``).
    """
    # ── Load inputs ───────────────────────────────────────────────────────────
    with open(result_path) as f:
        original_result: dict = json.load(f)

    with open(benchmark_path) as f:
        benchmark_data: dict = json.load(f)

    benchmark_cases: dict[str, dict] = {
        c["id"]: c for c in benchmark_data.get("cases", [])
    }

    original_cases: list[dict] = original_result.get("cases", [])

    # Determine which cases to actually rescore
    if case_filter:
        to_rescore_ids = set(case_filter)
    else:
        to_rescore_ids = {c["id"] for c in original_cases}

    # Build list of (result_case, benchmark_case) pairs that need rescoring
    rescore_pairs: list[tuple[dict, dict]] = []
    passthrough: list[dict] = []

    for rc in original_cases:
        cid = rc.get("id", "")
        if cid not in to_rescore_ids:
            passthrough.append(rc)
            continue
        bc = benchmark_cases.get(cid)
        if bc is None:
            logger.warning("[rescore] Case %s not found in benchmark — skipping rescore", cid)
            passthrough.append(rc)
            continue
        rescore_pairs.append((rc, bc))

    n_rescore = len(rescore_pairs)
    n_pass = len(passthrough)
    print(f"\nRescoring {n_rescore} case(s)  |  {n_pass} case(s) passed through unchanged")
    print(f"Extractor model : {extractor_model}")
    print(f"Reasoning model : {reasoning_model}")

    extractor_llm = _make_llm(extractor_model)
    reasoning_llm = _make_llm(reasoning_model)

    # ── Execute (serial or parallel) ──────────────────────────────────────────
    order: list[str] = [rc["id"] for rc in original_cases]  # preserve original order
    result_by_id: dict[str, dict] = {}

    for rc in passthrough:
        result_by_id[rc["id"]] = rc

    if parallel > 1:
        with concurrent.futures.ThreadPoolExecutor(max_workers=parallel) as executor:
            future_to_id = {
                executor.submit(
                    _rescore_single_case,
                    rc,
                    bc,
                    extractor_llm,
                    reasoning_llm,
                    verbose,
                ): rc["id"]
                for rc, bc in rescore_pairs
            }
            done_count = 0
            for future in concurrent.futures.as_completed(future_to_id):
                cid = future_to_id[future]
                done_count += 1
                try:
                    updated = future.result()
                    result_by_id[cid] = updated
                    score_str = f"judge={updated.get('judge_score', '?')} fact={updated.get('fact_score', '?'):.2f}"
                    print(f"  [{done_count}/{n_rescore}] {cid}: {score_str}", flush=True)
                except Exception as exc:
                    logger.error("[rescore] %s: EXECUTOR EXCEPTION — %s", cid, exc)
                    # Keep original result on unexpected executor failure
                    orig = next(rc for rc, _ in rescore_pairs if rc["id"] == cid)
                    orig["rescore_error"] = str(exc)
                    result_by_id[cid] = orig
                    print(f"  [{done_count}/{n_rescore}] {cid}: ERROR — {exc}", flush=True)
    else:
        for i, (rc, bc) in enumerate(rescore_pairs, 1):
            cid = rc["id"]
            updated = _rescore_single_case(rc, bc, extractor_llm, reasoning_llm, verbose)
            result_by_id[cid] = updated
            score_str = f"judge={updated.get('judge_score', '?')} fact={updated.get('fact_score', '?'):.2f}"
            print(f"  [{i}/{n_rescore}] {cid}: {score_str}", flush=True)

    # ── Reconstruct cases in original order ───────────────────────────────────
    new_cases: list[dict] = [result_by_id[cid] for cid in order if cid in result_by_id]

    # ── Diff summary ──────────────────────────────────────────────────────────
    orig_by_id: dict[str, dict] = {c["id"]: c for c in original_cases}
    changed: list[dict] = []
    for nc in new_cases:
        cid = nc["id"]
        orig = orig_by_id.get(cid, {})
        old_fact = orig.get("fact_score", None)
        new_fact = nc.get("fact_score", None)
        old_judge = orig.get("judge_score_raw", orig.get("judge_score", None))
        new_judge = nc.get("judge_score_raw", nc.get("judge_score", None))
        if old_fact is not None and new_fact is not None and abs(new_fact - old_fact) >= 0.001:
            changed.append(
                {
                    "id": cid,
                    "old_fact_score": old_fact,
                    "new_fact_score": new_fact,
                    "delta_fact": round(new_fact - old_fact, 3),
                    "old_judge_score_raw": old_judge,
                    "new_judge_score_raw": new_judge,
                    "delta_judge": round(new_judge - old_judge, 3) if (new_judge is not None and old_judge is not None) else None,
                }
            )

    print(f"\n{'=' * 60}")
    if changed:
        print(f"Score changed for {len(changed)} case(s):")
        for entry in sorted(changed, key=lambda x: abs(x["delta_fact"]), reverse=True):
            sign = "+" if entry["delta_fact"] >= 0 else ""
            print(
                f"  {entry['id']}: fact {entry['old_fact_score']:.3f} -> {entry['new_fact_score']:.3f} "
                f"({sign}{entry['delta_fact']:.3f})"
            )
    else:
        print("No score changes detected (or no original scores to compare).")

    # ── Recompute aggregate metrics ───────────────────────────────────────────
    all_scores: list[float] = [c.get("judge_score", 1) for c in new_cases]
    all_fact: list[float] = [c.get("fact_score", 0.0) for c in new_cases]
    all_reasoning: list[float] = [c.get("reasoning_score", 0.0) for c in new_cases]

    n = len(new_cases)
    mean_judge = round(sum(all_scores) / n, 3) if n > 0 else 0.0
    mean_fact = round(sum(all_fact) / n, 3) if n > 0 else 0.0
    mean_reasoning = round(sum(all_reasoning) / n, 3) if n > 0 else 0.0

    # Routing accuracy — count pipeline_state_match checks across cases
    routing_correct = 0
    routing_total = 0
    for nc in new_cases:
        for r in nc.get("fact_checks_passed", []) + nc.get("fact_checks_failed", []):
            if r.get("kind") == "pipeline_state_match":
                routing_total += 1
                if r.get("passed"):
                    routing_correct += 1

    routing_accuracy = round(routing_correct / routing_total, 3) if routing_total > 0 else None

    print(
        f"\nRescored summary: mean={mean_judge:.2f}/5 | fact={mean_fact:.2f} | reasoning={mean_reasoning:.2f}"
        + (f" | routing_accuracy={routing_accuracy:.0%}" if routing_accuracy is not None else "")
    )
    print(f"{'=' * 60}")

    # ── Build output dict ─────────────────────────────────────────────────────
    output_meta = dict(original_result)
    output_meta.update(
        {
            "rescored_from": str(result_path),
            "rescore_timestamp": datetime.now().strftime("%Y-%m-%dT%H:%M:%S"),
            "extractor_model": extractor_model,
            "reasoning_model": reasoning_model,
            "mean_judge_score": mean_judge,
            "mean_fact_score": mean_fact,
            "mean_reasoning_score": mean_reasoning,
            "routing_accuracy": routing_accuracy,
            "score_distribution": {str(i): all_scores.count(i) for i in range(1, 6)},
            "total": n,
            "score_diff": changed,
            "cases": new_cases,
        }
    )

    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w") as f:
        json.dump(output_meta, f, indent=2)

    print(f"\nRescored results saved to: {out}")
    return output_meta


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Re-score saved benchmark results without re-running the pipeline."
    )
    parser.add_argument(
        "--input",
        required=True,
        metavar="FILE",
        help="Path to existing benchmark result JSON to re-score.",
    )
    parser.add_argument(
        "--output",
        required=True,
        metavar="FILE",
        help="Path to write the updated result JSON.",
    )
    parser.add_argument(
        "--extractor-model",
        required=True,
        metavar="MODEL",
        help=(
            "OpenRouter model id to use for claim extraction "
            "(e.g. google/gemini-2.5-flash)."
        ),
    )
    parser.add_argument(
        "--reasoning-model",
        metavar="MODEL",
        default=None,
        help=(
            "OpenRouter model id to use for reasoning_llm checks. "
            "Defaults to the same model as --extractor-model."
        ),
    )
    parser.add_argument(
        "--benchmark",
        metavar="FILE",
        default=str(_DEFAULT_BENCHMARK),
        help=(
            "Path to benchmark JSON with gold checks "
            f"(default: {_DEFAULT_BENCHMARK})."
        ),
    )
    parser.add_argument(
        "--cases",
        nargs="+",
        metavar="CASE_ID",
        default=None,
        help=(
            "Rescore only these case ids; all others are passed through "
            "unchanged (e.g. incident_k02 incident_m06)."
        ),
    )
    parser.add_argument(
        "--parallel",
        type=int,
        default=1,
        metavar="N",
        help="Number of cases to extract/score concurrently (default: 1).",
    )
    parser.add_argument(
        "--verbose",
        "-v",
        action="store_true",
        help="Log per-case results even when score > 2.",
    )
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")

    reasoning_model = args.reasoning_model or args.extractor_model

    rescore_results(
        result_path=args.input,
        benchmark_path=args.benchmark,
        extractor_model=args.extractor_model,
        reasoning_model=reasoning_model,
        output_path=args.output,
        case_filter=args.cases,
        parallel=args.parallel,
        verbose=args.verbose,
    )


if __name__ == "__main__":
    main()
