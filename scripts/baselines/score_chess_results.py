"""Score CHESS pipeline outputs through the TRIAGE evaluation harness.

CHESS writes per-question SQL predictions to
``results/<data_mode>/<setting_name>/<dataset_name>/<run_ts>/-predictions.json``
with the BIRD format ``{"<qid>": "<sql>\\t----- bird -----\\t<db_id>"}``.

This script:
  1. Loads CHESS predictions and joins them back to TRIAGE benchmark cases via
     the ``triage_case_id`` field stitched in by ``triage_to_bird_format.py``.
  2. Executes each SQL against the SAME DuckDB instance used by the rest of
     the paper (NOT the SQLite snapshot CHESS used) — this keeps the
     comparison anchored to a single ground-truth database.
  3. Renders a simple natural-language answer string from the SQL + result
     rows, embedding the structured-fact patterns the TRIAGE claim extractor
     looks for (``- subject: X  metric: Y  value: Z`` lines).
  4. Reuses the TRIAGE benchmark harness's claim extractor + reasoning judge
     to score each case identically to other ablation runs.
  5. Emits a result JSON with the same schema as ``run_benchmark.py``'s
     incident-mode output, so downstream figures and paper tables can pick it
     up by file pattern.

Usage:
    PYTHONPATH=. conda run -n disaster_graph_rag --no-capture-output python -u \\
        scripts/baselines/score_chess_results.py \\
            --chess-run-dir external/chess/results/dev/CHESS_TRIAGE_IR_SS_CG/triage_heldout/<ts>/ \\
            --benchmark config/benchmark/benchmark_incident_heldout.json \\
            --output config/benchmark/results/gemini-3.1-flash-lite-preview/incident_chess_<ts>.json
"""
from __future__ import annotations

import argparse
import json
import logging
import os
import re
from pathlib import Path
from typing import Any, Iterable

import duckdb

# Reuse TRIAGE's extractor + judge code path. These imports execute scripts/run_benchmark.py
# at module level, which is fine because that file's top-level mostly just defines
# helpers and does not start a benchmark run.
import sys
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "scripts"))
import run_benchmark  # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger("score_chess")

DUCKDB_PATH = Path(os.environ.get("DISASTER_DB_PATH", "/data4/disaster_ai_sp26/disaster_ai_db.duckdb"))


def load_chess_predictions(run_dir: Path) -> dict[int, str]:
    """Load CHESS's -predictions.json and parse the BIRD-format SQL strings."""
    pred_path = run_dir / "-predictions.json"
    if not pred_path.exists():
        raise FileNotFoundError(f"CHESS predictions not found at {pred_path}")
    raw = json.loads(pred_path.read_text())
    parsed: dict[int, str] = {}
    for qid_str, val in raw.items():
        qid = int(qid_str)
        if isinstance(val, str) and "----- bird -----" in val:
            sql = val.split("----- bird -----")[0].strip()
        elif isinstance(val, str):
            sql = val.strip()
        else:
            sql = ""
        parsed[qid] = sql
    return parsed


def load_bird_data(bird_path: Path) -> list[dict]:
    return json.loads(bird_path.read_text())


def load_triage_cases(benchmark_path: Path) -> dict[str, dict]:
    data = json.loads(benchmark_path.read_text())
    return {case["id"]: case for case in data.get("cases", [])}


def execute_sql_safely(con: duckdb.DuckDBPyConnection, sql: str, timeout_s: float = 30.0):
    """Execute a SQL string against DuckDB and return (columns, rows) or
    ('ERROR', error-message)."""
    if not sql:
        return ("ERROR", "empty SQL")
    try:
        cur = con.execute(sql)
        rows = cur.fetchall()
        cols = [d[0] for d in cur.description] if cur.description else []
        return (cols, rows)
    except Exception as e:  # noqa: BLE001
        return ("ERROR", f"{type(e).__name__}: {e}")


def render_answer(question: str, sql: str, exec_result) -> str:
    """Produce a natural-language answer from the SQL and its result rows.

    The format mirrors TRIAGE's structured-fact pattern so the same claim
    extractor parses CHESS answers and TRIAGE answers symmetrically.
    """
    cols, rows_or_err = exec_result
    if cols == "ERROR":
        return (
            f"The CHESS baseline produced the following SQL but it failed to execute:\n\n"
            f"```sql\n{sql}\n```\n\n"
            f"Error: {rows_or_err}\n\n"
            f"STRUCTURED FACTS:\n"
        )

    rows = rows_or_err
    lines = [
        "The CHESS baseline produced the following SQL and executed it against the DuckDB:",
        "",
        f"```sql\n{sql}\n```",
        "",
        "Result rows:",
    ]
    if not rows:
        lines.append("  (no rows returned)")
    else:
        for r in rows[:20]:  # cap to first 20 rows
            lines.append("  " + " | ".join(f"{c}={v}" for c, v in zip(cols, r)))
        if len(rows) > 20:
            lines.append(f"  ... ({len(rows) - 20} more rows)")

    # Emit STRUCTURED FACTS the TRIAGE claim extractor recognises. The simplest
    # high-recall form: one fact per result-row scalar value.
    fact_lines = []
    for r in rows[:20]:
        for c, v in zip(cols, r):
            if isinstance(v, (int, float)) and v is not None:
                # subject defaults to the question's geo if recoverable, else
                # to the column name; metric is the column name.
                fact_lines.append(f"- subject: {c}  metric: {c}  value: {v}")
    if fact_lines:
        lines.append("")
        lines.append("STRUCTURED FACTS:")
        lines.extend(fact_lines)
    else:
        lines.append("")
        lines.append("STRUCTURED FACTS:")

    return "\n".join(lines)


def synthesise_routing_state(case: dict) -> dict:
    """CHESS has no orchestrator, so it produces no routing decision.

    For Tier R checks (which compare against expected criticality/cluster/
    query_type), we return an empty/dummy routing state so the deterministic
    checker treats them as failed. This is the correct comparison: CHESS has
    no routing layer, so it should score 0 on routing.
    """
    return {
        "criticality": 0,
        "cluster": "",
        "query_type": "",
        "area_of_interest": "",
        "hazard_type": "",
        "data_availability_warnings": [],
    }


def score_one(
    case: dict,
    system_answer: str,
    routing_state: dict,
    extractor_llm: Any,
    reasoning_llm: Any,
) -> dict:
    """Score a single (case, answer) pair using TRIAGE's harness internals."""
    deterministic_checks = [c for c in case.get("checks", []) if c.get("evaluator") == "deterministic"]
    reasoning_checks = [c for c in case.get("checks", []) if c.get("evaluator") == "reasoning_llm"]

    extracted_claims = run_benchmark._extract_case_claims(extractor_llm, case, system_answer)
    # Inject routing state so pipeline_state_match checks score consistently.
    extracted_claims["pipeline_state"] = routing_state

    deterministic_results = [
        run_benchmark._evaluate_deterministic_check(check, extracted_claims)
        for check in deterministic_checks
    ]
    reasoning_results = run_benchmark._score_reasoning_checks(
        reasoning_llm, case, system_answer, reasoning_checks
    )

    fact_score = round(sum(r["weighted_score"] for r in deterministic_results), 3)
    reasoning_score = round(sum(r["weighted_score"] for r in reasoning_results), 3)
    judge_score_raw = round(fact_score + reasoning_score, 3)
    judge_score = run_benchmark._compat_score(judge_score_raw)

    fact_passed = [r for r in deterministic_results if r["passed"]]
    fact_failed = [r for r in deterministic_results if not r["passed"]]
    return {
        "id": case["id"],
        "tier": case.get("tier", ""),
        "category": case.get("category", ""),
        "question": case["question"],
        "system_answer": system_answer,
        "judge_score": judge_score,
        "judge_score_raw": judge_score_raw,
        "fact_score": fact_score,
        "reasoning_score": reasoning_score,
        "extracted_claims": extracted_claims,
        "fact_checks_passed": fact_passed,
        "fact_checks_failed": fact_failed,
        "reasoning_checks": reasoning_results,
        "routing_state": routing_state,
    }


def main(argv: Iterable[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--chess-run-dir", type=Path, required=True,
                        help="CHESS run directory containing -predictions.json")
    parser.add_argument("--bird-data", type=Path,
                        default=PROJECT_ROOT / "external/chess/data/triage_heldout.json",
                        help="BIRD-format dataset CHESS consumed (for question_id → triage_case_id)")
    parser.add_argument("--benchmark", type=Path,
                        default=PROJECT_ROOT / "config/benchmark/benchmark_incident_heldout.json")
    parser.add_argument("--output", type=Path, required=True,
                        help="Destination result JSON (TRIAGE-compatible schema)")
    parser.add_argument("--extractor-model", type=str, default="google/gemini-2.5-flash")
    parser.add_argument("--reasoning-model", type=str, default="google/gemini-2.5-flash")
    args = parser.parse_args(argv)

    bird_records = load_bird_data(args.bird_data)
    qid_to_triage_id = {rec["question_id"]: rec["triage_case_id"] for rec in bird_records}
    cases_by_id = load_triage_cases(args.benchmark)
    predictions = load_chess_predictions(args.chess_run_dir)

    log.info("CHESS run dir: %s", args.chess_run_dir)
    log.info("Predictions: %d, benchmark cases: %d", len(predictions), len(cases_by_id))

    extractor_llm = run_benchmark._make_llm(args.extractor_model)
    reasoning_llm = run_benchmark._make_llm(args.reasoning_model)

    con = duckdb.connect(str(DUCKDB_PATH), read_only=True)

    results: list[dict] = []
    for qid, sql in sorted(predictions.items()):
        triage_id = qid_to_triage_id.get(qid)
        if triage_id is None:
            log.warning("qid=%s missing in BIRD data; skipping", qid)
            continue
        case = cases_by_id.get(triage_id)
        if case is None:
            log.warning("triage_case_id=%s not in benchmark; skipping", triage_id)
            continue

        # Cases CHESS hasn't reached or that produced no SQL are recorded as
        # judge=1 / fact=0 / reasoning=0. We skip the claim extractor entirely
        # rather than letting a "failed to execute" boilerplate answer
        # accidentally satisfy Tier D disclosure checks.
        if not sql:
            results.append({
                "id": case["id"],
                "tier": case.get("tier", ""),
                "category": case.get("category", ""),
                "question": case["question"],
                "system_answer": "",
                "judge_score": 1,
                "judge_score_raw": 0.0,
                "fact_score": 0.0,
                "reasoning_score": 0.0,
                "extracted_claims": {},
                "fact_checks_passed": [],
                "fact_checks_failed": [],
                "reasoning_checks": [],
                "routing_state": synthesise_routing_state(case),
                "chess_sql": "",
                "skipped_reason": "no SQL produced",
            })
            log.info("[%s] no SQL — recorded as judge=1 fact=0", triage_id)
            continue

        exec_result = execute_sql_safely(con, sql)
        system_answer = render_answer(case["question"], sql, exec_result)
        routing_state = synthesise_routing_state(case)
        result = score_one(case, system_answer, routing_state, extractor_llm, reasoning_llm)
        result["chess_sql"] = sql
        results.append(result)
        log.info("[%s] judge=%d fact=%.2f reasoning=%.2f", triage_id,
                 result["judge_score"], result["fact_score"], result["reasoning_score"])

    scores = [r["judge_score"] for r in results]
    fact_scores = [r["fact_score"] for r in results]
    reasoning_scores = [r["reasoning_score"] for r in results]

    output = {
        "ablation": "chess",
        "pipeline_model": "google/gemini-3.1-flash-lite-preview",  # caller can override per run
        "claim_extractor_model": args.extractor_model,
        "reasoning_judge_model": args.reasoning_model,
        "mean_judge_score": round(sum(scores) / max(len(scores), 1), 3),
        "mean_fact_score": round(sum(fact_scores) / max(len(fact_scores), 1), 3),
        "mean_reasoning_score": round(sum(reasoning_scores) / max(len(reasoning_scores), 1), 3),
        "score_distribution": {str(i): scores.count(i) for i in range(1, 6)},
        "total": len(results),
        "cases": results,
    }

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(output, indent=2, default=str))
    log.info("Wrote %s (mean=%.3f / %d cases)", args.output, output["mean_judge_score"], len(results))


if __name__ == "__main__":
    main()
