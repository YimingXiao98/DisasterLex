"""Run ReFoRCE end-to-end on the TRIAGE 75-case heldout split.

We invoke ReFoRCE's REFORCE class's self_refine() method directly (skipping
the BIRD-mode gold-SQL eval path that doesn't fit our fact-checking harness).
For each case:
  1. Load table_info via ReFoRCE's get_table_info (introspects the SQLite).
  2. Build a REFORCE agent + chat session against our triage_disaster.sqlite.
  3. Call self_refine() with max_iter=3, do_self_consistency=False.
  4. Read the chosen SQL from sql.sql, execute it against DuckDB (our actual DB),
     render an answer block, score it through TRIAGE's claim extractor + judge.

Pre-reqs:
  - external/reforce/methods/ReFoRCE/ cloned
  - external/reforce/data/BIRD/dev_databases/triage_disaster/triage_disaster.sqlite
    (symlink to CHESS's SQLite snapshot)
  - OPENROUTER_API_KEY in env
"""
from __future__ import annotations

import argparse
import json
import logging
import os
import sys
import types
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent

# Route LLM calls through OpenRouter via openai SDK's env-var path. We OVERRIDE
# OPENAI_API_KEY even if it was set in .env (the .env value is the real OpenAI
# key for HippoRAG embeddings; using it against OpenRouter would 401).
os.environ["OPENAI_BASE_URL"] = "https://openrouter.ai/api/v1"
or_key = os.environ.get("OPENROUTER_API_KEY_HELDOUT") or os.environ.get("OPENROUTER_API_KEY")
if or_key:
    os.environ["OPENAI_API_KEY"] = or_key
else:
    raise RuntimeError("OPENROUTER_API_KEY required (sets the OPENAI_API_KEY env that ReFoRCE reads)")

REFORCE_DIR = PROJECT_ROOT / "external/reforce/methods/ReFoRCE"
REFORCE_DATA_BIRD = PROJECT_ROOT / "external/reforce/data/BIRD"
sys.path.insert(0, str(REFORCE_DIR))
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "scripts"))
import run_benchmark  # noqa: E402

# ReFoRCE imports
import sql as _reforce_sql  # noqa: E402,F401
from agent import REFORCE  # type: ignore  # noqa: E402
from chat import GPTChat  # type: ignore  # noqa: E402
from prompt import Prompts  # type: ignore  # noqa: E402
from utils import get_table_info, initialize_logger  # type: ignore  # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger("reforce-bench")


def _make_args(generation_model: str, output_dir: Path) -> argparse.Namespace:
    """Build a minimal argparse Namespace that REFORCE.self_refine reads."""
    return argparse.Namespace(
        task="BIRD", subtask="sqlite", db_path=str(REFORCE_DATA_BIRD),
        output_path=str(output_dir),
        do_format_restriction=False, use_gold_format=False, format_model="o3",
        do_column_exploration=False, column_exploration_model="o3",
        do_self_refinement=True, do_self_consistency=False,
        generation_model=generation_model, azure=False,
        max_iter=3, temperature=0.0, early_stop=True,
        do_vote=False, revote=False, num_votes=1, random_vote_for_tie=False,
        model_vote=None, final_choose=False, save_all_results=False,
        rerun=False, overwrite_unfinished=False, num_workers=1,
        omnisql_format_pth=None,
        gold_result_path=str(REFORCE_DATA_BIRD / "gold_result"),
    )


def build_table_info_from_sqlite(sqlite_path: Path) -> str:
    """Generate a ReFoRCE-style table_info description by introspecting SQLite."""
    import sqlite3
    con = sqlite3.connect(f"file:{sqlite_path}?mode=ro", uri=True)
    cur = con.cursor()
    tables = [r[0] for r in cur.execute("SELECT name FROM sqlite_master WHERE type='table' AND name NOT LIKE 'sqlite_%' ORDER BY name").fetchall()]
    lines = ["The table structure information is "]
    for t in tables:
        cols = cur.execute(f'PRAGMA table_info("{t}")').fetchall()
        col_lines = [f'  "{c[1]}" ({c[2]})' for c in cols]
        lines.append(f'Table: "{t}"')
        lines.extend(col_lines)
        lines.append("")
    con.close()
    return "\n".join(lines)


def run_reforce_one(case_id: str, question: str, args: argparse.Namespace,
                    sqlite_path: Path, output_root: Path) -> str:
    """Run REFORCE.self_refine for one case and return generated SQL (or '')."""
    from sql import SqlEnv  # type: ignore

    case_dir = output_root / case_id
    case_dir.mkdir(parents=True, exist_ok=True)
    log_path = case_dir / "log.log"
    sql_path = case_dir / "sql.sql"
    csv_path = case_dir / "result.csv"

    # If a previous run already produced SQL, reuse it.
    if sql_path.exists() and sql_path.stat().st_size > 0:
        return sql_path.read_text().strip()

    logger = initialize_logger(str(log_path))
    # Bypass ReFoRCE's get_table_info (which reads a pre-built prompts.txt file)
    # and introspect our SQLite directly.
    table_info = build_table_info_from_sqlite(sqlite_path)
    table_struct = table_info[table_info.find("The table structure information is "):]

    chat_session = GPTChat(azure=False, model=args.generation_model, temperature=args.temperature)
    sql_env = SqlEnv()
    prompts = Prompts()

    try:
        # `sql_data` must start with "local" so ReFoRCE's get_api_name returns
        # "sqlite" (the prefix is its API discriminator). The actual SQLite
        # path is resolved from db_id+task by get_sqlite_path.
        agent = REFORCE(
            db_path=args.db_path, sql_data="local_triage", search_directory=str(case_dir),
            prompt_class=prompts, sql_env=sql_env,
            chat_session_pre=None, chat_session=chat_session,
            log_save_path=str(log_path), db_id="triage_disaster", task=args.task,
        )
    except Exception as e:  # noqa: BLE001
        log.warning("[%s] REFORCE init failed: %s", case_id, e)
        return ""

    try:
        agent.self_refine(args, logger, question, format_csv=None,
                           table_struct=table_struct, table_info=table_info,
                           response_pre_txt=None, pre_info=None,
                           csv_save_path=str(csv_path), sql_save_path=str(sql_path),
                           task=args.task)
    except SystemExit:
        log.warning("[%s] self_refine called sys.exit (chat failure)", case_id)
        return ""
    except Exception as e:  # noqa: BLE001
        log.warning("[%s] self_refine errored: %s", case_id, e)
        return ""
    finally:
        try:
            sql_env.close_db()
        except Exception:
            pass

    if sql_path.exists():
        return sql_path.read_text().strip()
    return ""


def execute_against_duckdb(con, sql_text):
    if not sql_text:
        return ("ERROR", "empty SQL")
    try:
        cur = con.execute(sql_text)
        rows = cur.fetchall()
        cols = [d[0] for d in cur.description] if cur.description else []
        return (cols, rows)
    except Exception as e:  # noqa: BLE001
        return ("ERROR", f"{type(e).__name__}: {e}")


def render_answer(sql_text: str, exec_result) -> str:
    cols, rows = exec_result
    if cols == "ERROR":
        return (f"The ReFoRCE baseline produced the following SQL but it failed on DuckDB:\n\n"
                f"```sql\n{sql_text}\n```\n\nError: {rows}\n\nSTRUCTURED FACTS:\n")
    lines = ["The ReFoRCE baseline produced the following SQL and executed it on DuckDB:",
             "", f"```sql\n{sql_text}\n```", "", "Result rows:"]
    if not rows:
        lines.append("  (no rows returned)")
    else:
        for r in rows[:20]:
            lines.append("  " + " | ".join(f"{c}={v}" for c, v in zip(cols, r)))
        if len(rows) > 20:
            lines.append(f"  ... ({len(rows) - 20} more rows)")
    facts = []
    for r in rows[:20]:
        for c, v in zip(cols, r):
            if isinstance(v, (int, float)) and v is not None:
                facts.append(f"- subject: {c}  metric: {c}  value: {v}")
    lines.append("")
    lines.append("STRUCTURED FACTS:")
    lines.extend(facts)
    return "\n".join(lines)


def synthesise_routing_state(case):
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


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--benchmark", type=Path,
                   default=PROJECT_ROOT / "configs/benchmark/benchmark_incident_heldout.json")
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--pipeline-model", type=str, default="google/gemini-3.1-flash-lite-preview")
    p.add_argument("--extractor-model", type=str, default="google/gemini-2.5-flash")
    p.add_argument("--reasoning-model", type=str, default="google/gemini-2.5-flash")
    p.add_argument("--cases", nargs="+", default=None)
    p.add_argument("--reforce-output", type=Path,
                   default=PROJECT_ROOT / "external/reforce/output_triage")
    args_cli = p.parse_args()

    bench = json.loads(args_cli.benchmark.read_text())
    cases = bench.get("cases", [])
    if args_cli.cases:
        cases = [c for c in cases if c["id"] in args_cli.cases]
    log.info("Running ReFoRCE on %d cases (model=%s)", len(cases), args_cli.pipeline_model)

    sqlite_path = REFORCE_DATA_BIRD / "dev_databases/triage_disaster/triage_disaster.sqlite"
    output_root = args_cli.reforce_output / args_cli.pipeline_model.replace("/", "_")
    output_root.mkdir(parents=True, exist_ok=True)

    # ReFoRCE's get_sqlite_path uses hardcoded relative paths like
    # ``../../data/BIRD/dev_databases/<db_id>/<db_id>.sqlite``. Cd into
    # methods/ReFoRCE so those resolve correctly.
    os.chdir(str(REFORCE_DIR))

    reforce_args = _make_args(args_cli.pipeline_model, output_root)

    duckdb_path = Path(os.environ.get("DISASTER_DB_PATH",
                                       "data/disaster_ai_db.duckdb"))
    import duckdb as _duckdb
    con = _duckdb.connect(str(duckdb_path), read_only=True)

    extractor_llm = run_benchmark._make_llm(args_cli.extractor_model)
    reasoning_llm = run_benchmark._make_llm(args_cli.reasoning_model)

    results = []
    for i, case in enumerate(cases, 1):
        sql_text = run_reforce_one(case["id"], case["question"], reforce_args,
                                     sqlite_path, output_root)
        if not sql_text:
            routing = synthesise_routing_state(case)
            results.append({
                "id": case["id"], "tier": case.get("tier",""), "category": case.get("category",""),
                "question": case["question"], "system_answer": "", "judge_score": 1,
                "judge_score_raw": 0.0, "fact_score": 0.0, "reasoning_score": 0.0,
                "extracted_claims": {}, "fact_checks_passed": [], "fact_checks_failed": [],
                "reasoning_checks": [], "routing_state": routing,
                "skipped_reason": "ReFoRCE produced no SQL",
            })
            log.info("[%d/%d] %s: SKIPPED", i, len(cases), case["id"])
            continue
        exec_result = execute_against_duckdb(con, sql_text)
        system_answer = render_answer(sql_text, exec_result)
        routing = synthesise_routing_state(case)
        result = score_one(case, system_answer, routing, extractor_llm, reasoning_llm)
        result["reforce_sql"] = sql_text
        results.append(result)
        log.info("[%d/%d] %s judge=%d fact=%.2f reasoning=%.2f", i, len(cases),
                 case["id"], result["judge_score"], result["fact_score"], result["reasoning_score"])

    scores = [r["judge_score"] for r in results]
    fact_scores = [r["fact_score"] for r in results]
    reasoning_scores = [r["reasoning_score"] for r in results]
    output = {
        "ablation": "reforce", "pipeline_model": args_cli.pipeline_model,
        "claim_extractor_model": args_cli.extractor_model,
        "reasoning_judge_model": args_cli.reasoning_model,
        "mean_judge_score": round(sum(scores)/max(len(scores),1), 3),
        "mean_fact_score": round(sum(fact_scores)/max(len(fact_scores),1), 3),
        "mean_reasoning_score": round(sum(reasoning_scores)/max(len(reasoning_scores),1), 3),
        "score_distribution": {str(i): scores.count(i) for i in range(1, 6)},
        "total": len(results), "cases": results,
    }
    args_cli.output.parent.mkdir(parents=True, exist_ok=True)
    args_cli.output.write_text(json.dumps(output, indent=2, default=str))
    log.info("Wrote %s (mean=%.3f / %d cases)", args_cli.output, output["mean_judge_score"], len(results))


if __name__ == "__main__":
    main()
