"""
validate_gold_facts.py — Re-execute evidence SQL queries from the benchmark JSON
and verify that stored gold_facts match the live DuckDB results.

Usage:
    python scripts/validate_gold_facts.py
    python scripts/validate_gold_facts.py --cases draft_r12 draft_k13
    python scripts/validate_gold_facts.py --benchmark configs/benchmark/dev.json

Exit code 1 if any case has at least one FAIL; 0 otherwise.

Schema of evidence_queries (list of objects per case):
    [{"id": "metric_name", "engine": "duckdb", "purpose": "...", "query": "SELECT ..."}]

Schema of gold_facts (dict keyed by evidence_query id):
    {"metric_name": 1270, ...}

Tolerances:
    - integer / count values: ±5%
    - float values: ±10%
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
logger = logging.getLogger(__name__)

import os

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DB_PATH = Path(os.environ.get(
    "DISASTER_DB_PATH",
    str(PROJECT_ROOT / "data" / "disaster_ai_db.duckdb"),
))
DEFAULT_BENCHMARK = PROJECT_ROOT / "configs" / "benchmark" / "dev.json"

INT_TOLERANCE = 0.05   # ±5% for integer/count values
FLOAT_TOLERANCE = 0.10  # ±10% for float values


def _within_tolerance(actual: float, expected: float, tol: float) -> bool:
    """Return True if actual is within ±tol fraction of expected."""
    if expected == 0:
        return actual == 0
    return abs(actual - expected) / abs(expected) <= tol


def validate_case(conn: object, case: dict) -> tuple[int, int]:
    """Validate all evidence_queries in a single case against gold_facts.

    Returns:
        (pass_count, fail_count) for this case.
    """
    cid = case["id"]
    evidence_queries: list[dict] | None = case.get("evidence_queries")
    gold_facts: dict = case.get("gold_facts", {})

    if not evidence_queries:
        logger.warning("[%s] No evidence_queries field — skipping", cid)
        return 0, 0

    passed = 0
    failed = 0

    for eq in evidence_queries:
        metric_id = eq["id"]
        engine = eq.get("engine", "duckdb")
        query = eq.get("query", "")

        if engine != "duckdb":
            logger.warning("[%s] %s: engine=%s not supported — skipping", cid, metric_id, engine)
            continue

        if metric_id not in gold_facts:
            logger.warning("[%s] %s: no gold_facts entry — skipping", cid, metric_id)
            continue

        expected = gold_facts[metric_id]

        try:
            rows = conn.execute(query).fetchall()
            if not rows or rows[0][0] is None:
                actual = None
            else:
                actual = rows[0][0]
        except Exception as exc:
            logger.error("[%s] %s: SQL ERROR — %s", cid, metric_id, exc)
            logger.error("  Query: %s", query)
            print(f"  FAIL  {cid}/{metric_id}: SQL error — {exc}")
            failed += 1
            continue

        if actual is None:
            print(f"  FAIL  {cid}/{metric_id}: got NULL, expected {expected}")
            failed += 1
            continue

        # Choose tolerance based on type of expected value
        if isinstance(expected, int):
            tol = INT_TOLERANCE
        else:
            tol = FLOAT_TOLERANCE

        ok = _within_tolerance(float(actual), float(expected), tol)
        pct_diff = (
            f"{(float(actual) - float(expected)) / float(expected) * 100:+.1f}%"
            if expected != 0 else "N/A"
        )
        if ok:
            print(f"  PASS  {cid}/{metric_id}: {actual} (expected {expected}, diff {pct_diff})")
            passed += 1
        else:
            print(f"  FAIL  {cid}/{metric_id}: {actual} (expected {expected}, diff {pct_diff}, tol ±{tol:.0%})")
            failed += 1

    return passed, failed


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Validate gold_facts in benchmark JSON against live DuckDB results."
    )
    parser.add_argument(
        "--benchmark",
        type=str,
        default=str(DEFAULT_BENCHMARK),
        help="Path to benchmark JSON (default: configs/benchmark/dev.json)",
    )
    parser.add_argument(
        "--db",
        type=str,
        default=str(DB_PATH),
        help="Path to DuckDB file (default: $DISASTER_DB_PATH or data/disaster_ai_db.duckdb)",
    )
    parser.add_argument(
        "--cases",
        nargs="+",
        metavar="CASE_ID",
        default=None,
        help="Validate only specific case IDs (default: all)",
    )
    args = parser.parse_args()

    benchmark_path = Path(args.benchmark)
    db_path = Path(args.db)

    if not benchmark_path.exists():
        logger.error("Benchmark file not found: %s", benchmark_path)
        sys.exit(1)
    if not db_path.exists():
        logger.error("DuckDB file not found: %s", db_path)
        sys.exit(1)

    with open(benchmark_path) as f:
        benchmark = json.load(f)

    cases: list[dict] = benchmark.get("cases", [])
    if args.cases:
        cases = [c for c in cases if c["id"] in args.cases]
        if not cases:
            logger.error("No cases matched: %s", args.cases)
            sys.exit(1)

    try:
        import duckdb
    except ImportError:
        logger.error("duckdb package not installed in current environment")
        sys.exit(1)

    conn = duckdb.connect(str(db_path), read_only=True)

    total_passed = 0
    total_failed = 0
    cases_with_failures: list[str] = []
    cases_skipped: list[str] = []

    print(f"\nValidating gold_facts for {len(cases)} case(s) from {benchmark_path.name}\n")
    print("=" * 60)

    for case in cases:
        cid = case["id"]
        if "DISABLED" in case:
            logger.info("[%s] DISABLED — skipping", cid)
            continue

        if not case.get("evidence_queries"):
            cases_skipped.append(cid)
            continue

        print(f"\n[{cid}]")
        p, f = validate_case(conn, case)
        total_passed += p
        total_failed += f
        if f > 0:
            cases_with_failures.append(cid)

    conn.close()

    print("\n" + "=" * 60)
    print(f"SUMMARY: {total_passed} PASS  |  {total_failed} FAIL")
    if cases_skipped:
        print(f"Skipped (no evidence_queries): {', '.join(cases_skipped)}")
    if cases_with_failures:
        print(f"Cases with failures: {', '.join(cases_with_failures)}")
    else:
        print("All validated metrics passed.")
    print("=" * 60)

    if total_failed > 0:
        sys.exit(1)


if __name__ == "__main__":
    main()
