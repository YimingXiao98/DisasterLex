"""Convert a TRIAGE benchmark JSON into the BIRD-style format CHESS consumes.

CHESS's ``src/main.py`` calls ``json.load`` on the dataset path and expects a
list of dicts with at least ``question_id``, ``db_id``, ``question``. The TRIAGE
benchmark is a dict containing a ``cases`` list with richer fields. This
converter strips it down to the BIRD essentials and writes an output file under
``external/chess/data/`` so CHESS can pick it up via ``--data_path``.

Usage:
    conda run -n disaster_graph_rag --no-capture-output env PYTHONPATH=. python -u \\
        scripts/baselines/triage_to_bird_format.py
"""
from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path
from typing import Iterable

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
DEFAULT_INPUT = PROJECT_ROOT / "config/benchmark/benchmark_incident_heldout.json"
DEFAULT_OUTPUT = PROJECT_ROOT / "external/chess/data/triage_disaster/triage_heldout.json"
DEFAULT_DB_ID = "triage_disaster"

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger("triage2bird")


def convert(triage_path: Path, bird_path: Path, db_id: str) -> None:
    triage = json.loads(triage_path.read_text())
    cases = triage.get("cases", [])
    bird_records = []
    for idx, c in enumerate(cases):
        record = {
            "question_id": idx,
            "db_id": db_id,
            "question": c["question"],
            # CHESS optionally consumes "evidence" as an extra hint. We don't
            # expose it: we want CHESS to solve the query unaided, and we score
            # with the TRIAGE judge, not CHESS's execution-accuracy metric.
            "evidence": "",
            # CHESS's runner crashes if SQL is missing because compare_sqls
            # passes None into sqlite3.execute. Provide a harmless stub query
            # that still parses; CHESS will execute it for the comparison and
            # see it doesn't match its candidate, which is fine — we discard
            # CHESS's gold-comparison metric and score via the TRIAGE judge.
            "SQL": "SELECT 1",
            "difficulty": "moderate",
            # Stash the original case id so we can join CHESS outputs back to
            # TRIAGE cases when scoring.
            "triage_case_id": c["id"],
            "triage_tier": c.get("tier", ""),
            "triage_set": c.get("set", ""),
        }
        bird_records.append(record)

    bird_path.parent.mkdir(parents=True, exist_ok=True)
    bird_path.write_text(json.dumps(bird_records, indent=2))
    log.info("Wrote %d cases to %s", len(bird_records), bird_path)


def main(argv: Iterable[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--db-id", type=str, default=DEFAULT_DB_ID)
    args = parser.parse_args(argv)
    convert(args.input, args.output, args.db_id)


if __name__ == "__main__":
    main()
