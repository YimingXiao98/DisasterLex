"""Export the TRIAGE DuckDB database to a SQLite snapshot for the CHESS baseline.

CHESS expects SQLite (`runner/database_manager.py` and `database_utils/execution.py`
both use ``sqlite3`` directly). This script materialises the 36 non-blocked TRIAGE
tables into a single SQLite file at ``external/chess/data/<db_id>/<db_id>.sqlite``.

Usage:
    conda run -n disaster_graph_rag --no-capture-output python -u \\
        scripts/baselines/duckdb_to_sqlite.py
"""
from __future__ import annotations

import argparse
import logging
import re
import sqlite3
import sys
import time
from pathlib import Path
from typing import Iterable

import duckdb

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
DEFAULT_DUCKDB = Path("/data4/disaster_ai_sp26/disaster_ai_db.duckdb")
DEFAULT_SQLITE = PROJECT_ROOT / "external/chess/data/triage_disaster/triage_disaster.sqlite"

# Tables blocked from the CHESS export — must match the FORBIDDEN_SCHEMA_PATTERNS
# regex in src/agent/text_to_sql_agent.py so the baseline plays under the same
# data-access constraints as the full pipeline.
FORBIDDEN_TABLE_PATTERNS = [
    re.compile(r"^HP_FLD_003$", re.IGNORECASE),
    re.compile(r"floodgenome", re.IGNORECASE),
]

DUCKDB_TO_SQLITE = {
    "BIGINT": "INTEGER",
    "INTEGER": "INTEGER",
    "DOUBLE": "REAL",
    "FLOAT": "REAL",
    "VARCHAR": "TEXT",
    "TEXT": "TEXT",
    "BLOB": "BLOB",
}

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger("dd2sqlite")


def map_type(duckdb_type: str) -> str:
    base = duckdb_type.upper().split("(")[0].strip()
    if base in DUCKDB_TO_SQLITE:
        return DUCKDB_TO_SQLITE[base]
    raise ValueError(f"Unmapped DuckDB type: {duckdb_type}")


def is_forbidden(table_name: str) -> bool:
    return any(p.search(table_name) for p in FORBIDDEN_TABLE_PATTERNS)


def list_tables(con: duckdb.DuckDBPyConnection) -> list[str]:
    rows = con.execute(
        "SELECT table_name FROM information_schema.tables "
        "WHERE table_schema='main' ORDER BY table_name"
    ).fetchall()
    return [r[0] for r in rows]


def describe_columns(con: duckdb.DuckDBPyConnection, table: str) -> list[tuple[str, str]]:
    rows = con.execute(f'DESCRIBE "{table}"').fetchall()
    return [(r[0], r[1]) for r in rows]


def make_create_table(table: str, columns: list[tuple[str, str]]) -> str:
    col_defs = []
    for name, dtype in columns:
        sqlite_type = map_type(dtype)
        if name == "hex_id":
            col_defs.append(f'"{name}" {sqlite_type} PRIMARY KEY')
        else:
            col_defs.append(f'"{name}" {sqlite_type}')
    return f'CREATE TABLE "{table}" ({", ".join(col_defs)})'


def copy_table(
    duckdb_con: duckdb.DuckDBPyConnection,
    sqlite_con: sqlite3.Connection,
    table: str,
    columns: list[tuple[str, str]],
    batch_size: int,
) -> int:
    col_names = [c[0] for c in columns]
    col_list = ", ".join(f'"{c}"' for c in col_names)
    placeholders = ", ".join(["?"] * len(col_names))
    insert_sql = f'INSERT INTO "{table}" ({col_list}) VALUES ({placeholders})'

    cursor = sqlite_con.cursor()
    total = 0
    select = duckdb_con.execute(f'SELECT {col_list} FROM "{table}"')
    while True:
        batch = select.fetchmany(batch_size)
        if not batch:
            break
        cursor.executemany(insert_sql, batch)
        total += len(batch)
    sqlite_con.commit()
    return total


def export(duckdb_path: Path, sqlite_path: Path, batch_size: int) -> None:
    if not duckdb_path.exists():
        log.error("DuckDB file not found: %s", duckdb_path)
        sys.exit(1)
    sqlite_path.parent.mkdir(parents=True, exist_ok=True)
    if sqlite_path.exists():
        log.info("Removing existing SQLite snapshot at %s", sqlite_path)
        sqlite_path.unlink()

    duckdb_con = duckdb.connect(str(duckdb_path), read_only=True)
    sqlite_con = sqlite3.connect(sqlite_path)
    sqlite_con.execute("PRAGMA journal_mode = OFF")
    sqlite_con.execute("PRAGMA synchronous = OFF")

    all_tables = list_tables(duckdb_con)
    exported = []
    skipped = []

    for table in all_tables:
        if is_forbidden(table):
            log.info("SKIP (forbidden) %s", table)
            skipped.append(table)
            continue
        columns = describe_columns(duckdb_con, table)
        log.info("EXPORT %s (%d columns)", table, len(columns))
        sqlite_con.execute(make_create_table(table, columns))
        t0 = time.time()
        n = copy_table(duckdb_con, sqlite_con, table, columns, batch_size)
        log.info("  → %d rows in %.1fs", n, time.time() - t0)
        exported.append((table, n))

    sqlite_con.execute("VACUUM")
    sqlite_con.close()
    duckdb_con.close()

    log.info("Done. Exported %d tables (%s skipped). Output: %s",
             len(exported), ",".join(skipped) or "none", sqlite_path)
    log.info("Total rows: %d", sum(n for _, n in exported))


def main(argv: Iterable[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--duckdb", type=Path, default=DEFAULT_DUCKDB,
                        help="Source DuckDB file")
    parser.add_argument("--sqlite", type=Path, default=DEFAULT_SQLITE,
                        help="Destination SQLite file")
    parser.add_argument("--batch-size", type=int, default=10000)
    args = parser.parse_args(argv)
    export(args.duckdb, args.sqlite, args.batch_size)


if __name__ == "__main__":
    main()
