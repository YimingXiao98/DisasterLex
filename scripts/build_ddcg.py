#!/usr/bin/env python3
"""
build_ddcg.py — Construct the Disaster Data Catalog Graph from DuckDB.

Usage:
    python scripts/build_ddcg.py                              # defaults
    python scripts/build_ddcg.py --db path/to/db.duckdb       # custom DB
    python scripts/build_ddcg.py --out configs/graph/ddcg.json # custom output

What it does (100 % deterministic — no LLM):
  1. Connects to DuckDB (read-only).
  2. Enumerates every table  → DATA_TABLE nodes.
  3. Enumerates every column → DATA_COLUMN nodes + HAS_COLUMN edges.
  4. Computes per-column null fraction for data-quality awareness.
  5. Generates JOIN_RULE nodes:
       a. PRIMARY_KEY — all tables share `hex_id`.
       b. CROSSWALK   — County / State / Zipcode via crosswalk table.
       c. HEX_AGG     — hex_id_l7, hex_id_l6 roll-ups via crosswalk.
       d. SPATIAL      — tables with a `geometry` column.
  6. Creates JOINABLE_VIA edges (table → join_rule).
  7. Exports the full DDCG to JSON.
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

import duckdb

# Make project root importable
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.graph.ddcg import (
    DDCG,
    ColumnDtype,
    DataColumnNode,
    DataTableNode,
    DDCGEdge,
    JoinRuleNode,
    JoinType,
    _classify_table,
)

logging.basicConfig(level=logging.INFO, format="%(levelname)s  %(message)s")
logger = logging.getLogger(__name__)

# ── defaults ────────────────────────────────────────────────────────────────

import os
DEFAULT_DB = Path(os.environ.get(
    "DISASTER_DB_PATH",
    str(PROJECT_ROOT / "data" / "disaster_ai_db.duckdb"),
))
DEFAULT_OUT = PROJECT_ROOT / "configs" / "graph" / "ddcg.json"
CROSSWALK_TABLE = "hex_county_state_zip_crosswalk"


# ── dtype mapping ──────────────────────────────────────────────────────────

def _map_dtype(duck_type: str) -> ColumnDtype:
    """Map DuckDB type strings to our enum."""
    duck_type = duck_type.upper()
    mapping = {
        "VARCHAR": ColumnDtype.VARCHAR,
        "DOUBLE": ColumnDtype.DOUBLE,
        "BIGINT": ColumnDtype.BIGINT,
        "INTEGER": ColumnDtype.INTEGER,
        "INT": ColumnDtype.INTEGER,
        "FLOAT": ColumnDtype.FLOAT,
        "BOOLEAN": ColumnDtype.BOOLEAN,
        "BOOL": ColumnDtype.BOOLEAN,
        "BLOB": ColumnDtype.BLOB,
    }
    return mapping.get(duck_type, ColumnDtype.OTHER)


def _sql_quote(name: str) -> str:
    """Quote a table name for safe SQL if it contains special chars."""
    if any(ch in name for ch in ("-", " ", ".")):
        return f'"{name}"'
    return name


# ── main builder ────────────────────────────────────────────────────────────

def build_ddcg(db_path: str | Path) -> DDCG:
    """
    Connect to DuckDB and build the full DDCG deterministically.
    """
    db_path = Path(db_path)
    if not db_path.exists():
        raise FileNotFoundError(f"DuckDB not found: {db_path}")

    conn = duckdb.connect(str(db_path), read_only=True)
    ddcg = DDCG()

    # ── 1. enumerate tables ─────────────────────────────────────────────
    tables_raw = conn.execute("SHOW TABLES").fetchall()
    table_names = sorted([t[0] for t in tables_raw])
    logger.info(f"Found {len(table_names)} tables in DuckDB")

    for tname in table_names:
        quoted = _sql_quote(tname)

        # row count
        row_result = conn.execute(f"SELECT COUNT(*) FROM {quoted}").fetchone()
        row_count: int = row_result[0] if row_result else 0

        # columns
        col_info = conn.execute(f"DESCRIBE {quoted}").fetchall()
        all_col_names = [c[0] for c in col_info]
        non_key_cols = [c[0] for c in col_info if c[0] != "hex_id"]
        has_geometry = any(c[0] == "geometry" for c in col_info)

        # ── table node ──────────────────────────────────────────────────
        table_node = DataTableNode(
            id=tname,
            name=tname.replace("-", " ").replace("_", " ").title(),
            category=_classify_table(tname),
            row_count=row_count,
            column_count=len(all_col_names),
            columns=non_key_cols,
            has_geometry=has_geometry,
            sql_quoted_name=quoted,
        )
        ddcg.tables.append(table_node)

        # ── column nodes + HAS_COLUMN edges ─────────────────────────────
        for col_name, col_type, *_ in col_info:
            is_key = col_name == "hex_id"

            # null fraction
            null_frac = 0.0
            if row_count > 0:
                null_result = conn.execute(
                    f"SELECT COUNT(*) FROM {quoted} WHERE {col_name} IS NULL"
                ).fetchone()
                null_count = null_result[0] if null_result else 0
                null_frac = round(null_count / row_count, 4)

            col_node = DataColumnNode(
                id=f"{tname}.{col_name}",
                table_name=tname,
                column_name=col_name,
                dtype=_map_dtype(col_type),
                is_join_key=is_key,
                null_fraction=null_frac,
            )
            ddcg.columns.append(col_node)

            ddcg.edges.append(DDCGEdge(
                source=tname,
                target=col_node.id,
                edge_type="HAS_COLUMN",
            ))

        logger.info(f"  {tname}: {len(all_col_names)} cols, {row_count:,} rows")

    # ── 2. JOIN RULES ───────────────────────────────────────────────────

    # All non-crosswalk tables (they all share hex_id)
    data_tables = [t for t in table_names if t != CROSSWALK_TABLE]

    # 2a. PRIMARY KEY join — hex_id (universal)
    jr_primary = JoinRuleNode(
        id="join__hex_id",
        join_type=JoinType.PRIMARY_KEY,
        join_key="hex_id",
        tables=data_tables,
        sql_template="A.hex_id = B.hex_id",
        description=(
            f"All {len(data_tables)} data tables share hex_id (H3 level-8). "
            "Direct equi-join on hex_id."
        ),
    )
    ddcg.join_rules.append(jr_primary)

    # JOINABLE_VIA edges for primary key
    for tname in data_tables:
        ddcg.edges.append(DDCGEdge(
            source=tname,
            target="join__hex_id",
            edge_type="JOINABLE_VIA",
        ))

    # 2b. CROSSWALK joins — County, State, Zipcode
    if CROSSWALK_TABLE in table_names:
        for geo_key in ("County", "State", "Zipcode"):
            jr_cross = JoinRuleNode(
                id=f"join__crosswalk_{geo_key.lower()}",
                join_type=JoinType.CROSSWALK,
                join_key=geo_key,
                tables=[CROSSWALK_TABLE] + data_tables,
                sql_template=(
                    f"A.hex_id = {CROSSWALK_TABLE}.hex_id "
                    f"AND {CROSSWALK_TABLE}.{geo_key} = '<value>'"
                ),
                description=(
                    f"Filter any data table by {geo_key} through the crosswalk table. "
                    f"Join data table to crosswalk on hex_id, then filter on {geo_key}."
                ),
            )
            ddcg.join_rules.append(jr_cross)

            # edges
            ddcg.edges.append(DDCGEdge(
                source=CROSSWALK_TABLE,
                target=jr_cross.id,
                edge_type="JOINABLE_VIA",
            ))
            for tname in data_tables:
                ddcg.edges.append(DDCGEdge(
                    source=tname,
                    target=jr_cross.id,
                    edge_type="JOINABLE_VIA",
                ))

        # 2c. HEX AGGREGATION — l7, l6
        for hex_level in ("hex_id_l7", "hex_id_l6"):
            level_num = hex_level.split("_l")[1]
            jr_agg = JoinRuleNode(
                id=f"join__hex_agg_l{level_num}",
                join_type=JoinType.HEX_AGGREGATION,
                join_key=hex_level,
                tables=[CROSSWALK_TABLE] + data_tables,
                sql_template=(
                    f"A.hex_id = {CROSSWALK_TABLE}.hex_id "
                    f"GROUP BY {CROSSWALK_TABLE}.{hex_level}"
                ),
                description=(
                    f"Aggregate any data table from H3 level-8 to level-{level_num}. "
                    f"Join to crosswalk on hex_id, then GROUP BY {hex_level}."
                ),
            )
            ddcg.join_rules.append(jr_agg)

            ddcg.edges.append(DDCGEdge(
                source=CROSSWALK_TABLE,
                target=jr_agg.id,
                edge_type="JOINABLE_VIA",
            ))
            for tname in data_tables:
                ddcg.edges.append(DDCGEdge(
                    source=tname,
                    target=jr_agg.id,
                    edge_type="JOINABLE_VIA",
                ))

    # 2d. SPATIAL join — tables with geometry column
    geo_tables = [t.id for t in ddcg.tables if t.has_geometry]
    if geo_tables:
        jr_spatial = JoinRuleNode(
            id="join__spatial_geometry",
            join_type=JoinType.SPATIAL,
            join_key="geometry",
            tables=geo_tables,
            sql_template="ST_Intersects(A.geometry, B.geometry)",
            description=(
                f"Spatial join via geometry column. "
                f"Tables with geometry: {', '.join(geo_tables)}."
            ),
        )
        ddcg.join_rules.append(jr_spatial)

        for tname in geo_tables:
            ddcg.edges.append(DDCGEdge(
                source=tname,
                target="join__spatial_geometry",
                edge_type="JOINABLE_VIA",
            ))

    conn.close()

    # ── summary ─────────────────────────────────────────────────────────
    logger.info("")
    logger.info("=== DDCG Summary ===")
    logger.info(f"  Tables:     {len(ddcg.tables)}")
    logger.info(f"  Columns:    {len(ddcg.columns)}")
    logger.info(f"  Join Rules: {len(ddcg.join_rules)}")
    logger.info(f"  Edges:      {len(ddcg.edges)}")

    return ddcg


# ── CLI ─────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Build the DDCG from DuckDB")
    parser.add_argument("--db", type=str, default=str(DEFAULT_DB),
                        help="Path to DuckDB database")
    parser.add_argument("--out", type=str, default=str(DEFAULT_OUT),
                        help="Output JSON path")
    args = parser.parse_args()

    ddcg = build_ddcg(args.db)
    ddcg.save_json(args.out)

    # Print prompt context preview
    print("\n" + "=" * 60)
    print("PROMPT CONTEXT PREVIEW (first 40 lines):")
    print("=" * 60)
    ctx = ddcg.to_prompt_context()
    for line in ctx.split("\n")[:40]:
        print(line)
    print("...")


if __name__ == "__main__":
    main()
