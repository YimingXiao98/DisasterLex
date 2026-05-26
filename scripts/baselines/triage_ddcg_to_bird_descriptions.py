"""Generate BIRD-format ``database_description/<table>.csv`` files from the DDCG.

CHESS's catalog vector store and Schema Selector consume per-table CSV files
with columns ``original_column_name, column_name, column_description, data_format,
value_description`` (BIRD benchmark format; see
``external/chess/src/database_utils/db_catalog/csv_utils.py``).

The TRIAGE DDCG (in Neo4j) supplies table-level metadata (description, category,
row count) and column dtypes. Column-level prose descriptions don't exist in the
DDCG so we synthesise reasonable BIRD-style hints from naming conventions; this
keeps the comparison fair (CHESS gets the same level of column documentation it
would get on a typical BIRD database).

Usage:
    conda run -n disaster_graph_rag --no-capture-output python -u \\
        scripts/baselines/triage_ddcg_to_bird_descriptions.py
"""
from __future__ import annotations

import argparse
import csv
import logging
import re
from pathlib import Path
from typing import Iterable

from src.graph.context_graph import get_context_graph

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
DEFAULT_OUT = PROJECT_ROOT / "external/chess/data/triage_disaster/database_description"

# Tables that the TRIAGE pipeline blocks; mirror them in the BIRD descriptions
# so the CHESS baseline cannot cheat by reading a description for a forbidden
# table.
FORBIDDEN_TABLE_PATTERNS = [
    re.compile(r"^HP_FLD_003$", re.IGNORECASE),
    re.compile(r"floodgenome", re.IGNORECASE),
]

# Heuristic column-name → BIRD description hints. Each entry is (regex, expanded
# name template, column_description, value_description). Templates use named
# groups from the regex.
COLUMN_HINTS: list[tuple[re.Pattern[str], str, str, str]] = [
    (re.compile(r"^hex_id$"),
     "H3 hexagon identifier",
     "Uber H3 hexagonal grid cell ID at resolution 8 covering the state of Texas",
     "15-character hexadecimal H3 cell identifier"),
    (re.compile(r"^hex_id_l(?P<lvl>\d+)$"),
     "H3 hexagon at resolution {lvl}",
     "Uber H3 hexagonal grid cell ID at resolution {lvl} (coarser than the base level-8 hex_id)",
     "15-character hexadecimal H3 cell identifier at the indicated resolution"),
    (re.compile(r"^County$"),
     "County name",
     "Texas county name including the literal suffix \" County\" (e.g., \"Harris County\")",
     "string with \" County\" suffix"),
    (re.compile(r"^State$"),
     "State name",
     "U.S. state full name; for this dataset always Texas",
     "string"),
    (re.compile(r"^Zipcode$"),
     "ZIP code",
     "5-digit U.S. ZIP code",
     "5-digit string"),
    (re.compile(r"^nri_(?P<haz>\w+)_(?P<metric>value|score|rating)$"),
     "FEMA NRI {haz} {metric}",
     "FEMA National Risk Index ({haz}) {metric}; one of expected annual loss in dollars (value), 0-100 normalized score (score), or hazard-rating string (rating)",
     "value: USD; score: 0-100 (higher = more risk); rating: categorical e.g. Very Low/Low/Relatively Moderate/Relatively High/Very High"),
    (re.compile(r"^nri_eal(?P<suffix>.*)$"),
     "NRI expected annual loss{suffix}",
     "FEMA National Risk Index expected annual loss in dollars{suffix}",
     "USD"),
    (re.compile(r"^nri_cri_(?P<metric>value|score)$"),
     "NRI Community Resilience Index {metric}",
     "FEMA National Risk Index Community Resilience Index {metric}; HIGHER score means MORE resilience",
     "score: 0-100 (higher = more resilient)"),
    (re.compile(r"^sovi$"),
     "Social Vulnerability Index",
     "CDC/ATSDR Social Vulnerability Index value; sentinel value -999 indicates missing data and must be filtered",
     "double; -999 = missing"),
    (re.compile(r"^psvi_score$"),
     "Population Sensitivity / Vulnerability Index score",
     "Population Sensitivity Vulnerability Index, normalized 0-100",
     "0-100"),
    (re.compile(r"^median_income$"),
     "Median household income",
     "U.S. Census median household income in dollars (raw, not inverse)",
     "USD"),
    (re.compile(r"^ve_ae_fraction$"),
     "FEMA VE/AE flood-zone coverage fraction",
     "Fraction of the hex covered by FEMA NFHL Special Flood Hazard Areas in zones VE and AE (1% annual chance flood, with or without wave action)",
     "0.0-1.0"),
    (re.compile(r"^hosp_n$"),
     "Hospital count",
     "Number of HIFLD hospital facilities whose centroid falls inside the hex",
     "non-negative integer"),
    (re.compile(r"^population_per_hex$"),
     "Population per hex (non-overlapping)",
     "Resident population assigned to this exact hex; safe to SUM without overcounting",
     "non-negative integer (count of people)"),
    (re.compile(r"^population_(?P<r>\d+km)$"),
     "Population within {r} buffer",
     "Resident population within a {r} buffer of the hex centroid; SUMming over hexes overcounts because buffers overlap",
     "non-negative integer (count of people; do not SUM across hexes)"),
    (re.compile(r"^hifld_(?P<rest>.+)$"),
     "HIFLD: {rest}",
     "Homeland Infrastructure Foundation-Level Data feature count or attribute: {rest}",
     "depends on column type"),
    (re.compile(r"_n$"),
     "{name} count",
     "Count of records or features for {name} within the hex",
     "non-negative integer"),
    (re.compile(r"_fraction$"),
     "{name} fraction",
     "Fraction of the hex covered by or attributed to {name}",
     "0.0-1.0"),
    (re.compile(r"_score$"),
     "{name} score",
     "Score column; usually 0-100 but check the source-table convention",
     "0-100 (verify per-table)"),
    (re.compile(r"_value$"),
     "{name} value",
     "Raw-value column (often dollar-denominated for NRI columns)",
     "raw numeric"),
]


logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger("ddcg2bird")


def is_forbidden(table_name: str) -> bool:
    return any(p.search(table_name) for p in FORBIDDEN_TABLE_PATTERNS)


def hint_for_column(name: str, dtype: str) -> tuple[str, str, str, str]:
    """Return (original_column_name, expanded_name, column_description, value_description)."""
    for pat, expanded_tpl, desc_tpl, value_tpl in COLUMN_HINTS:
        m = pat.match(name) or pat.search(name)
        if m:
            try:
                groups = m.groupdict()
                groups.setdefault("name", name)
                return (
                    name,
                    expanded_tpl.format(**groups),
                    desc_tpl.format(**groups),
                    value_tpl.format(**groups) if "{" in value_tpl else value_tpl,
                )
            except (KeyError, IndexError):
                continue
    # Fallback: just describe by dtype
    return (name, name.replace("_", " "), f"Column of type {dtype}", "")


def write_table_csv(
    out_dir: Path,
    table_name: str,
    table_description: str,
    columns: list[dict],
) -> None:
    out_path = out_dir / f"{table_name}.csv"
    with out_path.open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow([
            "original_column_name",
            "column_name",
            "column_description",
            "data_format",
            "value_description",
        ])
        for c in columns:
            name = c["column_name"]
            dtype = c.get("dtype", "")
            orig, expanded, desc, value = hint_for_column(name, dtype)
            # Prepend table description to column 0 only on the hex_id row, BIRD-style
            if name == "hex_id" and table_description:
                desc = f"{table_description.strip()} -- {desc}"
            w.writerow([orig, expanded, desc, dtype, value])
    log.info("WROTE %s (%d columns)", out_path.name, len(columns))


def main(argv: Iterable[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT,
                        help="Destination directory for per-table BIRD CSVs")
    args = parser.parse_args(argv)

    args.out.mkdir(parents=True, exist_ok=True)

    g = get_context_graph()
    schema = g.retrieve_full_schema()

    written = 0
    skipped = []
    for entry in schema:
        # retrieve_full_schema returns dicts with keys including table_id, columns, table_description.
        table_name = entry.get("table_id") or entry.get("sql_quoted_name", "").strip('"')
        if not table_name:
            continue
        if is_forbidden(table_name):
            skipped.append(table_name)
            continue
        # DDCG's retrieve_full_schema omits join keys (hex_id, hex_id_l6/l7).
        # Re-inject them so CHESS's Schema Selector sees the PK and can reason
        # about joins. Crosswalk table is the only one whose join keys are
        # already enumerated as data columns, so leave it alone.
        columns = list(entry.get("columns", []))
        if table_name != "hex_county_state_zip_crosswalk":
            columns = [{"column_name": "hex_id", "dtype": "VARCHAR"}] + columns
        write_table_csv(
            args.out,
            table_name,
            entry.get("table_description") or f"{entry.get('category', '')} table".strip(),
            columns,
        )
        written += 1

    log.info("Done. Wrote %d CSVs to %s. Skipped: %s",
             written, args.out, ",".join(skipped) or "none")


if __name__ == "__main__":
    main()
