"""
Disaster Data Catalog Graph (DDCG) — Pydantic schema + construction logic.

The DDCG encodes the physical database schema so that downstream agents
(Text-to-SQL, reasoning) can discover tables, columns, and join paths
without hard-coded knowledge.

Node types
----------
DATA_TABLE   – one per DuckDB table
DATA_COLUMN  – one per non-key column
JOIN_RULE    – encodes a join path between two or more tables

Edge types
----------
HAS_COLUMN   – DATA_TABLE  → DATA_COLUMN
JOINABLE_VIA – DATA_TABLE  → JOIN_RULE  (bidirectional implicit)
"""

from __future__ import annotations

import json
import logging
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)


# ── enums ───────────────────────────────────────────────────────────────────

class ColumnDtype(str, Enum):
    VARCHAR = "VARCHAR"
    DOUBLE = "DOUBLE"
    BIGINT = "BIGINT"
    FLOAT = "FLOAT"
    BLOB = "BLOB"
    INTEGER = "INTEGER"
    BOOLEAN = "BOOLEAN"
    OTHER = "OTHER"


class JoinType(str, Enum):
    """How two tables can be joined."""
    PRIMARY_KEY = "primary_key"       # shared hex_id
    CROSSWALK = "crosswalk"           # via crosswalk table (county/state/zip)
    HEX_AGGREGATION = "hex_agg"       # l8 → l7 / l6 roll-up
    SPATIAL = "spatial"               # geometry-based


class TableCategory(str, Enum):
    """Semantic grouping derived from table-name prefix."""
    CRITICAL_LIFE = "critical_life"
    COMMUNITY_RESILIENCE = "community_resilience"
    EXPOSURE_BUILDING = "exposure_building"
    EXPOSURE_INFRASTRUCTURE = "exposure_infrastructure"
    EXPOSURE_LIFE = "exposure_life"
    EXPOSURE_POPULATION = "exposure_population"
    HAZARD_FLOOD = "hazard_flood"
    HAZARD_HURRICANE = "hazard_hurricane"
    HAZARD_TORNADO = "hazard_tornado"
    HAZARD_WILDFIRE = "hazard_wildfire"
    HIFLD_EMERGENCY = "hifld_emergency"
    HIFLD_ENERGY = "hifld_energy"
    HIFLD_HEALTH = "hifld_health"
    HIFLD_TRANSPORT = "hifld_transport"
    HIFLD_WATER = "hifld_water"
    VULNERABILITY = "vulnerability"
    TRANSPORTATION = "transportation"
    CROSSWALK = "crosswalk"
    OTHER = "other"


# ── prefix → category mapping ──────────────────────────────────────────────

_PREFIX_TO_CATEGORY: List[tuple[str, TableCategory]] = [
    ("CRIT_LIFE",              TableCategory.CRITICAL_LIFE),
    ("CR_",                    TableCategory.COMMUNITY_RESILIENCE),
    ("EX_BLD",                 TableCategory.EXPOSURE_BUILDING),
    ("EX_INF",                 TableCategory.EXPOSURE_INFRASTRUCTURE),
    ("EX_LIFE",                TableCategory.EXPOSURE_LIFE),
    ("EX_POP",                 TableCategory.EXPOSURE_POPULATION),
    ("HP_FLD",                 TableCategory.HAZARD_FLOOD),
    ("HP_HUR",                 TableCategory.HAZARD_HURRICANE),
    ("HP_TOR",                 TableCategory.HAZARD_TORNADO),
    ("crown_fire",             TableCategory.HAZARD_WILDFIRE),
    ("HIFLD-EMERGENC",         TableCategory.HIFLD_EMERGENCY),
    ("HIFLD-ENERGY",           TableCategory.HIFLD_ENERGY),
    ("HIFLD-HEALTH",           TableCategory.HIFLD_HEALTH),
    ("HIFLD-TRANSP",           TableCategory.HIFLD_TRANSPORT),
    ("HIFLD-WATER",            TableCategory.HIFLD_WATER),
    ("VUL_",                   TableCategory.VULNERABILITY),
    ("TRANS-ROAD",             TableCategory.TRANSPORTATION),
    ("hex_county_state_zip",   TableCategory.CROSSWALK),
]


def _classify_table(name: str) -> TableCategory:
    for prefix, cat in _PREFIX_TO_CATEGORY:
        if name.startswith(prefix):
            return cat
    return TableCategory.OTHER


# ── node models ─────────────────────────────────────────────────────────────

class DataColumnNode(BaseModel):
    """A single column inside a DuckDB table."""
    id: str                     # "{table_name}.{column_name}"
    table_name: str
    column_name: str
    dtype: ColumnDtype
    is_join_key: bool = False
    null_fraction: Optional[float] = None   # 0.0-1.0
    description: str = ""

    class Config:
        use_enum_values = True


class DataTableNode(BaseModel):
    """A DuckDB table (one parquet file loaded into DuckDB)."""
    id: str                     # table name
    name: str                   # human-friendly label
    category: TableCategory
    row_count: int
    column_count: int           # including hex_id
    columns: List[str]          # column names (excluding hex_id)
    has_geometry: bool = False
    sql_quoted_name: str        # name safe for SQL  (e.g. '"HIFLD-EMERGENC-FIRE_EMS-N"')
    description: str = ""

    class Config:
        use_enum_values = True


class JoinRuleNode(BaseModel):
    """Describes how two or more tables can be joined."""
    id: str                     # e.g. "join__hex_id" or "join__crosswalk_county"
    join_type: JoinType
    join_key: str               # column name used (hex_id, County, etc.)
    tables: List[str]           # table names participating
    sql_template: str           # e.g. 'A.hex_id = B.hex_id'
    description: str = ""

    class Config:
        use_enum_values = True


# ── edge models ─────────────────────────────────────────────────────────────

class DDCGEdge(BaseModel):
    source: str
    target: str
    edge_type: str              # HAS_COLUMN | JOINABLE_VIA
    properties: Dict[str, Any] = Field(default_factory=dict)


# ── the full DDCG ──────────────────────────────────────────────────────────

class DDCG(BaseModel):
    """Disaster Data Catalog Graph — complete serialisable graph."""
    tables: List[DataTableNode] = Field(default_factory=list)
    columns: List[DataColumnNode] = Field(default_factory=list)
    join_rules: List[JoinRuleNode] = Field(default_factory=list)
    edges: List[DDCGEdge] = Field(default_factory=list)

    # ── helpers ──────────────────────────────────────────────────────────

    @property
    def table_ids(self) -> List[str]:
        return [t.id for t in self.tables]

    def get_table(self, table_id: str) -> Optional[DataTableNode]:
        for t in self.tables:
            if t.id == table_id:
                return t
        return None

    def get_columns_for_table(self, table_id: str) -> List[DataColumnNode]:
        return [c for c in self.columns if c.table_name == table_id]

    def get_join_rules_for_table(self, table_id: str) -> List[JoinRuleNode]:
        return [j for j in self.join_rules if table_id in j.tables]

    def tables_by_category(self, cat: TableCategory) -> List[DataTableNode]:
        cat_val = cat.value if isinstance(cat, TableCategory) else cat
        return [t for t in self.tables if t.category == cat_val]

    # ── serialisation ────────────────────────────────────────────────────

    def to_dict(self) -> dict:
        return {
            "tables": [t.model_dump() for t in self.tables],
            "columns": [c.model_dump() for c in self.columns],
            "join_rules": [j.model_dump() for j in self.join_rules],
            "edges": [e.model_dump() for e in self.edges],
        }

    def save_json(self, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w") as f:
            json.dump(self.to_dict(), f, indent=2)
        logger.info(f"DDCG saved → {path}  "
                     f"({len(self.tables)} tables, {len(self.columns)} columns, "
                     f"{len(self.join_rules)} join rules, {len(self.edges)} edges)")

    @classmethod
    def load_json(cls, path: str | Path) -> "DDCG":
        with open(path) as f:
            data = json.load(f)
        ddcg = cls(
            tables=[DataTableNode(**t) for t in data["tables"]],
            columns=[DataColumnNode(**c) for c in data["columns"]],
            join_rules=[JoinRuleNode(**j) for j in data["join_rules"]],
            edges=[DDCGEdge(**e) for e in data["edges"]],
        )
        logger.info(f"DDCG loaded ← {path}  "
                     f"({len(ddcg.tables)} tables, {len(ddcg.columns)} columns, "
                     f"{len(ddcg.join_rules)} join rules, {len(ddcg.edges)} edges)")
        return ddcg

    # ── summary for LLM context injection ────────────────────────────────

    def to_prompt_context(self, categories: Optional[List[str]] = None) -> str:
        """
        Render a compact text description of the catalog suitable for
        injection into an LLM system prompt (e.g. Text-to-SQL agent).

        Parameters
        ----------
        categories : optional list of TableCategory values to filter on.
        """
        lines: list[str] = ["=== Data Catalog (DDCG) ===", ""]

        target_tables = self.tables
        if categories:
            target_tables = [t for t in self.tables if t.category in categories]

        for tbl in target_tables:
            cols = self.get_columns_for_table(tbl.id)
            # Exclude hex_id from listed columns (it's always PK)
            data_cols = [c for c in cols if not c.is_join_key]
            col_strs = [f"{c.column_name} ({c.dtype})" for c in data_cols]
            lines.append(f"TABLE {tbl.sql_quoted_name}  [{tbl.category}]")
            lines.append(f"  rows: {tbl.row_count:,}")
            lines.append(f"  columns: hex_id (VARCHAR, PK), {', '.join(col_strs)}")
            if tbl.description:
                lines.append(f"  note: {tbl.description}")
            # Warn about high-null columns (>50%)
            high_null = [c for c in data_cols if c.null_fraction and c.null_fraction > 0.5]
            if high_null:
                warnings = [f"{c.column_name} ({c.null_fraction:.0%} null)" for c in high_null]
                lines.append(f"  WARNING sparse: {', '.join(warnings)}")
            lines.append("")

        # Join rules summary
        lines.append("--- Join Rules ---")
        for jr in self.join_rules:
            lines.append(f"  {jr.id}: {jr.sql_template}  ({jr.description})")

        return "\n".join(lines)
