"""
Load DDCG + EKG into Neo4j.

Usage:
    docker compose up -d          # start Neo4j
    python scripts/load_neo4j.py  # populate the graph

Flags:
    --no-clear   Skip clearing the graph before loading (append mode).
    --stats      Print graph stats and exit (no loading).
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

# Allow running from repo root: `python scripts/load_neo4j.py`
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from dotenv import load_dotenv

load_dotenv()

from src.graph.context_graph import ContextGraph

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(name)-30s  %(levelname)-7s  %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("load_neo4j")

# ── Config paths ─────────────────────────────────────────────────────────────
DDCG_PATH = PROJECT_ROOT / "configs" / "graph" / "ddcg.json"
EKG_PATH = PROJECT_ROOT / "configs" / "graph" / "ekg_curated.json"


def print_stats(graph: ContextGraph) -> None:
    """Pretty-print graph node and relationship counts."""
    stats = graph.get_stats()
    print("\n── Neo4j Graph Stats ──────────────────────────")
    print("  Nodes:")
    for label in ("DataTable", "DataColumn", "JoinRule", "Concept"):
        print(f"    {label:15s}  {stats.get(label, 0):>5d}")
    print("  Relationships:")
    for rel in ("HAS_COLUMN", "JOINABLE_VIA", "MAPS_TO", "INCREASES", "REDUCES", "INDICATES"):
        print(f"    {rel:15s}  {stats.get(rel, 0):>5d}")
    total_nodes = sum(stats.get(l, 0) for l in ("DataTable", "DataColumn", "JoinRule", "Concept"))
    total_rels = sum(
        stats.get(r, 0)
        for r in ("HAS_COLUMN", "JOINABLE_VIA", "MAPS_TO", "INCREASES", "REDUCES", "INDICATES")
    )
    print(f"  {'TOTAL NODES':15s}  {total_nodes:>5d}")
    print(f"  {'TOTAL RELS':15s}  {total_rels:>5d}")
    print("───────────────────────────────────────────────\n")


def main() -> None:
    parser = argparse.ArgumentParser(description="Load DDCG + EKG into Neo4j")
    parser.add_argument(
        "--no-clear",
        action="store_true",
        help="Do not clear the graph before loading (append mode)",
    )
    parser.add_argument(
        "--stats",
        action="store_true",
        help="Print graph stats and exit (no loading)",
    )
    args = parser.parse_args()

    graph = ContextGraph()

    if args.stats:
        print_stats(graph)
        graph.close()
        return

    # ── Clear ─────────────────────────────────────────────────────────────
    if not args.no_clear:
        logger.info("Clearing existing graph...")
        graph.clear_graph()

    # ── Constraints & indexes ─────────────────────────────────────────────
    logger.info("Creating constraints and indexes...")
    graph.create_constraints()
    graph.create_indexes()

    # ── Load DDCG ─────────────────────────────────────────────────────────
    if not DDCG_PATH.exists():
        logger.error(f"DDCG not found at {DDCG_PATH}. Run: python scripts/build_ddcg.py")
        graph.close()
        sys.exit(1)

    logger.info(f"Loading DDCG from {DDCG_PATH}...")
    graph.load_ddcg(DDCG_PATH)

    # ── Load EKG ──────────────────────────────────────────────────────────
    if not EKG_PATH.exists():
        logger.error(f"EKG not found at {EKG_PATH}. Run: python scripts/extract_ekg_tdis.py")
        graph.close()
        sys.exit(1)

    logger.info(f"Loading EKG from {EKG_PATH}...")
    graph.load_ekg(EKG_PATH)

    # ── Verify ────────────────────────────────────────────────────────────
    print_stats(graph)

    logger.info("Neo4j graph loaded successfully.")
    graph.close()


if __name__ == "__main__":
    main()
