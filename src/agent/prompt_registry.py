"""
Prompt Registry — loads and indexes all prompt markdown files from src/prompts/.

Provides structured access to:
  - Cluster definitions (auto-generated from folder names and prompt titles)
  - Query type index (objective, key datasets, core logic per prompt file)

All loading happens at import time so downstream modules get instant access.
"""
from __future__ import annotations

import logging
import re
from pathlib import Path
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
PROMPTS_DIR = PROJECT_ROOT / "src" / "prompts"


# ── Markdown parsing ─────────────────────────────────────────────────────────

def _parse_prompt_file(path: Path) -> Dict[str, Any]:
    """
    Parse a prompt markdown file into structured sections.

    Expected format:
        # Title
        ## Objective
        ...
        ## Key Datasets
        - `TABLE_NAME`
        ...
        ## Core Logic
        1. ...
        2. ...
    """
    text = path.read_text(encoding="utf-8")
    sections: Dict[str, str] = {}

    # Extract title from first H1
    title_match = re.search(r"^#\s+(.+)$", text, re.MULTILINE)
    title = title_match.group(1).strip() if title_match else path.stem.replace("_", " ").title()

    # Split on ## headers
    header_pattern = re.compile(r"^##\s+(.+)$", re.MULTILINE)
    headers = list(header_pattern.finditer(text))

    for i, match in enumerate(headers):
        header_name = match.group(1).strip().lower()
        start = match.end()
        end = headers[i + 1].start() if i + 1 < len(headers) else len(text)
        sections[header_name] = text[start:end].strip()

    # Extract key datasets as a list
    key_datasets: List[str] = []
    datasets_text = sections.get("key datasets", "")
    if datasets_text:
        for line in datasets_text.split("\n"):
            # Match backtick-quoted dataset names or bare list items
            ds_match = re.search(r"`([^`]+)`", line)
            if ds_match:
                key_datasets.append(ds_match.group(1))
            elif line.strip().startswith("- "):
                key_datasets.append(line.strip().lstrip("- ").strip())

    # Core logic
    core_logic = sections.get("core logic", "")

    return {
        "title": title,
        "objective": sections.get("objective", ""),
        "key_datasets": key_datasets,
        "core_logic": core_logic,
        "raw_text": text,
    }


def _folder_name_to_display(folder_name: str) -> str:
    """Convert folder_name_like_this to Title Case display name."""
    return folder_name.replace("_", " ").title()


# ── Registry data structures ─────────────────────────────────────────────────

class ClusterInfo:
    """Metadata for a single prompt cluster (directory)."""

    def __init__(self, folder_name: str, display_name: str):
        self.folder_name = folder_name
        self.display_name = display_name
        self.query_types: Dict[str, Dict[str, Any]] = {}
        self.cluster_description: str = ""

    def add_query_type(self, file_stem: str, parsed: Dict[str, Any]) -> None:
        self.query_types[file_stem] = parsed

    def set_cluster_description(self, text: str) -> None:
        self.cluster_description = text

    @property
    def description(self) -> str:
        """Return cluster description from description.md, or auto-generate."""
        if self.cluster_description:
            return self.cluster_description
        titles = [qt["title"] for qt in self.query_types.values()]
        if not titles:
            return self.display_name
        return f"Covers: {', '.join(titles)}"


# ── Module-level registry (populated at import time) ─────────────────────────

CLUSTERS: Dict[str, ClusterInfo] = {}


def _load_registry() -> None:
    """Scan src/prompts/ and populate the CLUSTERS registry."""
    if not PROMPTS_DIR.exists():
        logger.warning(f"Prompts directory not found: {PROMPTS_DIR}")
        return

    for cluster_dir in sorted(PROMPTS_DIR.iterdir()):
        if not cluster_dir.is_dir():
            continue

        folder_name = cluster_dir.name
        display_name = _folder_name_to_display(folder_name)
        cluster = ClusterInfo(folder_name, display_name)

        for md_file in sorted(cluster_dir.glob("*.md")):
            stem = md_file.stem

            if stem == "description":
                cluster.set_cluster_description(
                    md_file.read_text(encoding="utf-8")
                )
                continue

            # Regular query type prompt
            parsed = _parse_prompt_file(md_file)
            cluster.add_query_type(stem, parsed)

        CLUSTERS[folder_name] = cluster
        logger.info(
            f"Loaded cluster '{folder_name}': "
            f"{len(cluster.query_types)} query types"
        )

    logger.info(f"Prompt registry loaded: {len(CLUSTERS)} clusters")


# Run at import time
_load_registry()


# ── Public API ────────────────────────────────────────────────────────────────

def get_cluster_names() -> List[str]:
    """Return all cluster folder names."""
    return list(CLUSTERS.keys())


def get_cluster_summary() -> str:
    """
    Return a compact text block listing all clusters and their descriptions.
    Used for LLM injection in the cluster classification step.
    """
    lines = []
    for name, cluster in CLUSTERS.items():
        lines.append(f"Cluster: {name}")
        lines.append(f"  Display Name: {cluster.display_name}")
        lines.append(f"  {cluster.description}")
        lines.append("")
    return "\n".join(lines)


def get_query_types_for_cluster(cluster_name: str) -> str:
    """
    Return a text block listing all query types in a cluster with their
    objectives. Used for LLM injection in the query type identification step.
    """
    cluster = CLUSTERS.get(cluster_name)
    if not cluster:
        return f"Unknown cluster: {cluster_name}"

    lines = [f"Query types in cluster '{cluster.display_name}':", ""]
    for stem, qt in cluster.query_types.items():
        lines.append(f"Query Type: {stem}")
        lines.append(f"  Title: {qt['title']}")
        lines.append(f"  Objective: {qt['objective']}")
        lines.append("")
    return "\n".join(lines)


def get_prompt_details(
    cluster_name: str, query_type: str
) -> Optional[Dict[str, Any]]:
    """
    Return the full parsed prompt for a specific cluster + query type.

    Returns dict with keys: title, objective, key_datasets, core_logic, raw_text.
    Returns None if not found.
    """
    cluster = CLUSTERS.get(cluster_name)
    if not cluster:
        return None
    return cluster.query_types.get(query_type)


def get_cluster_description(cluster_name: str) -> str:
    """Return the full cluster description text (from description.md)."""
    cluster = CLUSTERS.get(cluster_name)
    if not cluster:
        return ""
    return cluster.cluster_description
