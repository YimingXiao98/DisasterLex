"""Named-region geographic helper.

Detects when a user query references a named multi-county Texas region
(e.g., "Texas Panhandle", "Permian Basin", "Gulf Coast") and emits the
canonical member-county list so the SQL agent can aggregate over the
correct scope.

This is factual geography, not a query-routing rule. The county lists
are documented in config/regions.json; this module loads them once and
matches a query against the alias list.

A query that does NOT reference a named region returns no hint and the
agent proceeds normally.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List, Optional, Tuple

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
REGIONS_PATH = PROJECT_ROOT / "configs" / "regions.json"


def _load_regions() -> Dict[str, Dict]:
    if not REGIONS_PATH.exists():
        return {}
    with open(REGIONS_PATH) as f:
        data = json.load(f)
    return data.get("regions", {})


REGIONS: Dict[str, Dict] = _load_regions()


def detect_region(user_query: str) -> Optional[Tuple[str, Dict]]:
    """Return (region_id, region_dict) if the query mentions a named region, else None.

    Matches the lowercase query against each region's alias list. First
    longest-alias match wins (e.g., "outside major metros" beats "major").
    """
    if not user_query or not REGIONS:
        return None
    lowered = " ".join(user_query.lower().split())
    matches: List[Tuple[int, str, Dict]] = []
    for region_id, region in REGIONS.items():
        for alias in region.get("aliases", []):
            if alias and alias.lower() in lowered:
                matches.append((len(alias), region_id, region))
                break
    if not matches:
        return None
    matches.sort(key=lambda x: -x[0])
    _, region_id, region = matches[0]
    return region_id, region


def region_hint(user_query: str) -> str:
    """Return a one-paragraph regional-scope hint for the system prompt, or empty string."""
    detected = detect_region(user_query)
    if not detected:
        return ""
    region_id, region = detected
    counties = region.get("counties", [])
    if not counties:
        return ""
    aliases = region.get("aliases", [])
    canonical = aliases[0] if aliases else region_id
    counties_str = ", ".join(f"'{c}'" for c in counties)
    return (
        f"REGIONAL SCOPE — the user query references '{canonical}', "
        f"which is the multi-county region: {counties_str}. "
        "If the question asks for a regional aggregate (SUM, AVG, COUNT across the region), "
        "the SQL must filter on the FULL list of counties above using "
        f"`WHERE x.County IN ({counties_str})` and emit the result as a SINGLE region-level "
        "row. Do not substitute a single county for the regional aggregate, and do not "
        "produce a per-county breakdown unless the question explicitly asks for one."
    )
