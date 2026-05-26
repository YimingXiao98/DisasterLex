"""
Neo4j Context Graph — unified DDCG + EKG graph manager.

Merges the data catalog (DDCG: tables, columns, joins) and the expert
knowledge graph (EKG: causal rules) into a single Neo4j graph.

Node labels
-----------
  DataTable   — one per DuckDB table  (36 nodes)
  DataColumn  — one per non-key column  (145 nodes)
  JoinRule    — join path between tables  (7 nodes)
  Concept     — EKG causal concept  (14 nodes: features, intermediates, outcomes)

Relationship types
------------------
  HAS_COLUMN   — DataTable → DataColumn
  JOINABLE_VIA — DataTable → JoinRule
  MAPS_TO      — Concept  → DataTable  (semantic link: concept ↔ data)
  INCREASES    — Concept  → Concept  (causal: source INCREASES target)
  REDUCES      — Concept  → Concept  (causal: source REDUCES target)
  INDICATES    — Concept  → Concept  (causal: source INDICATES target)

Usage
-----
    graph = ContextGraph()
    graph.load_ddcg("configs/graph/ddcg.json")
    graph.load_ekg("configs/graph/ekg_curated.json")
    graph.close()
"""

from __future__ import annotations

import json
import logging
import os
import re
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from dotenv import load_dotenv
from neo4j import GraphDatabase

load_dotenv()
logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent


# ── Concept → DataTable mappings ─────────────────────────────────────────────
# Maps EKG concept IDs to the DuckDB table(s) that contain relevant data.
# This creates the MAPS_TO edges that bridge causal knowledge to actual data.

CONCEPT_TABLE_MAP: Dict[str, List[str]] = {
    # input features → data tables
    "elevation":        ["HP_FLD_002"],
    "hand":             ["HP_FLD_002"],
    "impervious":       ["HP_FLD_003"],
    "stream_dist":      ["HP_FLD_003"],
    "rainfall":         ["HP_FLD_002"],
    "foundation":       ["EX_BLD_001"],
    "far":              ["EX_BLD_001"],
    "claims":           ["HP_FLD_002"],
    "roughness":        ["HP_FLD_003"],
    # intermediate concepts — no direct table mapping
    "runoff":           [],
    "drainage":         [],
    "vulnerability":    ["VUL_002", "VUL_004"],
    # outcome concepts
    "flood_occurrence": ["HP_FLD_002", "HP_FLD_003"],
    "flood_severity":   ["HP_FLD_002"],
    # new hazard mappings
    "geography":        ["hex_county_state_zip_crosswalk"],
    "hurricane":        ["HP_HUR_001", "HP_HUR_002"],
    "wildfire":         ["crown_fire_probability", "HP_WFIR_001"],
    "earthquake":       [],
    "tornado":          ["HP_TOR_001"],
    "severe_weather":   [],
    "community_resilience": ["CR_001"],
    "shelters":         ["HIFLD-EMERGENC-SHELTER-N"],
    "population":       ["EX_POP_001"],
    "hospitals":        ["EX_LIFE_004"],
    "local_eoc":        ["HIFLD-EMERGENC-LOCAL_EOC-N"],
    "state_eoc":        ["HIFLD-EMERGENC-STATE_EOC-N"],
    "substations":      ["HIFLD-ENERGY-SUBSTN-N"],
    "power_plants":     ["HIFLD-ENERGY-PLANTS-N"],
    "power_infrastructure": ["HIFLD-ENERGY-SUBSTN-N", "HIFLD-ENERGY-PLANTS-N"],
    "water_infrastructure": ["HIFLD-WATER-WTP-N"],
    "emergency_infrastructure": [
        "HIFLD-EMERGENC-SHELTER-N",
        "EX_LIFE_004",
        "HIFLD-EMERGENC-LOCAL_EOC-N",
        "HIFLD-EMERGENC-STATE_EOC-N",
        "HIFLD-EMERGENC-DEPENDENT_CA",   # missing — triggers data availability warning
    ],
    "power_disruption": ["ER_POW_001"],          # missing — triggers data availability warning
    "hospital_operations": [],
    "shelter_operations": [],
    "emergency_response_capacity": [],
    # New coastal/compound flood features (2026-04-15)
    "storm_surge":      ["HP_HUR_002"],   # ve_ae_fraction captures VE/AE surge zones
    "sea_level_rise":   [],                # no direct DataTable (long-term projection)
    "land_subsidence":  [],                # no direct DataTable
    "wetland_restoration": [],             # intervention concept — no DataTable
    # ── Phase A additions (2026-05-08): close MAPS_TO gaps for existing concepts ──
    "road_segment_criticality": ["TRANS-ROAD-CRIT-INDEX"],
    "low_income":               ["VUL_001"],
    "floodplain":               ["HP_FLD_002", "HP_HUR_002"],
    "floodplain_location":      ["HP_FLD_002", "HP_HUR_002"],
    "sfha_designation":         ["HP_HUR_002"],
    "inundation":               ["HP_HUR_002", "HP_FLD_002"],
    "population_density":       ["EX_POP_001"],
    "physical_vulnerability":   ["VUL_002", "VUL_004"],
    # New concepts added in EKG curated graph for missing data-grounded entities:
    "expected_annual_loss":     ["VUL_003"],
    "buildings":                ["EX_BLD_001", "EX_BLD_002"],
    "fire_ems":                 ["HIFLD-EMERGENC-FIRE_EMS-N"],
    "transmission_lines":       ["HIFLD-ENERGY-TXKM-230P"],
    "primary_roads":            ["HIFLD-TRANSP-PRIMARY_RD-L"],
    "local_law_enforcement":    ["HIFLD-EMERGENC-LOCAL_LAW-N"],
    "infrastructure_density":   ["EX_INF_001"],
    "lifeline_access":          ["EX_LIFE_001", "EX_LIFE_002", "EX_LIFE_003", "EX_LIFE_004"],
}

# ── Data availability catalog ────────────────────────────────────────────────
# Tables that exist in the codebase but are unreachable by the agent at runtime.
# BLOCKED = present in DuckDB but excluded by policy / IP / schema mismatch.
# MISSING = referenced in concept mappings but never loaded into the database.

BLOCKED_TABLES: frozenset[str] = frozenset({
    "HP_FLD_003",           # Resolytics IP — physically in DuckDB but agent-blocked
})

MISSING_TABLES: frozenset[str] = frozenset({
    "ER_POW_001",                   # Power outage footprint — not loaded
    "HIFLD-EMERGENC-DEPENDENT_CA",  # Dependent care facilities — not loaded
})

UNAVAILABLE_TABLES: frozenset[str] = BLOCKED_TABLES | MISSING_TABLES

# Data quality warnings: tables that are available but have known data quality issues.
# Keyed by table ID; value is the warning text to prepend to schema context.
DATA_QUALITY_WARNINGS: Dict[str, str] = {
    "VUL_002": (
        "VUL_002 (SoVI) has 1,665 hexes statewide where sovi = -999 (missing data). "
        "REQUIRED: Run a separate query to count sovi=-999 rows in the analysis county "
        "(e.g., SELECT COUNT(*) FROM VUL_002 v JOIN hex_county_state_zip_crosswalk x "
        "ON v.hex_id=x.hex_id WHERE x.County='<County>' AND v.sovi=-999) and explicitly "
        "state this count in your answer. Also filter aggregations with WHERE sovi != -999."
    ),
}

# Per-column documentation that the LLM should see whenever the column appears
# in retrieved schema. These are factual schema notes (semantics, ranges, common
# mistakes) — they apply uniformly to any user query that touches the column.
# Keyed by "TABLE_ID.column_name".
COLUMN_NOTES: Dict[str, str] = {
    # Population — non-obvious which to SUM
    "EX_POP_001.population_per_hex": (
        "non-overlapping population — use SUM(...) for accurate totals across hexes."
    ),
    "EX_POP_001.population_7km": (
        "7km-radius buffer; overlaps ~247 neighboring hexes — NEVER SUM (will overcount). "
        "Only AVG as a per-hex density proxy."
    ),
    # Social vulnerability — sentinel-encoded missingness
    "VUL_002.sovi": (
        "DOUBLE in roughly [0,1]; -999 = MISSING DATA. Always include "
        "AND v.sovi != -999 in WHERE/JOIN clauses, otherwise -999 rows pollute aggregates."
    ),
    # Income — raw dollars (no inverted helper exists)
    "VUL_001.median_income": (
        "Raw dollars (e.g., 65000). Lower = more vulnerable. No 'inv_median_income' column exists."
    ),
    # Hospital column naming
    "EX_LIFE_004.hosp_n": (
        "Hospital facility count per hex (BIGINT). For region totals use SUM(hosp_n). "
        "Do NOT use 'hospital_per100k' (column does not exist)."
    ),
    # NRI scores — 0-100 not 0-1
    "HP_FLD_002.nri_riverine_flood_score": (
        "0–100 risk score (NOT 0-1). Threshold >= 75 = high risk; >= 80 = critical."
    ),
    "HP_FLD_002.nri_riverine_flood_value": (
        "Expected annual loss in dollars from riverine flood."
    ),
    "HP_TOR_001.nri_tornado_score": (
        "0–100 tornado risk score. Threshold >= 75 = high risk."
    ),
    "HP_TOR_001.nri_tornado_value": (
        "Expected annual loss in dollars from tornado (use this for tornado EAL, "
        "not flood EAL or VUL_003.nri_eal which is total)."
    ),
    "HP_WFIR_001.nri_wildfire_score": (
        "0–100 wildfire risk score. Threshold >= 75 = high risk."
    ),
    "HP_WFIR_001.nri_wildfire_value": (
        "Expected annual loss in dollars from wildfire."
    ),
    # FEMA flood zone — common name confusion
    "HP_HUR_002.ve_ae_fraction": (
        "FEMA VE/AE special flood hazard area fraction in [0,1]. "
        "NOT 'fema_nfhl_sfha_fraction' (does not exist)."
    ),
    # Multi-hazard EAL
    "VUL_003.nri_eal": (
        "Total expected annual loss across ALL hazards combined. "
        "For per-hazard EAL use the suffixed columns: nri_eal_RFLD (riverine flood), "
        "nri_eal_TRND (tornado), nri_eal_HRCN (hurricane), nri_eal_WFIR (wildfire)."
    ),
    # PSVI — power system vulnerability
    "VUL_004.psvi_score": (
        "Power System Vulnerability Index, 0–100. Higher = more vulnerable. "
        "Threshold > 70 = high vulnerability; for percentile-based queries use "
        "QUANTILE_CONT(psvi_score, 0.75) instead of literal 75."
    ),
    # Community resilience — counterintuitive direction
    "CR_001.nri_cri_score": (
        "Community Resilience Index, 0–100. HIGHER = MORE resilient (NOT inverted). "
        "Low CRI (< 60) means weak resilience and is associated with worse outcomes."
    ),
    # Building stock
    "EX_BLD_001.building_count": (
        "Total building count per hex. For county-wide building stock use "
        "SUM(building_count) over all county hexes (not just high-risk subset)."
    ),
    "EX_BLD_002.hEE": (
        "Home-value weighted economic exposure of buildings in USD per hex. "
        "Use SUM for region-wide economic exposure totals."
    ),
    # Infrastructure exposure
    "EX_INF_001.CID": (
        "Critical Infrastructure Density index (0-1). Higher = denser infrastructure. "
        "For hex-level filtering use thresholds like CID > 0.3 (moderate) or > 0.5 (high)."
    ),
    "TRANS-ROAD-CRIT-INDEX.road_crit_index": (
        "Road criticality index — structural importance in the road network. "
        "Higher = more critical. Use a percentile threshold "
        "(QUANTILE_CONT(road_crit_index, 0.80)) for 'critical corridors' rather than a fixed cutoff."
    ),
    # Hurricane / surge
    "HP_HUR_001.hurr_strike_rate_10y": (
        "Expected hurricane strikes per hex over a 10-year window (BIGINT). "
        "Higher = more hurricane-exposed. Coastal counties have rates 1-3+; inland is 0."
    ),
    # Per-capita rates — DO NOT SUM
    "EX_LIFE_001.groc_per1k": (
        "Grocery stores per 1,000 residents (per-capita rate). Use AVG, "
        "NOT SUM (rates are not additive)."
    ),
    "EX_LIFE_002.pharm_per10k": (
        "Pharmacies per 10,000 residents (per-capita rate). Use AVG, NOT SUM."
    ),
    "EX_LIFE_003.fuel_per10k": (
        "Fuel/gas stations per 10,000 residents (per-capita rate). Use AVG, NOT SUM."
    ),
    # ── Per-hazard EAL columns (VUL_003) — disambiguate from total nri_eal ──
    "VUL_003.nri_eal_TRND": (
        "Total expected annual loss in USD from TORNADO. "
        "Use this column (NOT nri_eal which is total across all hazards, "
        "NOT HP_FLD_002.nri_riverine_flood_value) for tornado-specific EAL questions."
    ),
    "VUL_003.nri_eal_RFLD": (
        "Total expected annual loss in USD from RIVERINE FLOOD. "
        "Use this for riverine-flood-specific EAL questions (also "
        "available as HP_FLD_002.nri_riverine_flood_value at the hazard table)."
    ),
    "VUL_003.nri_eal_HRCN": (
        "Total expected annual loss in USD from HURRICANE. ~6% null (only "
        "coastal counties have hurricane exposure). Use this for hurricane-specific "
        "EAL questions, NOT nri_eal which is multi-hazard total."
    ),
    "VUL_003.nri_eal_WFIR": (
        "Total expected annual loss in USD from WILDFIRE. "
        "Use this for wildfire-specific EAL questions, NOT nri_eal which "
        "is multi-hazard total, NOT HP_WFIR_001.nri_wildfire_value (similar metric "
        "but on the wildfire-hazard table)."
    ),
    "VUL_003.nri_eal_CFLD": (
        "Total expected annual loss in USD from COASTAL FLOOD. "
        "~91% null (only coastal counties). For coastal-flood EAL questions; do not "
        "confuse with riverine flood (nri_eal_RFLD)."
    ),
    "VUL_003.nri_eal_HWAV": (
        "Total expected annual loss in USD from HEAT WAVE. Use this for heat-wave-"
        "specific EAL questions; nri_eal is the multi-hazard total."
    ),
    "VUL_003.nri_eal_HAIL": (
        "Total expected annual loss in USD from HAIL."
    ),
    "VUL_003.nri_eal_LTNG": (
        "Total expected annual loss in USD from LIGHTNING."
    ),
    "VUL_003.nri_eal_SWND": (
        "Total expected annual loss in USD from STRONG WIND."
    ),
    "VUL_003.nri_eal_DRGT": (
        "Total expected annual loss in USD from DROUGHT (agriculture-only)."
    ),
    # ── HIFLD facility-count columns — facility totals vs facility-bearing-hex counts ──
    "HIFLD-EMERGENC-FIRE_EMS-N.hifld_fire_and_emergency_medical_service_stations_fire_stations_ems_stations_n": (
        "Per-hex count of fire stations + EMS facilities (BIGINT). For 'how many fire/EMS "
        "facilities in <region>' use SUM(...). For 'how many hexes have at least one fire/EMS' "
        "use COUNT(*) FILTER (WHERE col > 0). The two are different metrics."
    ),
    "HIFLD-EMERGENC-LOCAL_EOC-N.hifld_local_emergency_operations_center_local_eoc_n": (
        "Per-hex count of local Emergency Operations Centers. SUM for region-wide totals; "
        "COUNT(*) FILTER WHERE col > 0 for hex-coverage footprint."
    ),
    "HIFLD-EMERGENC-LOCAL_LAW-N.hifld_local_law_enforcement_n": (
        "Per-hex count of local law enforcement facilities. SUM for facility totals."
    ),
    "HIFLD-EMERGENC-SHELTER-N.hifld_national_shelter_system_facilities_shelter_locations_n": (
        "Per-hex count of National Shelter System shelter locations. SUM for shelter totals; "
        "filter WHERE col > 0 to find hexes with any shelter coverage."
    ),
    "HIFLD-EMERGENC-STATE_EOC-N.hifld_state_emergency_operations_centers_n": (
        "Per-hex count of state Emergency Operations Centers. SUM for state-EOC totals."
    ),
    "HIFLD-ENERGY-PLANTS-N.hifld_energy_plants_n": (
        "Per-hex count of power plants. SUM for total plants in a region (not COUNT of hexes)."
    ),
    "HIFLD-ENERGY-SUBSTN-N.hifld_energy_substations_n": (
        "Per-hex count of electrical substations. SUM for substation totals; "
        "COUNT(*) FILTER WHERE col > 0 for substation-bearing hex count."
    ),
    "HIFLD-ENERGY-TXKM-230P.hifld_energy_tx_km_230p": (
        "Total length in km of high-voltage (230kV+) transmission lines per hex. "
        "DOUBLE — use SUM for region-wide transmission-line km. NOT a count."
    ),
    "HIFLD-HEALTH-HOSP-N.hifld_health_hospitals_n": (
        "Per-hex count of hospitals from HIFLD. SUM for facility totals. "
        "Note: EX_LIFE_004.hosp_n is the redundant per-hex hospital count typically "
        "preferred in queries; prefer EX_LIFE_004.hosp_n unless HIFLD provenance is required."
    ),
    "HIFLD-WATER-WTP-N.hifld_water_wtp_n": (
        "Per-hex count of water treatment plants. SUM for WTP totals; "
        "COUNT(*) FILTER WHERE col > 0 for WTP-bearing hex count."
    ),
    "HIFLD-TRANSP-PRIMARY_RD-L.hifld_primary_roads_km": (
        "Total length in km of primary roads per hex (DOUBLE). SUM for region-wide road km."
    ),
}

# Human-readable proxy suggestions surfaced to the LLM when a table is unavailable.
TABLE_PROXY_SUGGESTIONS: Dict[str, str] = {
    "HP_FLD_003":
        "HP_FLD_003 is unavailable (restricted). Use HP_FLD_002 for flood risk metrics "
        "(nri_riverine_flood_score, nri_riverine_flood_value).",
    "ER_POW_001":
        "ER_POW_001 (power outage footprint) is not available in this database. "
        "Use HIFLD-ENERGY-SUBSTN-N (substations) as a proxy for power infrastructure.",
    "HIFLD-EMERGENC-DEPENDENT_CA":
        "HIFLD-EMERGENC-DEPENDENT_CA (dependent care facilities) is not available. "
        "As a partial proxy, use HIFLD-EMERGENC-SHELTER-N for shelter locations and "
        "EX_LIFE_004 for hospital locations that may serve medically-dependent populations.",
}

# ── Concept synonyms (stored on Concept nodes for Cypher full-text search) ───
CONCEPT_SYNONYMS: Dict[str, List[str]] = {
    "elevation":        ["elevation", "height", "altitude", "topography", "ground level"],
    "hand":             ["hand", "height above nearest drainage", "drainage height"],
    "impervious":       ["impervious", "imperviousness", "pavement", "concrete",
                         "developed", "urbanization", "land cover", "land use"],
    "stream_dist":      ["stream", "river", "creek", "bayou", "channel",
                         "waterway", "waterbody", "proximity"],
    "rainfall":         ["rain", "rainfall", "precipitation", "storm"],
    "foundation":       ["foundation", "first floor", "floor height", "bfe",
                         "base flood elevation", "freeboard"],
    "far":              ["floor area ratio", "far", "building size"],
    "claims":           ["claims", "insurance", "nfip", "repetitive loss",
                         "flood history", "historical"],
    "roughness":        ["roughness", "manning", "surface roughness", "friction"],
    "runoff":           ["runoff", "surface runoff"],
    "drainage":         ["drainage", "drainage capacity"],
    "vulnerability":    ["vulnerability", "structural vulnerability",
                         "social vulnerability", "sovi", "psvi",
                         "vulnerable population", "socially vulnerable",
                         "vul_002", "vul_004"],
    "flood_occurrence": ["flood", "flooding", "flood risk", "flood occurrence",
                         "inundation", "floods", "flooded", "deluge",
                         "water damage", "high water", "overflow",
                         "coastal flood", "riverine flood",
                         "flash flood", "floodplain", "flood zone",
                         "fld", "flooding event"],
    "flood_severity":   ["damage", "severity", "flood damage", "loss",
                         "flood severity", "impact", "destruction"],
    "geography":        ["geography", "location", "county", "state", "zip",
                         "area", "region", "place", "geographic", "spatial"],
    "hurricane":        ["hurricane", "hurricanes", "tropical storm",
                         "tropical cyclone", "cyclone", "typhoon",
                         "named storm", "major hurricane",
                         "cat 1", "cat 2", "cat 3", "cat 4", "cat 5",
                         "category 1", "category 2", "category 3",
                         "hur", "hrcn", "tropical weather"],
    "wildfire":         ["wildfire", "wildfires", "forest fire",
                         "brush fire", "blaze",
                         "vegetation fire", "fire perimeter",
                         "wf", "wfire", "wildland fire", "wild-fire",
                         "crown fire", "active fire"],
    "earthquake":       ["earthquake", "earthquakes", "quake", "quakes",
                         "seismic", "seismic activity", "seismicity",
                         "tremor", "tremors", "ground shaking",
                         "ground motion", "eq", "eqk", "seismic event"],
    "tornado":          ["tornado", "tornados", "tornadoes",
                         "twister", "twisters", "funnel cloud",
                         "tor", "supercell", "tornado outbreak"],
    "severe_weather":   ["severe weather", "weather", "storm", "storms",
                         "thunderstorm", "thunderstorms",
                         "severe storm", "weather alert", "alerts",
                         "lightning", "hail", "high winds",
                         "weather event", "severe conditions"],
    "community_resilience": ["community resilience", "resilience", "nri cri", "cri score",
                              "recovery capacity", "adaptive capacity", "resilience score",
                              "community recovery", "cri", "resilient"],
    "shelters":         ["shelter", "shelters", "emergency shelter", "evacuation shelter",
                         "refuge", "shelter location", "shelter system", "evacuation site",
                         "hifld"],
    "population":       ["population", "residents", "people", "exposed population",
                         "population_7km", "nearby population"],
    "hospitals":        ["hospital", "hospitals", "medical center", "medical centers",
                         "healthcare facility", "healthcare facilities"],
    "local_eoc":        ["local eoc", "local eocs", "emergency operations center",
                         "emergency operations centers", "local emergency operations center",
                         "local emergency operations centers", "eoc", "eocs"],
    "state_eoc":        ["state eoc", "state eocs", "state emergency operations center",
                         "state emergency operations centers"],
    "water_infrastructure": ["water treatment", "water treatment plant", "water treatment plants",
                              "wtp", "wtps", "water plant", "water plants", "water supply",
                              "water service", "water infrastructure", "water system"],
    "substations":      ["substation", "substations", "electrical substation",
                         "electrical substations"],
    "power_plants":     ["power plant", "power plants", "generation plant",
                         "generation plants"],
    "power_infrastructure": ["power infrastructure", "electric infrastructure",
                             "grid infrastructure", "grid", "power grid", "utility grid"],
    "emergency_infrastructure": ["emergency infrastructure", "emergency facilities",
                                 "response infrastructure", "shelters and eocs",
                                 "shelters and hospitals", "dependent care",
                                 "dependent care facilities", "care facilities",
                                 "assisted living", "nursing home"],
    "power_disruption": ["power disruption", "power outage", "power outages", "grid failure",
                         "power failure", "electric outage", "electrical outage",
                         "without power", "power loss", "no power", "no electricity",
                         "knocked out power", "power knocked out"],
    "hospital_operations": ["hospital operations", "hospital functionality",
                            "medical operations", "clinical operations"],
    "shelter_operations": ["shelter operations", "shelter functionality",
                           "shelter access", "shelter accessibility"],
    "emergency_response_capacity": ["emergency response capacity", "response capacity",
                                    "response capability", "operational capacity",
                                    "institutional capacity"],
    "storm_surge":      ["storm surge", "coastal surge", "surge height", "storm-induced surge",
                         "surge inundation", "hurricane surge"],
    "sea_level_rise":   ["sea level rise", "slr", "rising sea level", "rising sea levels",
                         "sea-level rise", "global sea level", "relative sea level rise", "rslr"],
    "land_subsidence":  ["land subsidence", "subsidence", "ground subsidence",
                         "vertical land motion", "vlm", "land sinking"],
    "wetland_restoration": ["wetland restoration", "wetland", "wetlands",
                             "nature-based mitigation", "green infrastructure",
                             "restored wetland", "natural flood mitigation"],
    # ── Phase A additions: synonyms for previously unreachable concepts ──
    "low_income":            ["low income", "low-income", "poverty",
                              "household income", "median income",
                              "median household income",
                              "low household income", "low median income",
                              "disadvantaged communities", "disadvantaged community",
                              "lower income", "low-income households",
                              "below median income"],
    "road_segment_criticality": ["road criticality", "road segment criticality",
                                  "road crit index", "road_crit_index",
                                  "critical roads", "critical corridors",
                                  "road network criticality"],
    "floodplain":            ["floodplain", "flood plain", "flood zone",
                              "fema flood zone", "flood-prone area",
                              "100-year floodplain"],
    "floodplain_location":   ["floodplain location", "in floodplain",
                              "within floodplain", "floodplain extent"],
    "sfha_designation":      ["sfha", "special flood hazard area",
                              "fema sfha", "ve zone", "ae zone",
                              "ve/ae", "fema flood designation"],
    "inundation":            ["inundation", "inundated", "submerged",
                              "underwater", "surge inundation",
                              "flood inundation"],
    "population_density":    ["population density", "density of population",
                              "population per area", "people per square mile",
                              "densely populated"],
    "physical_vulnerability": ["physical vulnerability", "structural vulnerability",
                                "infrastructure vulnerability",
                                "property vulnerability"],
    "expected_annual_loss":  ["expected annual loss", "eal", "annual loss",
                              "expected loss", "annualized loss",
                              "annual expected loss", "expected dollar loss"],
    "buildings":             ["buildings", "building stock", "building count",
                              "structures", "building inventory",
                              "total buildings", "buildings at risk",
                              "building exposure"],
    "fire_ems":              ["fire station", "fire stations", "ems station",
                              "ems stations", "fire and ems",
                              "fire/ems", "emergency medical services",
                              "first responder facilities", "responder stations",
                              "fire department", "fire departments",
                              "fire and emergency medical"],
    "transmission_lines":    ["transmission line", "transmission lines",
                              "high voltage line", "high-voltage line",
                              "230kv", "transmission corridor",
                              "power line", "electric transmission",
                              "transmission infrastructure"],
    "primary_roads":         ["primary road", "primary roads", "highway",
                              "highways", "interstate", "major road",
                              "major roads", "principal arterial"],
    "local_law_enforcement": ["police", "police station", "police stations",
                              "law enforcement", "sheriff", "law enforcement facility",
                              "local police"],
    "infrastructure_density": ["infrastructure density", "critical infrastructure density",
                                "cid index", "infrastructure concentration"],
    "lifeline_access":       ["lifeline access", "lifeline functionality",
                              "lifeline criticality", "essential services access",
                              "access to essential services",
                              "grocery access", "pharmacy access",
                              "fuel access"],
}


def _term_matches_question(term: str, question: str) -> bool:
    """Match a synonym phrase using token boundaries instead of raw substrings."""
    escaped = re.escape(term.lower())
    escaped = escaped.replace(r"\ ", r"\s+").replace(r"\-", r"[-\s]?")
    pattern = re.compile(rf"(?<![A-Za-z0-9]){escaped}(?![A-Za-z0-9])")
    return bool(pattern.search(question.lower()))


class ContextGraph:
    """
    Unified Neo4j graph manager for the disaster context graph.

    Loads both DDCG (data catalog) and EKG (causal knowledge) into Neo4j,
    and provides Cypher-based retrieval for downstream agents.
    """

    def __init__(
        self,
        uri: str | None = None,
        user: str | None = None,
        password: str | None = None,
    ):
        self.uri = uri or os.getenv("NEO4J_URI", "bolt://localhost:7687")
        self.user = user or os.getenv("NEO4J_USER", "neo4j")
        self.password = password or os.getenv("NEO4J_PASSWORD", "password")
        self.driver = GraphDatabase.driver(
            self.uri, auth=(self.user, self.password)
        )
        logger.info(f"Connected to Neo4j at {self.uri}")

    def close(self) -> None:
        self.driver.close()

    # ── Schema constraints ───────────────────────────────────────────────

    def create_constraints(self) -> None:
        """Create uniqueness constraints for node IDs."""
        constraints = [
            "CREATE CONSTRAINT IF NOT EXISTS FOR (n:DataTable) REQUIRE n.id IS UNIQUE",
            "CREATE CONSTRAINT IF NOT EXISTS FOR (n:DataColumn) REQUIRE n.id IS UNIQUE",
            "CREATE CONSTRAINT IF NOT EXISTS FOR (n:JoinRule) REQUIRE n.id IS UNIQUE",
            "CREATE CONSTRAINT IF NOT EXISTS FOR (n:Concept) REQUIRE n.id IS UNIQUE",
        ]
        with self.driver.session() as session:
            for cypher in constraints:
                session.run(cypher)
        logger.info("Neo4j constraints created")

    def create_indexes(self) -> None:
        """Create indexes for fast lookup."""
        indexes = [
            "CREATE INDEX IF NOT EXISTS FOR (n:DataTable) ON (n.category)",
            "CREATE INDEX IF NOT EXISTS FOR (n:Concept) ON (n.type)",
        ]
        with self.driver.session() as session:
            for cypher in indexes:
                session.run(cypher)
        logger.info("Neo4j indexes created")

    def clear_graph(self) -> None:
        """Remove all nodes and relationships. Use with caution."""
        with self.driver.session() as session:
            session.run("MATCH (n) DETACH DELETE n")
        logger.info("Neo4j graph cleared")

    # ── DDCG Loading ─────────────────────────────────────────────────────

    def load_ddcg(self, ddcg_path: str | Path) -> None:
        """
        Load the DDCG (data catalog) from JSON into Neo4j.

        Creates:
          - DataTable nodes (36)
          - DataColumn nodes (145)
          - JoinRule nodes (7)
          - HAS_COLUMN edges (DataTable → DataColumn)
          - JOINABLE_VIA edges (DataTable → JoinRule)
        """
        path = Path(ddcg_path)
        with open(path) as f:
            data = json.load(f)

        tables = data.get("tables", [])
        columns = data.get("columns", [])
        join_rules = data.get("join_rules", [])
        edges = data.get("edges", [])

        with self.driver.session() as session:
            # DataTable nodes
            for t in tables:
                session.run(
                    """
                    MERGE (dt:DataTable {id: $id})
                    SET dt.name = $name,
                        dt.category = $category,
                        dt.row_count = $row_count,
                        dt.column_count = $column_count,
                        dt.columns = $columns,
                        dt.has_geometry = $has_geometry,
                        dt.sql_quoted_name = $sql_quoted_name,
                        dt.description = $description
                    """,
                    id=t["id"],
                    name=t.get("name", t["id"]),
                    category=t.get("category", "other"),
                    row_count=t.get("row_count", 0),
                    column_count=t.get("column_count", 0),
                    columns=t.get("columns", []),
                    has_geometry=t.get("has_geometry", False),
                    sql_quoted_name=t.get("sql_quoted_name", t["id"]),
                    description=t.get("description", ""),
                )

            # DataColumn nodes
            for c in columns:
                session.run(
                    """
                    MERGE (dc:DataColumn {id: $id})
                    SET dc.table_name = $table_name,
                        dc.column_name = $column_name,
                        dc.dtype = $dtype,
                        dc.is_join_key = $is_join_key,
                        dc.null_fraction = $null_fraction,
                        dc.description = $description
                    """,
                    id=c["id"],
                    table_name=c["table_name"],
                    column_name=c["column_name"],
                    dtype=c.get("dtype", "OTHER"),
                    is_join_key=c.get("is_join_key", False),
                    null_fraction=c.get("null_fraction"),
                    description=c.get("description", ""),
                )

            # JoinRule nodes
            for j in join_rules:
                session.run(
                    """
                    MERGE (jr:JoinRule {id: $id})
                    SET jr.join_type = $join_type,
                        jr.join_key = $join_key,
                        jr.tables = $tables,
                        jr.sql_template = $sql_template,
                        jr.description = $description
                    """,
                    id=j["id"],
                    join_type=j.get("join_type", "primary_key"),
                    join_key=j.get("join_key", "hex_id"),
                    tables=j.get("tables", []),
                    sql_template=j.get("sql_template", ""),
                    description=j.get("description", ""),
                )

            # HAS_COLUMN edges
            for e in edges:
                if e.get("edge_type") == "HAS_COLUMN":
                    session.run(
                        """
                        MATCH (dt:DataTable {id: $source})
                        MATCH (dc:DataColumn {id: $target})
                        MERGE (dt)-[:HAS_COLUMN]->(dc)
                        """,
                        source=e["source"],
                        target=e["target"],
                    )

            # JOINABLE_VIA edges
            for e in edges:
                if e.get("edge_type") == "JOINABLE_VIA":
                    session.run(
                        """
                        MATCH (dt:DataTable {id: $source})
                        MATCH (jr:JoinRule {id: $target})
                        MERGE (dt)-[:JOINABLE_VIA]->(jr)
                        """,
                        source=e["source"],
                        target=e["target"],
                    )

        logger.info(
            f"DDCG loaded into Neo4j: {len(tables)} tables, "
            f"{len(columns)} columns, {len(join_rules)} join rules"
        )

    # ── EKG Loading ──────────────────────────────────────────────────────

    def load_ekg(self, ekg_path: str | Path) -> None:
        """
        Load the EKG (causal knowledge graph) from JSON into Neo4j.

        Creates:
          - Concept nodes (14: features, intermediates, outcomes)
          - INCREASES/REDUCES/INDICATES edges between Concept nodes
          - MAPS_TO edges from Concept → DataTable
        """
        path = Path(ekg_path)
        with open(path) as f:
            data = json.load(f)

        nodes = data.get("nodes", [])
        edges = data.get("edges", [])

        with self.driver.session() as session:
            # Concept nodes
            for n in nodes:
                node_id = n["id"]
                thresholds = n.get("properties", {}).get("thresholds", {})
                # Prefer the hand-curated CONCEPT_SYNONYMS list when available;
                # fall back to any synonyms provided on the node JSON itself
                # (auto-imported concepts set their own minimal synonym list).
                synonyms = CONCEPT_SYNONYMS.get(node_id) or n.get("synonyms") or [
                    node_id.replace("_", " ")
                ]

                session.run(
                    """
                    MERGE (c:Concept {id: $id})
                    SET c.type = $type,
                        c.description = $description,
                        c.synonyms = $synonyms,
                        c.thresholds = $thresholds
                    """,
                    id=node_id,
                    type=n.get("type", "feature"),
                    description=n.get("description", ""),
                    synonyms=synonyms,
                    thresholds=json.dumps(thresholds) if thresholds else "{}",
                )

            # Causal edges: INCREASES, REDUCES, INDICATES
            for e in edges:
                edge_type = e["type"]
                props = e.get("properties", {})

                session.run(
                    f"""
                    MATCH (src:Concept {{id: $source}})
                    MATCH (tgt:Concept {{id: $target}})
                    MERGE (src)-[r:{edge_type} {{condition: $condition}}]->(tgt)
                    SET r.weight = $weight,
                        r.evidence = $evidence,
                        r.tdis_count = $tdis_count,
                        r.indicates_risk_reducing = $indicates_risk_reducing,
                        r.doc_source = $doc_source
                    """,
                    source=e["source"],
                    target=e["target"],
                    condition=props.get("condition", ""),
                    weight=props.get("weight", 0.5),
                    evidence=props.get("evidence", props.get("context", "")),
                    tdis_count=props.get("tdis_count", 0),
                    indicates_risk_reducing=props.get("indicates_risk_reducing", False),
                    doc_source=props.get("doc_source", ""),
                )

            # MAPS_TO edges: Concept → DataTable
            for concept_id, table_ids in CONCEPT_TABLE_MAP.items():
                for table_id in table_ids:
                    session.run(
                        """
                        MATCH (c:Concept {id: $concept_id})
                        MATCH (dt:DataTable {id: $table_id})
                        MERGE (c)-[:MAPS_TO]->(dt)
                        """,
                        concept_id=concept_id,
                        table_id=table_id,
                    )

        logger.info(
            f"EKG loaded into Neo4j: {len(nodes)} concepts, "
            f"{len(edges)} causal edges, "
            f"{sum(len(v) for v in CONCEPT_TABLE_MAP.values())} MAPS_TO edges"
        )

    # ══════════════════════════════════════════════════════════════════════
    # RETRIEVAL API — used by downstream agents
    # ══════════════════════════════════════════════════════════════════════

    def retrieve_causal_rules(
        self, concept_ids: List[str]
    ) -> List[Dict[str, Any]]:
        """
        Given a list of EKG concept IDs, retrieve all causal edges where any
        of the concepts appears as source or target (1-hop).

        Returns list of dicts with keys: source, target, type, condition,
        weight, evidence.
        """
        query = """
        MATCH (src:Concept)-[r]->(tgt:Concept)
        WHERE (src.id IN $ids OR tgt.id IN $ids)
          AND type(r) IN ['INCREASES', 'REDUCES', 'INDICATES',
                          'REQUIRES', 'TRIGGERS', 'PRECEDES', 'SCALES']
        RETURN DISTINCT
            src.id AS source,
            tgt.id AS target,
            type(r) AS type,
            r.condition AS condition,
            r.weight AS weight,
            r.evidence AS evidence,
            r.tdis_count AS tdis_count,
            r.indicates_risk_reducing AS indicates_risk_reducing
        """
        with self.driver.session() as session:
            result = session.run(query, ids=concept_ids)
            return [dict(record) for record in result]

    def retrieve_causal_paths(
        self, source_ids: List[str], target_ids: List[str], max_hops: int = 3
    ) -> List[Dict[str, Any]]:
        """
        Find all causal paths from source concepts to target concepts,
        up to max_hops edges.

        Returns list of paths with nodes and relationships.
        """
        query = f"""
        MATCH path = (src:Concept)-[*1..{max_hops}]->(tgt:Concept)
        WHERE src.id IN $source_ids AND tgt.id IN $target_ids
          AND ALL(r IN relationships(path) WHERE type(r) IN ['INCREASES', 'REDUCES', 'INDICATES', 'REQUIRES', 'TRIGGERS', 'PRECEDES', 'SCALES'])
        RETURN
            [n IN nodes(path) | n.id] AS node_ids,
            [r IN relationships(path) |
                {{type: type(r), condition: r.condition, weight: r.weight, evidence: r.evidence}}
            ] AS edges,
            length(path) AS hops
        ORDER BY hops ASC
        """
        with self.driver.session() as session:
            result = session.run(
                query, source_ids=source_ids, target_ids=target_ids
            )
            return [dict(record) for record in result]

    def match_concepts(self, question: str) -> List[Dict[str, Any]]:
        """
        Match a natural language question to Concept nodes using synonym lists.

        For each Concept node, checks if any of its synonyms appear in the
        question text. Returns matching Concept nodes sorted by specificity
        (multi-word matches first).

        Returns list of dicts with keys: id, type, description, synonyms,
        matched_term.
        """
        question_lower = question.lower()

        query = """
        MATCH (c:Concept)
        RETURN c.id AS id, c.type AS type, c.description AS description,
               c.synonyms AS synonyms, c.thresholds AS thresholds
        """
        with self.driver.session() as session:
            result = session.run(query)
            all_concepts = [dict(record) for record in result]

        matched = []
        for concept in all_concepts:
            synonyms = concept.get("synonyms", [])
            if not synonyms:
                continue
            # Sort by length descending so multi-word phrases match first
            for term in sorted(synonyms, key=len, reverse=True):
                if _term_matches_question(term, question_lower):
                    matched.append({
                        **concept,
                        "matched_term": term,
                    })
                    break  # one match per concept is enough

        return matched

    def retrieve_schema_for_concepts(
        self, concept_ids: List[str]
    ) -> List[Dict[str, Any]]:
        """
        Given concept IDs, traverse MAPS_TO edges to find the relevant
        DataTables, then return their schema (columns, joins).

        This is the bridge between causal knowledge and SQL generation.

        Returns list of table dicts with columns and join rules.
        """
        query = """
        MATCH (c:Concept)-[:MAPS_TO]->(dt:DataTable)
        WHERE c.id IN $ids
        OPTIONAL MATCH (dt)-[:HAS_COLUMN]->(dc:DataColumn)
        WHERE dc.is_join_key = false
        OPTIONAL MATCH (dt)-[:JOINABLE_VIA]->(jr:JoinRule)
        RETURN
            dt.id AS table_id,
            dt.sql_quoted_name AS sql_quoted_name,
            dt.category AS category,
            dt.row_count AS row_count,
            dt.description AS table_description,
            collect(DISTINCT {
                column_name: dc.column_name,
                dtype: dc.dtype,
                null_fraction: dc.null_fraction
            }) AS columns,
            collect(DISTINCT {
                join_id: jr.id,
                join_type: jr.join_type,
                sql_template: jr.sql_template
            }) AS join_rules,
            collect(DISTINCT c.id) AS linked_concepts
        """
        with self.driver.session() as session:
            result = session.run(query, ids=concept_ids)
            tables = []
            for record in result:
                table_dict = dict(record)
                # Filter out null entries from optional matches
                table_dict["columns"] = [
                    c for c in table_dict["columns"]
                    if c.get("column_name") is not None
                ]
                table_dict["join_rules"] = [
                    j for j in table_dict["join_rules"]
                    if j.get("join_id") is not None
                ]
                tables.append(table_dict)
            return tables

    def check_concept_runnability(self, concept_ids: List[str]) -> Dict[str, Any]:
        """
        Check whether the data tables mapped to the given concepts are available.

        Returns:
            {
              "runnable": list of concept IDs whose tables are available (or have none),
              "warnings": list of {"concept": str, "table": str, "suggestion": str}
            }

        A concept is runnable if at least one of its mapped tables is available,
        OR if it has no table mapping (abstract concept like runoff, drainage).
        Warnings are emitted for every unavailable table regardless of runnability.
        No Neo4j call — uses module-level CONCEPT_TABLE_MAP and UNAVAILABLE_TABLES.
        """
        runnable: List[str] = []
        warnings: List[Dict[str, str]] = []

        for cid in concept_ids:
            tables = CONCEPT_TABLE_MAP.get(cid, [])
            if not tables:
                runnable.append(cid)
                continue

            unavailable = [t for t in tables if t in UNAVAILABLE_TABLES]
            available = [t for t in tables if t not in UNAVAILABLE_TABLES]

            if available:
                runnable.append(cid)

            for t in unavailable:
                warnings.append({
                    "concept": cid,
                    "table": t,
                    "suggestion": TABLE_PROXY_SUGGESTIONS.get(
                        t, f"Table '{t}' is unavailable and has no registered proxy."
                    ),
                })

        return {"runnable": runnable, "warnings": warnings}

    def retrieve_full_schema(self) -> List[Dict[str, Any]]:
        """
        Retrieve the complete DDCG schema from Neo4j for LLM context injection.

        Returns all DataTable nodes with their columns and join rules.
        Used as fallback when no specific concepts are matched.
        """
        query = """
        MATCH (dt:DataTable)
        OPTIONAL MATCH (dt)-[:HAS_COLUMN]->(dc:DataColumn)
        WHERE dc.is_join_key = false
        OPTIONAL MATCH (dt)-[:JOINABLE_VIA]->(jr:JoinRule)
        RETURN
            dt.id AS table_id,
            dt.sql_quoted_name AS sql_quoted_name,
            dt.category AS category,
            dt.row_count AS row_count,
            dt.description AS table_description,
            dt.has_geometry AS has_geometry,
            collect(DISTINCT {
                column_name: dc.column_name,
                dtype: dc.dtype,
                null_fraction: dc.null_fraction
            }) AS columns,
            collect(DISTINCT {
                join_id: jr.id,
                join_type: jr.join_type,
                sql_template: jr.sql_template,
                description: jr.description
            }) AS join_rules
        ORDER BY dt.category, dt.id
        """
        with self.driver.session() as session:
            result = session.run(query)
            tables = []
            for record in result:
                table_dict = dict(record)
                table_dict["columns"] = [
                    c for c in table_dict["columns"]
                    if c.get("column_name") is not None
                ]
                table_dict["join_rules"] = [
                    j for j in table_dict["join_rules"]
                    if j.get("join_id") is not None
                ]
                tables.append(table_dict)
            return tables

    def schema_to_prompt_context(
        self, tables: List[Dict[str, Any]] | None = None
    ) -> str:
        """
        Render schema data (from retrieve_full_schema or
        retrieve_schema_for_concepts) as compact text for LLM injection.

        Same format as the old DDCG.to_prompt_context() so downstream
        agents don't need to change their prompts.
        """
        if tables is None:
            tables = self.retrieve_full_schema()

        lines: list[str] = ["=== Data Catalog (DDCG) ===", ""]

        for tbl in tables:
            cols = tbl.get("columns", [])
            table_id = tbl.get("table_id", "")
            col_strs = [
                f"{c['column_name']} ({c['dtype']})"
                for c in cols
                if c.get("column_name")
            ]
            sql_name = tbl.get("sql_quoted_name", table_id)
            category = tbl.get("category", "other")
            row_count = tbl.get("row_count", 0)

            lines.append(f"TABLE {sql_name}  [{category}]")
            lines.append(f"  rows: {row_count:,}")
            lines.append(f"  columns: hex_id (VARCHAR, PK), {', '.join(col_strs)}")

            desc = tbl.get("table_description", "")
            if desc:
                lines.append(f"  note: {desc}")

            # Warn about high-null columns
            high_null = [
                c for c in cols
                if c.get("null_fraction") and c["null_fraction"] > 0.5
            ]
            if high_null:
                warnings = [
                    f"{c['column_name']} ({c['null_fraction']:.0%} null)"
                    for c in high_null
                ]
                lines.append(f"  WARNING sparse: {', '.join(warnings)}")

            # Per-column factual notes (semantics, ranges, common mistakes)
            for c in cols:
                col_name = c.get("column_name")
                if not col_name:
                    continue
                note = COLUMN_NOTES.get(f"{table_id}.{col_name}")
                if note:
                    lines.append(f"  note {col_name}: {note}")

            lines.append("")

        # Join rules summary (deduplicated)
        all_joins: Dict[str, Dict] = {}
        for tbl in tables:
            for jr in tbl.get("join_rules", []):
                jid = jr.get("join_id", "")
                if jid and jid not in all_joins:
                    all_joins[jid] = jr

        if all_joins:
            lines.append("--- Join Rules ---")
            for jid, jr in all_joins.items():
                template = jr.get("sql_template", "")
                desc = jr.get("description", "")
                lines.append(f"  {jid}: {template}  ({desc})")

        return "\n".join(lines)

    # ── Graph stats ──────────────────────────────────────────────────────

    def get_stats(self) -> Dict[str, int]:
        """Return node/relationship counts for verification."""
        queries = {
            "DataTable": "MATCH (n:DataTable) RETURN count(n) AS cnt",
            "DataColumn": "MATCH (n:DataColumn) RETURN count(n) AS cnt",
            "JoinRule": "MATCH (n:JoinRule) RETURN count(n) AS cnt",
            "Concept": "MATCH (n:Concept) RETURN count(n) AS cnt",
            "HAS_COLUMN": "MATCH ()-[r:HAS_COLUMN]->() RETURN count(r) AS cnt",
            "JOINABLE_VIA": "MATCH ()-[r:JOINABLE_VIA]->() RETURN count(r) AS cnt",
            "MAPS_TO": "MATCH ()-[r:MAPS_TO]->() RETURN count(r) AS cnt",
            "INCREASES": "MATCH ()-[r:INCREASES]->() RETURN count(r) AS cnt",
            "REDUCES": "MATCH ()-[r:REDUCES]->() RETURN count(r) AS cnt",
            "INDICATES": "MATCH ()-[r:INDICATES]->() RETURN count(r) AS cnt",
        }
        stats = {}
        with self.driver.session() as session:
            for label, q in queries.items():
                result = session.run(q)
                stats[label] = result.single()["cnt"]
        return stats


# ── Module-level singleton ───────────────────────────────────────────────────

_graph: Optional[ContextGraph] = None


def get_context_graph() -> ContextGraph:
    """Get or create the singleton ContextGraph."""
    global _graph
    if _graph is None:
        _graph = ContextGraph()
    return _graph
