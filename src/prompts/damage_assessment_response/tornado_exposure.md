# Tornado Exposure
Quantify the building and population exposure to tornado-prone zones.

## Objective
Rapidly estimate potential impacts following a tornado advisory or strike by mapping the building stock in high-score zones.

## Key Datasets
- `HP_TOR_001` (NRI Tornado)
- `EX_BLD_001` (Building count)
- `EX_BLD_002` (Economic exposure)
- `EX_POP_001`

## Core Logic
1. **High-risk hex count (required):** `SELECT COUNT(*) AS high_tornado_risk_hexes FROM HP_TOR_001 t JOIN hex_county_state_zip_crosswalk x ON t.hex_id=x.hex_id WHERE x.County='<AREA>' AND t.nri_tornado_score >= 75` — report this count explicitly. Use fixed threshold `>= 75` (not a local percentile).
2. **Economic exposure:** `SELECT ROUND(SUM(t.nri_tornado_value), 2) AS total_tornado_econ_loss FROM HP_TOR_001 t JOIN hex_county_state_zip_crosswalk x ON t.hex_id=x.hex_id WHERE x.County='<AREA>' AND t.nri_tornado_score >= 75` — report as total EAL for high-risk hexes.
3. Join with `EX_BLD_001` — sum `building_count` for hexes where `nri_tornado_score >= 75`.
4. Calculate `AVG(population_7km)` from `EX_POP_001` for high-tornado-risk hexes.
5. If Criticality >= 3: Recommend activating tornado shelters and pre-alerting SAR (Search and Rescue) teams.
