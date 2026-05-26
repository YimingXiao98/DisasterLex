# Structural Damage
Estimate building damage and economic losses based on hazard exposure and building stock data.

## Objective
Provide rapid loss estimation to prioritize damage assessment teams and FEMA assistance requests.

## Key Datasets
- `EX_BLD_001`
- `EX_BLD_002`
- `HP_FLD_002`
- `EX_POP_001`

## Core Logic
1. Query `EX_BLD_001` for `SUM(building_count)` in the analysis area (all hexes in county).
2. **High-risk building count:** Join with `HP_FLD_002` — count buildings in hexes where `nri_riverine_flood_score >= 75`.
3. **Flood EAL (required):** `SELECT ROUND(SUM(f.nri_riverine_flood_value), 2) AS total_flood_eal FROM HP_FLD_002 f JOIN hex_county_state_zip_crosswalk x ON f.hex_id=x.hex_id WHERE x.County='<AREA>' AND f.nri_riverine_flood_score >= 75` — report as total riverine flood Expected Annual Loss for high-risk hexes. This is DISTINCT from economic exposure (hEE). **CRITICAL: The `nri_riverine_flood_score >= 75` filter is MANDATORY. Omitting it sums losses across all county hexes (including low-risk ones) and produces a result ~360× too large.**
4. **Average population at risk:** `SELECT ROUND(AVG(p.population_7km), 2) FROM EX_POP_001 p JOIN HP_FLD_002 f ON p.hex_id=f.hex_id JOIN hex_county_state_zip_crosswalk x ON p.hex_id=x.hex_id WHERE x.County='<AREA>' AND f.nri_riverine_flood_score >= 75`
5. Use `EX_BLD_002.hEE` — calculate total economic exposure (`SUM(hEE)`) for context (separate from EAL).
6. If Criticality >= 4: Flag for FEMA Preliminary Damage Assessment (PDA).
