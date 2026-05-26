# Evacuation Routing
Determine safe evacuation corridors and identify priority populations for displacement.

## Objective
Map safe evacuation routes by identifying critical road infrastructure that remains uncompromised by hazard zones.

## Key Datasets
- `TRANS-ROAD-CRIT-INDEX`
- `HIFLD-TRANSP-PRIMARY_RD-L`
- `EX_POP_001`
- `VUL_002`
- `HP_FLD_002` (flood/hurricane events)
- `HP_WFIR_001` (wildfire events)

## Core Logic
1. **Road criticality check:** `SELECT COUNT(*) AS nonzero_hexes, AVG(road_crit_index) AS avg_crit FROM "TRANS-ROAD-CRIT-INDEX" t JOIN hex_county_state_zip_crosswalk x ON t.hex_id = x.hex_id WHERE x.County = '<AREA>' AND road_crit_index > 0` — if nonzero_hexes = 0 (or avg_crit < 0.01), skip to step 1b.
   **Step 1b (fallback):** Use `CR_001.nri_cri_score` as proxy for community connectivity when road_crit_index is all zero.
2. **Viable corridor count:**
   - **Flood/hurricane events:** `WITH global_p80 AS (SELECT QUANTILE_CONT(road_crit_index, 0.80) AS p80_val FROM "TRANS-ROAD-CRIT-INDEX") SELECT COUNT(*) AS viable_corridor_hexes FROM "TRANS-ROAD-CRIT-INDEX" t JOIN hex_county_state_zip_crosswalk x ON t.hex_id = x.hex_id JOIN HP_FLD_002 f ON t.hex_id = f.hex_id CROSS JOIN global_p80 WHERE x.County = '<AREA>' AND t.road_crit_index >= global_p80.p80_val AND f.nri_riverine_flood_score < 75` — use `QUANTILE_CONT(column, fraction)` NOT `PERCENTILE_CONT`. Report this count as viable evacuation corridor hexes. Note: most hexes have road_crit_index=0, so global p80 is typically 0, meaning all hexes with flood_score < 75 qualify as viable corridors.
   - **Wildfire events:** `SELECT COUNT(*) AS lower_fire_risk_hexes FROM HP_WFIR_001 w JOIN hex_county_state_zip_crosswalk x ON w.hex_id=x.hex_id WHERE x.County='<AREA>' AND w.nri_wildfire_score < 75` — hexes outside the high-fire zone are viable corridors. Also count fire-path hexes: `SELECT COUNT(*) AS fire_path_hexes FROM HP_WFIR_001 w JOIN hex_county_state_zip_crosswalk x ON w.hex_id=x.hex_id WHERE x.County='<AREA>' AND w.nri_wildfire_score > 50` — report as fire path extent. **CRITICAL (wildfire): Use ONLY `HP_WFIR_001` and column `nri_wildfire_score` (0–100). Do NOT join with `HP_FLD_002` for wildfire scenarios. In high-risk counties, nearly all hexes may have `nri_wildfire_score > 50` — this is expected and correct.**
3. **Compromised route check:**
   - **Flood/hurricane:** `SELECT COUNT(*) AS compromised_hexes FROM HP_FLD_002 f JOIN hex_county_state_zip_crosswalk x ON f.hex_id = x.hex_id WHERE x.County = '<AREA>' AND f.nri_riverine_flood_score >= 75` — flag as "DO NOT USE" routes.
   - **Wildfire:** `SELECT COUNT(*) AS fire_compromised_hexes FROM HP_WFIR_001 w JOIN hex_county_state_zip_crosswalk x ON w.hex_id=x.hex_id WHERE x.County='<AREA>' AND w.nri_wildfire_score >= 75` — flag as "DO NOT USE" routes.
4. Overlay with `EX_POP_001` — prioritize evacuation of hexes with highest `population_7km`.
5. Use `VUL_002.sovi` (excluding -999) to prioritize vulnerable populations.
6. Query `HIFLD-EMERGENC-SHELTER-N` — count hexes where `hifld_national_shelter_system_facilities_shelter_locations_n > 0` AND `nri_riverine_flood_score < 75` (viable shelter destinations not in high-flood zone). Report this count explicitly.
7. If Criticality >= 3: Mark compromised routes as "DO NOT USE" and recommend alternates.
