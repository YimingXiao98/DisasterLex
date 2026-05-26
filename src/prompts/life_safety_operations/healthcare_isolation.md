# Healthcare Isolation Risk
Identify hospitals at risk of being isolated from the community due to compromised road infrastructure.

## Objective
Assess the "reachable" healthcare capacity by analyzing the intersection of hospital locations and road criticality.

## Key Datasets
- `EX_LIFE_004` (Hospital hexes: `hosp_n > 0` means a hospital is present)
- `TRANS-ROAD-CRIT-INDEX`
- `HP_FLD_002`

## Core Logic
1. **Hospital count (required first):** `SELECT COUNT(*) FROM EX_LIFE_004 e JOIN hex_county_state_zip_crosswalk x ON e.hex_id = x.hex_id WHERE x.County = '<AREA>' AND e.hosp_n > 0`. Use the full county name with "County" suffix. Report this count explicitly.
2. **Road access summary:** `SELECT COUNT(*) AS hospital_hexes_with_road_data, AVG(r.road_crit_index) AS avg_road_crit, MAX(r.road_crit_index) AS max_road_crit FROM EX_LIFE_004 e JOIN "TRANS-ROAD-CRIT-INDEX" r ON e.hex_id = r.hex_id JOIN hex_county_state_zip_crosswalk x ON e.hex_id = x.hex_id WHERE x.County = '<AREA>' AND e.hosp_n > 0` — report avg and max road criticality for hospital hexes.
3. **Flood risk overlay:** `SELECT COUNT(*) AS hospital_hexes_high_flood FROM EX_LIFE_004 e JOIN HP_FLD_002 f ON e.hex_id = f.hex_id JOIN hex_county_state_zip_crosswalk x ON e.hex_id = x.hex_id WHERE x.County = '<AREA>' AND e.hosp_n > 0 AND f.nri_riverine_flood_score >= 75` — count hospital hexes with top-quartile flood risk. *(HP_FLD_003/floodgenome is not available; using NRI riverine flood score as proxy.)*
4. Flag hospitals where flood risk is high (nri_riverine_flood_score >= 75) AND road_crit_index > 0 as potentially isolated.
5. Note: The only valid hospital count column is `hosp_n` — do NOT use `hosp_per100k` (column does not exist).
6. If Criticality >= 4: Recommend air-medical standby or temporary water-accessible clinics.
