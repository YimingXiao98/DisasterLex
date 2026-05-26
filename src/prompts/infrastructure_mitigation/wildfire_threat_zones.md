# Wildfire Threat Zones
Identify high-probability wildfire zones and at-risk infrastructure/buildings.

## Objective
Assess the wildfire risk to transmission lines and general building stock to inform PSPS and protection activities.

## Key Datasets
- `crown_fire_probability`
- `EX_INF_001`
- `HIFLD-ENERGY-TXKM-230P`
- `EX_BLD_001`
- `VUL_004`

## Core Logic
1. **High-risk wildfire hex count:** `WITH county_hexes AS (SELECT cf.hex_id, cf.crown_fire_prob FROM crown_fire_probability cf JOIN hex_county_state_zip_crosswalk x ON cf.hex_id = x.hex_id WHERE x.County = '<AREA>'), p80 AS (SELECT QUANTILE_CONT(crown_fire_prob, 0.80) AS p80_val FROM county_hexes) SELECT COUNT(*) AS high_fire_risk_hexes FROM county_hexes CROSS JOIN p80 WHERE crown_fire_prob >= p80.p80_val` — use `QUANTILE_CONT(column, fraction)` NOT `PERCENTILE_CONT` or window functions in WHERE. `crown_fire_prob` values range 0.0–0.32.
2. Join with \`EX_INF_001\` — identify hexes with \`CID > 0.3\` (moderate+ infrastructure).
3. Join with \`HIFLD-ENERGY-TXKM-230P\` — flag hexes with transmission lines at risk.
4. Join with \`EX_BLD_001\` — count buildings in fire-prone hexes.
5. Use \`VUL_004.psvi_score\` to assess power system vulnerability in fire zones.
6. If Criticality >= 3: Recommend Public Safety Power Shutoff (PSPS) for affected lines.
