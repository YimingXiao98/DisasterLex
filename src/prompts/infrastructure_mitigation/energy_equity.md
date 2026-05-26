# Energy Equity Analysis
Identify communities facing both high power system vulnerability and low financial resilience.

## Objective
Prioritize power grid restoration and backup generator distribution for disadvantaged communities.

## Key Datasets
- `VUL_004` (PSVI)
- `VUL_001` (Median Income)
- `EX_POP_001`
- `VUL_002` (SoVI)

## Core Logic
1. **High-PSVI hex count (required, report explicitly):** `SELECT COUNT(*) AS high_psvi_hex_count FROM VUL_004 v JOIN hex_county_state_zip_crosswalk x ON v.hex_id=x.hex_id WHERE x.County='<AREA>' AND v.psvi_score > 70` — report this COUNT (e.g., 1293), NOT the psvi_score values.
2. **Low-income filter:** `SELECT COUNT(*) FROM VUL_004 v JOIN VUL_001 inc ON v.hex_id=inc.hex_id JOIN hex_county_state_zip_crosswalk x ON v.hex_id=x.hex_id WHERE x.County='<AREA>' AND v.psvi_score > 70 AND inc.median_income < 35000` — **CRITICAL: VUL_001 has `median_income` (raw dollars). There is NO `inv_median_income` column. Low income = LOW `median_income`.**
3. Join with `VUL_002` to prioritize communities with highest social vulnerability. Exclude `sovi = -999` (missing sentinel).
4. Rank hexes by composite vulnerability: high `psvi_score` + low `median_income` + high valid `sovi`.
5. If Criticality >= 3: Recommend prioritizing these areas for community solar or backup power assistance programs.
