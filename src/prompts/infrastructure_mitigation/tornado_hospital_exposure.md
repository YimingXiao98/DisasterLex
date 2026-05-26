# Tornado Hospital Exposure
Identify hospitals and healthcare capacity inside significant tornado-risk zones.

## Objective
Map hospital exposure during tornado outbreaks by aggregating healthcare facilities across the full warned region.

## Key Datasets
- `HP_TOR_001` (NRI Tornado)
- `EX_LIFE_004` (Hospital count)
- `hex_county_state_zip_crosswalk`

## Core Logic
1. **Primary healthcare exposure metric (required first):** For the full named region, return `hospitals_in_high_tornado_zone = SUM(EX_LIFE_004.hosp_n)` for hexes where `CAST(HP_TOR_001.nri_tornado_score AS DOUBLE) > 0.5`. Use a SINGLE aggregate across all requested counties.
2. Use facility counts, not occupied-hex counts. Do **NOT** replace the primary metric with `COUNT(*)` of hospital-bearing hexes.
3. If useful for context, also report `hospital_hexes_in_high_tornado_zone = COUNT(*) FILTER (WHERE COALESCE(hosp_n,0) > 0)`.
4. Keep the analysis on tornado exposure. Do **NOT** switch to hurricane surge or coastal strike-rate logic unless the user explicitly asks for it.
5. If Criticality >= 3: recommend immediate hospital readiness checks, backup-power verification, and redundant communications with local EOCs.
