# Water Supply Risk
Assess the risk of flood, surge, and wildfire contamination impact to Water Treatment Plants (WTP).

## Objective
Prevent water service interruptions by identifying critical treatment facilities in active hazard zones.

## Key Datasets
- `HIFLD-WATER-WTP-N`
- `HP_FLD_002`
- `HP_HUR_002` (Surge)
- `HP_WFIR_001` (Wildfire — ash/runoff contamination, use ONLY when hazard_type=wildfire)

## Core Logic
1. Count total WTPs in the analysis area: `SELECT COUNT(*) FROM "HIFLD-WATER-WTP-N" w JOIN hex_county_state_zip_crosswalk x ON w.hex_id=x.hex_id WHERE x.County='<County>' AND w.hifld_water_wtp_n>0`
2. **For flood/hurricane events ONLY:** Flag WTPs at high flood risk using EXACTLY this threshold (`nri_riverine_flood_score >= 80`): `SELECT COUNT(*) FROM "HIFLD-WATER-WTP-N" w JOIN HP_FLD_002 f ON w.hex_id=f.hex_id JOIN hex_county_state_zip_crosswalk x ON w.hex_id=x.hex_id WHERE x.County='<County>' AND w.hifld_water_wtp_n>0 AND f.nri_riverine_flood_score >= 80` — use ONLY this threshold, do NOT run additional queries with other thresholds.
3. **For coastal/hurricane events:** Also check `HP_HUR_002.ve_ae_fraction > 0.5` (surge zone).
4. **ONLY IF hazard_type=wildfire (skip entirely for flood/hurricane):** Count high wildfire risk hexes: `SELECT COUNT(*) AS high_fire_risk_hexes FROM HP_WFIR_001 w JOIN hex_county_state_zip_crosswalk x ON w.hex_id=x.hex_id WHERE x.County='<County>' AND w.nri_wildfire_score >= 75`. Then count WTPs in that zone. Report this as the contamination risk zone extent.
5. Report: total WTPs, count at flood risk (score >= 80), count in surge zone (if applicable), count in high wildfire risk zone (if wildfire event only).
6. If Criticality >= 4: Recommend activation of backup pumps and flood-barrier deployment.
