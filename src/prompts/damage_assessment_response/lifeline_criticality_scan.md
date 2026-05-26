# Lifeline Criticality Scan
Triage critical infrastructure damage — count operational hospitals, power substations, and lifeline hubs in the analysis area.

## Objective
Assess the operational status of critical lifeline infrastructure (hospitals, power substations) and identify community dependencies at risk following a disaster event.

## Key Datasets
- `EX_LIFE_004` — hospital presence (`hosp_n > 0` means a hospital hex; `hosp_n` is BIGINT count)
- `HIFLD-ENERGY-SUBSTN-N` — power substations (`hifld_energy_substations_n > 0`)
- `CRIT_LIFE_001-004` (Lifeline RAC)
- `HP_TOR_001`, `HP_FLD_002` — hazard exposure
- `hex_county_state_zip_crosswalk` — geography filter

## Core Logic
1. **Hazard baseline (always report first):** Count all hexes in the county at high hazard risk using the standard threshold:
   - Flood: `SELECT COUNT(*) FROM HP_FLD_002 f JOIN hex_county_state_zip_crosswalk x ON f.hex_id=x.hex_id WHERE x.County='<County>' AND f.nri_riverine_flood_score >= 75` — **always use `>= 75`, never "above county average" or other relative thresholds.**
   - Tornado: same pattern with `HP_TOR_001.nri_tornado_score >= 75`.
   - Hurricane: same pattern with `HP_HUR_002.ve_ae_fraction > 0`.
   Report this total as county-wide hazard exposure before any infrastructure-specific counts.
2. Count total hospital hexes: `SELECT COUNT(*) FROM EX_LIFE_004 h JOIN hex_county_state_zip_crosswalk x ON h.hex_id=x.hex_id WHERE x.County='<County>' AND h.hosp_n > 0`
3. Count total substation hexes: `SELECT COUNT(*) FROM "HIFLD-ENERGY-SUBSTN-N" s JOIN hex_county_state_zip_crosswalk x ON s.hex_id=x.hex_id WHERE x.County='<County>' AND s.hifld_energy_substations_n > 0`
4. For tornado events: join with `HP_TOR_001`, flag hexes where `nri_tornado_score >= 75` (top 25% risk).
5. For flood events: join with `HP_FLD_002`, flag hexes where `nri_riverine_flood_score >= 75`.
6. Report: total high-hazard hexes (from step 1), total hospital hexes, total substation hexes, count of each at high hazard risk.
7. If Criticality >= 4: Recommend activating backup power for hospitals in high-risk zones; pre-position repair crews at exposed substations.
