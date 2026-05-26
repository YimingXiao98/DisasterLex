# Medical Triage
Identify medical facilities and hospitals at risk of hazard impact or isolation.

## Objective
Assess the vulnerability of the healthcare network and identify potential medical deserts during an active event.

## Key Datasets
- `EX_LIFE_004` — hospital presence: `hosp_n > 0` means a hospital hex; `hosp_n` is BIGINT count
- `HP_FLD_002` — riverine and coastal flood risk scores (0–100 percentile)
- `CR_001` — community resilience score (`nri_cri_score`, 0–100; higher = more resilient)
- `hex_county_state_zip_crosswalk` — geography filter

## Core Logic
1. Query `EX_LIFE_004` for hexes with `hosp_n > 0` in the analysis area (join via crosswalk).
2. Join with `HP_FLD_002` — flag hospital hexes where `nri_riverine_flood_score >= 75` or `nri_coastal_flood_score >= 75` (top 25% flood risk).
3. Join with `CR_001` — identify low-resilience hospital hexes (`nri_cri_score < 50`).
4. Count total hospital hexes, flagged hexes, and compute percentage at risk.
5. If Criticality >= 4: Recommend activating mutual aid and field hospital staging.
