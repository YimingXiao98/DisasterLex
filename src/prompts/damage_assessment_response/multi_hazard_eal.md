# Multi-Hazard Loss Estimation
Aggregate Expected Annual Loss (EAL) across multiple hazard types to determine cumulative risk.

## Objective
Identify "high-loss clusters" where multiple disaster types (Flood, Wind, Fire) create extreme economic vulnerability.

## Key Datasets
- `VUL_003` (NRI EAL)
- `EX_BLD_002` (Economic exposure)
- `CR_001` (Community Resilience)

## Core Logic
1. **County average EAL (required, always report first):** `SELECT ROUND(AVG(v.nri_eal), 2) AS county_avg_eal FROM VUL_003 v JOIN hex_county_state_zip_crosswalk x ON v.hex_id=x.hex_id WHERE x.County='<AREA>'` — report this value explicitly in dollars (e.g., 710891.30). **CRITICAL: report raw AVG(nri_eal) in dollars. Do NOT report nri_eal/nri_cri_score ratios — those are internal ranking values only.**
2. Identify hexes where `nri_eal` is significantly higher than the county average (e.g., > 2× average).
3. Join with hazard-specific scores: `nri_eal_HRCN`, `nri_eal_RFLD`, `nri_eal_WFIR` — to show which hazard drives the loss.
4. Cross-reference with `nri_cri_score` (`CR_001`) — flag high-EAL hexes that have low resilience (`nri_cri_score < 40`).
5. Rank by `nri_eal / nri_cri_score` ratio for prioritization (internal use — do not report raw ratio as EAL).
6. If Criticality >= 4: Prioritize these clusters for post-event Federal Disaster Declaration assistance.
