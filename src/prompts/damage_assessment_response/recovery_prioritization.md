# Recovery Prioritization
Identify and rank geographic areas for post-event recovery resource deployment based on compounding damage exposure, low community resilience, and high social vulnerability.

## Objective
Guide post-disaster recovery prioritization so that FEMA Individual Assistance, Community Development Block Grant - Disaster Recovery (CDBG-DR), and state recovery resources are directed to the highest-need hexes first.

## Key Datasets
- `EX_BLD_001`
- `CR_001`
- `VUL_002`
- `HP_HUR_002` (hurricane surge), `HP_FLD_002` (flood), or `HP_WFIR_001` (wildfire) — select by hazard_type

## Core Logic
1. Query the hazard exposure table for the analysis area using the hazard-specific threshold:
   - **Hurricane:** `HP_HUR_002.ve_ae_fraction > 0.3` (surge zone fraction; NOT a score — use this threshold, not >= 50)
   - **Flood:** `HP_FLD_002.nri_riverine_flood_score >= 50`
   - **Wildfire:** `HP_WFIR_001.nri_wildfire_score >= 50`
2. Join with `CR_001` — filter for hexes where `nri_cri_score < 50` (below-median resilience; lower score = less resilient).
3. **Report first (required):** Count and report hexes meeting criteria from steps 1+2 ONLY (hazard exposure + low resilience) — this is the primary recovery zone count. For hurricane: `SELECT COUNT(*) FROM HP_HUR_002 h JOIN CR_001 c ON h.hex_id=c.hex_id JOIN hex_county_state_zip_crosswalk x ON h.hex_id=x.hex_id WHERE x.County='<AREA>' AND h.ve_ae_fraction>0.3 AND c.nri_cri_score<50`.
4. Then also join with `VUL_002` — filter for hexes where `sovi > 0` AND `sovi != -999` for additional vulnerability context. Count these as Priority Tier 1 recovery targets.
5. Query `EX_BLD_001` — sum `building_count` for Priority Tier 1 hexes to estimate affected building stock.
6. Report: primary recovery zone hex count (step 3), Priority Tier 1 hexes (step 4), building count, average CRI score, average SoVI.
7. If Criticality >= 3: Recommend submission of FEMA Preliminary Damage Assessment (PDA) request for Priority Tier 1 areas. If Criticality >= 4: Flag for immediate Individual Assistance (IA) declaration consideration.
