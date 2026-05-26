# EJ Triple Burden
Identify statewide environmental-justice hotspots where hazard exposure, social vulnerability, and service gaps overlap.

## Objective
Quantify the most extreme EJ-burden hexes by combining high hazard exposure, high social vulnerability, and lack of emergency services.

## Key Datasets
- `VUL_002` (SoVI)
- `HP_FLD_002`
- `HP_HUR_001`
- `HP_TOR_001`
- `crown_fire_probability`
- `HIFLD-EMERGENC-FIRE_EMS-N`
- `hex_county_state_zip_crosswalk`

## Core Logic
1. Compute statewide percentile thresholds using `QUANTILE_CONT`: `sovi_p75` from `VUL_002.sovi` excluding `-999`, and `haz_p75` from a weighted multi-hazard score built from flood, hurricane, tornado, and wildfire components.
2. Return a SINGLE statewide aggregate `high_sovi_high_hazard_no_ems_hex_count = COUNT(DISTINCT hex_id)` for hexes where `sovi > sovi_p75`, weighted hazard score `> haz_p75`, and EMS coverage count is zero.
3. Define these locations explicitly as a **triple burden**: high hazard exposure increases infrastructure or power disruption, while low community resilience reduces emergency response capacity, and both pathways are amplified when no nearby EMS services exist.
4. Use statewide scope. Do **NOT** fall back to a single-hazard population-impact summary.
5. Recommend equity-focused mitigation and emergency-service gap closure.
