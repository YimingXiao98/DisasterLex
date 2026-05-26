# Responder Facility Risk
Assess the direct hazard risk to first responder facilities including EOCs, Fire, and Law Enforcement.

## Objective
Ensure the continuity of emergency operations by identifying facilities at risk of flooding or surge.

## Key Datasets
- `HIFLD-EMERGENC-FIRE_EMS-N`
- `HIFLD-EMERGENC-LOCAL_EOC-N`
- `HIFLD-EMERGENC-STATE_EOC-N`
- `HIFLD-EMERGENC-LOCAL_LAW-N`
- `HP_FLD_002` (NRI Flood)
- `HP_HUR_002` (Surge)

## Core Logic
1. Query the locations of all EOC, Fire, and Law facilities in the analysis area.
2. Join with \`HP_FLD_002\` to get \`nri_riverine_flood_score\`.
3. Join with \`HP_HUR_002\` to find facilities in high \`ve_ae_fraction\` surge zones.
4. Flag any facility where \`nri_riverine_flood_score > 75\` or surge fraction > 0.5.
5. If Criticality >= 3: Recommend facility relocation or temporary flood shielding for active response hubs.
