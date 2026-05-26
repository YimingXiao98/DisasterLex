# Shelter Placement
Identify viable locations for emergency shelters by cross-referencing shelter capacity with hazard exposure and accessibility.

## Objective
Optimize shelter network configuration during an active event to maximize capacity while minimizing risk to displaced populations.

## Key Datasets
- `HIFLD-EMERGENC-SHELTER-N`
- `HP_FLD_002`
- `TRANS-ROAD-CRIT-INDEX`
- `EX_POP_001`

## Core Logic
1. Query \`HIFLD-EMERGENC-SHELTER-N\` for hexes in the analysis area where shelter count > 0.
2. Cross-reference with \`HP_FLD_002\` — exclude hexes with \`nri_riverine_flood_score >= 75\` (top-quartile flood risk). *(HP_FLD_003/floodgenome is not available; using NRI riverine flood score as proxy.)*
3. Cross-reference with \`TRANS-ROAD-CRIT-INDEX\` — exclude hexes with \`road_crit_index\` in bottom 20% (inaccessible).
4. Estimate demand: Count hexes in analysis area, multiply \`AVG(population_7km)\` from \`EX_POP_001\` by 0.10.
5. Calculate gap: \`demand - (viable_shelter_hexes * 200)\`. If gap > 0, recommend opening more shelters.
6. If Criticality >= 3: Flag as "IMMEDIATE ACTION REQUIRED".
