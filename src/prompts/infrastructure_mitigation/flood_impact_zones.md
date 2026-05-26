# Flood Impact Zones
Identify areas where high flood hazard overlaps with dense critical infrastructure.

## Objective
Prioritize infrastructure protection and pre-positioning based on the intersection of flooding and CID (Critical Infrastructure Density).

## Key Datasets
- `HP_FLD_002`
- `EX_INF_001`
- `HIFLD-ENERGY-SUBSTN-N`
- `HIFLD-WATER-WTP-N`

## Core Logic
*(HP_FLD_003/floodgenome is not available. Use \`HP_FLD_002.nri_riverine_flood_score >= 75\` as the high-risk threshold — top-quartile NRI riverine flood score — as proxy.)*
1. Query \`HP_FLD_002\` for hexes where \`nri_riverine_flood_score >= 75\` in the analysis area.
2. Add \`nri_coastal_flood_score\` and \`nri_riverine_flood_score\` from \`HP_FLD_002\` for severity context.
3. Join with \`EX_INF_001\` — identify hexes where \`CID > 0.5\` (high critical infrastructure density) AND \`nri_riverine_flood_score >= 75\`.
4. Join with \`HIFLD-ENERGY-SUBSTN-N\` — count substations in flood-exposed hexes.
5. Join with \`HIFLD-WATER-WTP-N\` — count water treatment plants in flood-exposed hexes.
6. Rank results by \`CID * (nri_riverine_flood_score / 100.0)\` (composite exposure score).
7. If Criticality >= 3: Recommend pre-positioning backup generators and sandbagging.
