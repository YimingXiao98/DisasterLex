# Population Impact
Quantify the number of affected people and assess their social vulnerability and resilience.

## Objective
Prioritize humanitarian aid and recovery resources based on the intersection of hazard impact and community vulnerability.

## Key Datasets
- `EX_POP_001`
- `VUL_001`
- `VUL_002`
- `CR_001`

## Core Logic
1. **Hazard exposure baseline (required when a specific hazard is mentioned):** Count hexes at high hazard risk using the standard threshold — `nri_riverine_flood_score >= 75` for flood, `nri_tornado_score >= 75` for tornado, `ve_ae_fraction > 0` for hurricane. Also count hexes where `sovi != -999` (valid SoVI) within the hazard-exposed set. Report both as geographic context.
2. Use \`EX_POP_001\` — calculate \`AVG(population_7km)\` for hexes in the analysis area (NOT SUM — overlapping radii).
3. Count total hexes in the analysis area and multiply by hex area (~0.74 km²) for geographic coverage.
4. Join with \`VUL_002\` — identify hexes where \`sovi > 0\` (valid SoVI, excluding \`sovi = -999\`) within the high-hazard zone. Also flag the subset with \`sovi > 0.8\` (top 20% social vulnerability) for priority targeting.
5. Join with \`VUL_001\` — identify hexes with low \`median_income\` (low-income areas — note: column is `median_income` in raw dollars, not `inv_median_income`).
6. Join with \`CR_001\` — identify hexes with \`nri_cri_score < 50\` (low community resilience).
7. If Criticality >= 3: Recommend prioritizing low-resilience, high-vulnerability hexes for aid distribution.
