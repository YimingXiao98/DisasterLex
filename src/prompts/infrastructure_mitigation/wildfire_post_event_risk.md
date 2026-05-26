# Wildfire Post-Event Risk
Assess regional wildfire severity and supporting infrastructure exposure after or during a major fire.

## Objective
Compute regional wildfire risk across the full named footprint and pair it with population or infrastructure exposure totals.

## Key Datasets
- `HP_WFIR_001` (NRI Wildfire)
- `EX_POP_001`
- `HIFLD-ENERGY-SUBSTN-N`
- `EX_BLD_001`
- `hex_county_state_zip_crosswalk`

## Core Logic
1. **Primary regional wildfire metric (required first):** Return `avg_nri_wildfire_score = ROUND(AVG(HP_WFIR_001.nri_wildfire_score), 2)` across the full named region. For multi-county regions, compute one regional aggregate, not county-by-county outputs.
2. **Population baseline:** Return `regional_population_total = SUM(EX_POP_001.population_per_hex)` for the same region. Use `population_per_hex`, not `SUM(population_7km)`.
3. If the user asks about critical infrastructure exposure, add supporting counts for substations, buildings, or other requested assets, but keep the regional wildfire score as the lead metric.
4. Do **NOT** substitute local percentile wildfire-zone counts when the question asks for regional wildfire severity.
5. If Criticality >= 3: recommend active fire-weather monitoring, utility coordination, and pre-positioning for infrastructure protection.
