# Essential Resource Access
Analyze the proximity and availability of essential lifelines (groceries, pharmacies, fuel) for at-risk populations.

## Objective
Identify "resource deserts" during an active event where vulnerable populations lack access to essential supplies.

## Key Datasets
- `EX_LIFE_001` (Groceries)
- `EX_LIFE_002` (Pharmacies)
- `EX_LIFE_003` (Fuel)
- `EX_POP_001`
- `VUL_002` (SoVI)
- `CRIT_LIFE_001` (Lifeline RAC)

## Core Logic
1. Identify hexes in the analysis area with high social vulnerability (\`sovi > 0.8\`).
2. Query \`EX_LIFE_001\`, \`EX_LIFE_002\`, and \`EX_LIFE_003\` for these vulnerable hexes.
3. Flag hexes where \`groc_per1k\`, \`pharm_per10k\`, or \`fuel_per10k\` is 0.
4. Cross-reference with \`CRIT_LIFE_001\` — prioritize hexes where \`hex_fc_rac_grocery\` is high but actual count is low (dependence on few facilities).
5. If Criticality >= 3: Recommend pre-positioning mobile resource units (PODs) in these deserts.
