# SAR Priority Zones
Identify priority zones for search and rescue (SAR) operations based on population exposure, social vulnerability, and flood risk.

## Objective
Map the highest-priority hexes for SAR deployment during an active flood event by intersecting flood exposure with social vulnerability indicators.

## Key Datasets
- `HP_FLD_002` — riverine flood risk score (0–100 percentile)
- `VUL_002` — social vulnerability index (sovi; exclude sovi = -999 missing values)
- `EX_POP_001` — population density (population_7km)
- `hex_county_state_zip_crosswalk` — geography filter

## Core Logic
1. Query `HP_FLD_002` for hexes in the analysis area with `nri_riverine_flood_score >= 50` (above-median flood exposure).
2. Join with `VUL_002` — flag hexes where `sovi > 0.7` (top 30% most vulnerable), excluding `sovi = -999`.
3. Count high-priority SAR hexes (flood_score >= 50 AND sovi > 0.7).
4. **Average population in flood-exposed hexes (required):** `SELECT ROUND(AVG(p.population_7km), 2) AS avg_pop_flood_exposed FROM EX_POP_001 p JOIN HP_FLD_002 f ON p.hex_id=f.hex_id JOIN hex_county_state_zip_crosswalk x ON p.hex_id=x.hex_id WHERE x.County='<County>' AND f.nri_riverine_flood_score >= 50` — MUST filter by `nri_riverine_flood_score >= 50`, NOT all hexes. Report this value explicitly.
5. Report: total SAR-priority hex count, average population in flood-exposed hexes, top 3 hexes by combined risk.
6. REQUIRED: Also run `SELECT COUNT(*) FROM VUL_002 v JOIN hex_county_state_zip_crosswalk x ON v.hex_id=x.hex_id WHERE x.County='<County>' AND v.sovi=-999` and report the count of hexes with missing SoVI data.
7. If Criticality >= 3: Recommend deploying SAR assets to top-priority hexes; stage swift-water teams at high-flood, high-vulnerability clusters.
