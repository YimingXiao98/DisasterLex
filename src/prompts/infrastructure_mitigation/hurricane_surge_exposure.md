# Hurricane Surge Exposure
Identify coastal surge zones and the critical infrastructure exposed to hurricane conditions.

## Objective
Assess the risk of storm surge to energy, water, healthcare, and economic assets in the coastal zone.

## Key Datasets
- `HP_HUR_001`
- `HP_HUR_002`
- `HIFLD-ENERGY-SUBSTN-N` (substations, column: `hifld_energy_substations_n`)
- `HIFLD-WATER-WTP-N` (water treatment plants, column: `hifld_water_wtp_n`)
- `EX_LIFE_004` (hospitals, column: `hosp_n > 0`)
- `EX_BLD_002`

## Core Logic
Run exactly these 3 queries. Use `<AREA>` = exact county name with "County" suffix.

1. **Surge + infrastructure exposure (CTE — hurricane tables only):**
```sql
WITH county_surge AS (
    SELECT h.hex_id FROM HP_HUR_002 h
    JOIN hex_county_state_zip_crosswalk x ON h.hex_id = x.hex_id
    WHERE x.County = '<AREA>' AND h.ve_ae_fraction > 0.5
)
SELECT
    (SELECT COUNT(*) FROM county_surge) AS surge_exposed_hexes,
    (SELECT COUNT(*) FROM "HIFLD-ENERGY-SUBSTN-N" e JOIN county_surge cs ON e.hex_id=cs.hex_id WHERE e.hifld_energy_substations_n > 0) AS substation_hexes_in_surge,
    (SELECT COUNT(*) FROM "HIFLD-WATER-WTP-N" w JOIN county_surge cs ON w.hex_id=cs.hex_id WHERE w.hifld_water_wtp_n > 0) AS wtp_hexes_in_surge
```
Report all three values explicitly.

2. **Hospital count (separate query — hospitals table only):**
```sql
SELECT COUNT(*) AS hospital_hexes FROM EX_LIFE_004 h
JOIN hex_county_state_zip_crosswalk x ON h.hex_id = x.hex_id
WHERE x.County = '<AREA>' AND h.hosp_n > 0
```
Report as total hospital hexes in the county.

3. **Strike rate + economic exposure:**
```sql
SELECT
    (SELECT AVG(hurr_strike_rate_10y) FROM HP_HUR_001 hr JOIN hex_county_state_zip_crosswalk x ON hr.hex_id=x.hex_id WHERE x.County='<AREA>') AS avg_strike_rate,
    (SELECT SUM(b.hEE) FROM EX_BLD_002 b JOIN HP_HUR_002 h ON b.hex_id=h.hex_id JOIN hex_county_state_zip_crosswalk x ON b.hex_id=x.hex_id WHERE h.ve_ae_fraction > 0.5 AND x.County='<AREA>') AS total_economic_exposure
```

4. If Criticality >= 4: Recommend immediate facility shut-down protocols.
