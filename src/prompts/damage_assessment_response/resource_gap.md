# Resource Gap
Calculate the shortfall in emergency resources (fire, law enforcement, shelters) versus population needs.

## Objective
Identify critical resource gaps to inform EMAC mutual aid requests and state/federal resource allocation.

## Key Datasets
- `HIFLD-EMERGENC-FIRE_EMS-N`
- `HIFLD-EMERGENC-LOCAL_LAW-N`
- `HIFLD-EMERGENC-SHELTER-N`
- `EX_POP_001`
- `CR_001`

## Core Logic
1. Count `HIFLD-EMERGENC-FIRE_EMS-N` stations in the analysis area (`SUM(hifld_fire_and_emergency_medical_service_stations_fire_stations_ems_stations_n)`).
2. Count `HIFLD-EMERGENC-LOCAL_LAW-N` facilities.
3. Count `HIFLD-EMERGENC-SHELTER-N` shelter locations (`SUM(hifld_national_shelter_system_facilities_shelter_locations_n)`).
4. Use `EX_POP_001.population_7km` — calculate `AVG(population_7km)` for the area.
5. **Community resilience (required):** `SELECT ROUND(AVG(c.nri_cri_score), 2) AS avg_cri_score FROM CR_001 c JOIN hex_county_state_zip_crosswalk x ON c.hex_id=x.hex_id WHERE x.County='<AREA>'` — report as average community resilience index for the county. **CRITICAL: `nri_cri_score` is 0–100 (NOT normalized to 0–1). Report the raw value as-is (e.g., 73.36). Never divide by 100. Do NOT use `nri_cri_value` (raw dollar index).**
6. Calculate ratios: fire stations per 10k population, law enforcement per 10k, shelters per 10k.
7. Compare against standard benchmarks (NFPA: 1 station per 10k, shelters: 1 per 5k displaced).
8. If Criticality >= 3: Recommend EMAC (Emergency Management Assistance Compact) mutual aid request.
