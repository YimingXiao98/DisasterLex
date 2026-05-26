# EJ Resilience Gap
Identify counties where environmental-justice pressures are amplified by weak community resilience.

## Objective
Count and summarize counties where county-level social vulnerability is high and county-level community resilience is low.

## Key Datasets
- `VUL_002` (SoVI)
- `CR_001` (Community Resilience)
- `hex_county_state_zip_crosswalk`

## Core Logic
1. Build county-level aggregates across Texas: `avg_sovi = AVG(VUL_002.sovi)` excluding `-999`, and `avg_cri = AVG(CR_001.nri_cri_score)`.
2. Compute statewide county-level percentile thresholds with `QUANTILE_CONT`: SoVI 75th percentile and CRI 25th percentile.
3. Return one statewide aggregate `high_sovi_low_cri_county_count = COUNT(*)` for counties where `avg_sovi > sovi_p75` and `avg_cri < cri_p25`. This is a county-count query, not a hex-count query.
4. In the written analysis, explicitly state that low community resilience reduces emergency response capacity and therefore amplifies EJ exposure.
5. Recommend equity-weighted pre-disaster mitigation funding for the qualifying counties.
