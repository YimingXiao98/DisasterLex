# DisasterLex

Anonymous code release accompanying the EMNLP 2026 submission.

DisasterLex is a knowledge-graph-mediated agentic framework that inserts an
Expert Knowledge Graph (EKG) between natural-language queries and a relational
database. The EKG is bridged to the database by typed concept-to-schema edges;
a four-stage orchestration pipeline (criticality extraction, operationally-
motivated routing, causal-informed planning, tool-augmented execution) reduces
the schema context from the full schema to a per-query subset.

This repository contains the code, configuration, benchmark, and per-seed
result artifacts needed to reproduce every cell of Tables 1 and 2 in the paper.

## Repository layout

```
.
├── configs/
│   ├── graph/
│   │   ├── ekg_curated.json     # 107 concepts, 117 typed causal edges
│   │   └── ddcg.json            # 36 tables, 150 columns, 7 join rules
│   ├── benchmark/test.json      # 75-case test split (R/K/M/D tiers)
│   ├── text_rag/chunks.jsonl    # Text-RAG source corpus (n=2,577 chunks)
│   ├── regions.json             # County/area lookup for the case study
│   └── schema.yaml              # legacy concept-table schema (informational)
├── src/
│   ├── agent/                   # orchestrator, text-to-SQL agent,
│   │   │                        # graph/Text-RAG/LightRAG retrievers
│   │   └── ...
│   ├── graph/                   # ContextGraph (Neo4j wrapper), DDCG types
│   ├── prompts/                 # 18 routing templates across 3 clusters
│   └── config.py                # env-driven configuration singleton
├── scripts/
│   ├── run_benchmark.py         # main entry point (Full + 4 internal ablations)
│   ├── load_neo4j.py            # populate Neo4j from EKG + DDCG
│   ├── build_ddcg.py            # auto-introspect DDCG from DuckDB
│   ├── build_lightrag_index.py  # build the LightRAG index (used by both
│   │                            # the in-pipeline retriever and the
│   │                            # external LightRAG-e2e baseline)
│   ├── build_text_rag_index.py  # rebuild the Text-RAG ablation embeddings
│   ├── validate_gold_facts.py   # sanity-check benchmark gold against live DB
│   ├── rescore_results.py       # re-score a saved run without re-executing
│   ├── aggregate_results.py     # produce results/aggregated.json + LaTeX rows
│   ├── download_data.sh         # fetch DuckDB from Zenodo (4.7 GB)
│   └── baselines/               # external-baseline runners (see below)
│       ├── run_lightrag_baseline.py    # LightRAG end-to-end
│       ├── run_hipporag_baseline.py    # HippoRAG 2 end-to-end
│       ├── run_reforce_baseline.py     # ReFoRCE multi-agent text-to-SQL
│       ├── score_chess_results.py      # score CHESS pipeline outputs
│       ├── index_hipporag.py           # build HippoRAG index
│       ├── index_lightrag.py           # build LightRAG index for baseline mode
│       ├── triage_to_bird_format.py    # convert benchmark to BIRD format
│       ├── triage_ddcg_to_bird_descriptions.py
│       └── duckdb_to_sqlite.py         # convert DuckDB to SQLite for baselines
├── results/                     # per-seed result JSONs (see below)
├── environment.yml              # conda environment specification
├── .env.example                 # template for required environment variables
├── LICENSE                      # Apache-2.0
└── README.md
```

### `results/` layout

For each of seven base models (Gemini 3.1 Flash-Lite Preview, DeepSeek V3.2,
Qwen 3.6 Flash, Llama 3.1 8B, Qwen3 8B, Qwen3 32B, Llama 3.3 70B), the
release contains nine condition × three seeds = 27 result files:

```
results/<model>/
├── full_seed{1,2,3}.json          # Table 2 — Full pipeline
├── no-routing_seed{1,2,3}.json    # Table 2 — internal ablation
├── no-plan_seed{1,2,3}.json       # Table 2 — internal ablation
├── react_seed{1,2,3}.json         # Table 2 — internal ablation
├── text-rag_seed{1,2,3}.json      # Table 2 — internal ablation
├── lightrag-e2e_seed{1,2,3}.json  # Table 1 — external baseline (LightRAG e2e)
├── hipporag_seed{1,2,3}.json      # Table 1 — external baseline (HippoRAG 2)
├── reforce_seed{1,2,3}.json       # Table 1 — external baseline (ReFoRCE)
└── chess_seed{1,2,3}.json         # Table 1 — external baseline (CHESS)
```

`results/aggregated.json` is the canonical mean ± std summary across seeds,
produced by `scripts/aggregate_results.py`. One cell is permanently missing
(qwen3-8b / reforce / seed 2 exceeded the OpenRouter 1M-token context window
on the multi-agent self-refinement loop; documented in Table 1 footnote).

## Installation

```bash
# 1. Create the Python environment
conda env create -f environment.yml
conda activate disasterlex
```

```bash
# 2. Start Neo4j (any 5.x image will do)
docker run -d --name disasterlex-neo4j \
  -p 7474:7474 -p 7687:7687 \
  -e NEO4J_AUTH=neo4j/password \
  neo4j:5
```

```bash
# 3. Download the DuckDB backend (~4.7 GB, anonymous Zenodo)
#    Edit scripts/download_data.sh to set ZENODO_URL before running.
bash scripts/download_data.sh
```

```bash
# 4. Set environment variables
cp .env.example .env
# Edit .env to add OPENROUTER_API_KEY (and DASHSCOPE_API_KEY if running Qwen 3.6 Flash).
```

```bash
# 5. Populate Neo4j from the EKG + DDCG JSON files
python scripts/load_neo4j.py
```

## Reproducing Tables 1 and 2

The release ships per-seed result files for all reported cells. To reproduce
the table numbers without re-running the pipeline, just invoke the aggregator:

```bash
python scripts/aggregate_results.py
```

The script writes `results/aggregated.json` and prints LaTeX-ready rows for
both Table 2 (internal ablations) and Table 1 (external baselines).

### Re-running internal ablations from scratch

Each (model, condition) cell is run three times with seeds 1–3. With seven
base models and five internal conditions, this is 105 cells (~60–120 min per
condition with `--parallel 3`).

```bash
MODELS=(
  google/gemini-3.1-flash-lite-preview
  deepseek/deepseek-v3.2
  qwen3.6-flash
  meta-llama/llama-3.1-8b-instruct
  qwen/qwen3-8b
  qwen/qwen3-32b
  meta-llama/llama-3.3-70b-instruct
)
for MODEL in "${MODELS[@]}"; do
  for COND in full no-routing no-plan react text-rag; do
    for SEED in 1 2 3; do
      python scripts/run_benchmark.py \
        --pipeline-model "$MODEL" \
        --ablation "$COND" \
        --seed "$SEED" \
        --parallel 3 \
        --extractor-model google/gemini-2.5-flash
    done
  done
done
```

Notes on per-model quirks:

- **Llama 3.1 8B** requires `response_format=json_object` on Stage 1/2 JSON
  prompts (handled automatically by `src/agent/orchestrator.py`).
- **Qwen3 32B ReAct** benefits from a 600-second per-case execute budget:
  ```bash
  export REACT_TIMEOUT=600
  ```

### Re-running external baselines

Each external baseline (LightRAG end-to-end, HippoRAG 2, ReFoRCE, CHESS) has
a runner under `scripts/baselines/`. All four use the same 75-case test split
and the same TRIAGE claim-extraction + reasoning judge as the main pipeline,
so the numbers in `results/<model>/<baseline>_seed{1,2,3}.json` plug straight
into `aggregate_results.py`.

```bash
# LightRAG (end-to-end mode)
python scripts/baselines/index_lightrag.py       # one-time, ~20–40 min
python scripts/baselines/run_lightrag_baseline.py \
  --pipeline-model google/gemini-3.1-flash-lite-preview --seed 1 \
  --output results/gemini-3.1-flash-lite-preview/lightrag-e2e_seed1.json

# HippoRAG 2
python scripts/baselines/index_hipporag.py       # one-time, ~30–60 min
python scripts/baselines/run_hipporag_baseline.py \
  --pipeline-model google/gemini-3.1-flash-lite-preview --seed 1 \
  --output results/gemini-3.1-flash-lite-preview/hipporag_seed1.json

# ReFoRCE (multi-agent text-to-SQL; requires the SQLite-converted DB)
python scripts/baselines/duckdb_to_sqlite.py     # one-time conversion
python scripts/baselines/run_reforce_baseline.py \
  --pipeline-model google/gemini-3.1-flash-lite-preview --seed 1 \
  --output results/gemini-3.1-flash-lite-preview/reforce_seed1.json

# CHESS (clone https://github.com/ShayanTalaei/CHESS into external/chess/,
# install its requirements, then run from the CHESS repo against the
# BIRD-formatted benchmark dump produced by triage_to_bird_format.py.
# Score the CHESS run-dir back through the TRIAGE harness:)
python scripts/baselines/score_chess_results.py \
  --chess-run-dir external/chess/results/dev/<setting>/<dataset>/<timestamp>/ \
  --pipeline-model google/gemini-3.1-flash-lite-preview --seed 1 \
  --output results/gemini-3.1-flash-lite-preview/chess_seed1.json
```

### Re-running the Text-RAG ablation

Build the chunk embeddings (~5 minutes) before the first run:

```bash
python scripts/build_text_rag_index.py
```

## DuckDB build details

The 4.7 GB DuckDB backing the case study is built from public-domain U.S.
federal data sources:

- per-hex hazard scores from FEMA NRI (riverine flood, hurricane, tornado, wildfire)
- exposure and population from US Census + ACS
- social-vulnerability indices from CDC/ATSDR SVI
- community resilience from FEMA NRI CRI
- facility inventories from HIFLD (hospitals, fire stations, shelters, power plants)

All sources are resampled to an H3 resolution-8 hexagonal grid (~0.74 km² per
cell) and joined on a shared `hex_id` key. The hosted DuckDB binary is the
output of this preprocessing pass.

## License

Apache 2.0 — see [LICENSE](LICENSE). The benchmark cases, EKG, and DDCG are
released under the same license.

## Anonymity note

This repository is hosted on an anonymous mirror for double-blind review. Code,
data, and result artifacts will be re-released under their canonical names
upon acceptance.
