"""Aggregate multi-seed ablation results into the table numbers reported in the paper.

Reads `results/<model>/<cond>_seed<N>.json` for seven base models, five
internal ablations (Table 2), and four external baselines (Table 1).
Computes mean +/- std of judge_score_raw (overall + per tier R/K/M/D)
across seeds, writes `results/aggregated.json`, and prints LaTeX-ready
rows for both Table 1 (external baselines) and Table 2 (internal ablations).
"""
from __future__ import annotations

import json
import statistics
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
RESULTS = PROJECT_ROOT / "results"

# Seven base models evaluated in the paper.
MODELS = [
    ("gemini-3.1-flash-lite-preview", "Gemini 3.1 Flash Lite Preview"),
    ("deepseek-v3.2",                  "DeepSeek V3.2"),
    ("qwen3.6-flash",                  "Qwen 3.6 Flash"),
    ("llama-3.1-8b",                   "Llama 3.1 8B"),
    ("qwen3-8b",                       "Qwen3 8B"),
    ("qwen3-32b",                      "Qwen3 32B"),
    ("llama-3.3-70b",                  "Llama 3.3 70B"),
]

# Internal ablation conditions (Table 2: Full + four ablations).
ABLATIONS = ["full", "no-routing", "no-plan", "react", "text-rag"]

# External baselines (Table 1).
BASELINES = ["lightrag-e2e", "hipporag", "reforce", "chess"]

COND_LABEL = {
    "full":          "Full system",
    "no-routing":    "No routing",
    "no-plan":       "No plan",
    "react":         "ReAct",
    "text-rag":      "Text-RAG",
    "lightrag-e2e": "LightRAG",
    "hipporag":      "HippoRAG 2",
    "reforce":       "ReFoRCE",
    "chess":         "CHESS",
}
TIERS = ["R", "K", "M", "D"]


def _find_seed_files(model_short: str, cond: str) -> dict[int, Path]:
    """Return {seed: path}. Files are named <cond>_seed<N>.json."""
    out: dict[int, Path] = {}
    for p in sorted((RESULTS / model_short).glob(f"{cond}_seed*.json")):
        try:
            seed = int(p.stem.split("_seed")[1])
        except (IndexError, ValueError):
            continue
        out[seed] = p
    return out


def _per_tier_mean(cases: list[dict]) -> dict[str, float | None]:
    out = {}
    for t in TIERS:
        scores = [
            c.get("judge_score_raw", c["judge_score"])
            for c in cases
            if c.get("tier") == t and c.get("judge_score") is not None
        ]
        out[t] = statistics.mean(scores) if scores else None
    return out


def _aggregate_cell(model_short: str, cond: str) -> dict:
    files = _find_seed_files(model_short, cond)
    overall = []
    per_tier_means: list[dict] = []
    seeds_sorted: list[int] = []
    file_paths: list[str] = []
    for seed in sorted(files):
        data = json.loads(files[seed].read_text())
        cases = data.get("cases", [])
        raws = [c.get("judge_score_raw", c["judge_score"]) for c in cases if c.get("judge_score") is not None]
        overall.append(statistics.mean(raws) if raws else None)
        per_tier_means.append(_per_tier_mean(cases))
        seeds_sorted.append(seed)
        file_paths.append(str(files[seed].relative_to(PROJECT_ROOT)))

    overall = [v for v in overall if v is not None]
    out = {
        "n_seeds":      len(overall),
        "overall_mean": statistics.mean(overall) if overall else None,
        "overall_std":  statistics.stdev(overall) if len(overall) >= 2 else 0.0,
        "seeds":        seeds_sorted,
        "files":        file_paths,
        "per_tier":     {},
    }
    for t in TIERS:
        vals = [m[t] for m in per_tier_means if m.get(t) is not None]
        out["per_tier"][t] = {
            "mean": statistics.mean(vals) if vals else None,
            "std":  statistics.stdev(vals) if len(vals) >= 2 else 0.0,
            "n":    len(vals),
        }
    return out


def _format_cell(cell: dict, model_label: str, cond_label: str) -> str:
    overall_mu = cell["overall_mean"]
    overall_sd = cell["overall_std"]
    if overall_mu is None:
        return f"  {model_label:30s} & {cond_label:14s} & ---"
    row = (f"  {model_label:30s} & {cond_label:14s} & "
           f"{overall_mu:.3f}\\,$\\pm$\\,{overall_sd:.3f}")
    for t in TIERS:
        pt = cell["per_tier"][t]
        if pt["mean"] is None:
            row += " & ---"
        else:
            row += f" & {pt['mean']:.3f}\\,$\\pm$\\,{pt['std']:.3f}"
    return row + " \\\\"


def main() -> None:
    table: dict[str, dict[str, dict]] = {}
    for model_short, _ in MODELS:
        table[model_short] = {}
        for cond in ABLATIONS + BASELINES:
            table[model_short][cond] = _aggregate_cell(model_short, cond)

    out_path = RESULTS / "aggregated.json"
    out_path.write_text(json.dumps(table, indent=2))
    print(f"Wrote {out_path}\n")

    # ─── Table 2: internal ablations ───────────────────────────────────────
    print("=== Table 2: Internal ablations (Full + 4 ablations) ===")
    for model_short, model_label in MODELS:
        for cond in ABLATIONS:
            print(_format_cell(table[model_short][cond], model_label, COND_LABEL[cond]))
        print("  \\midrule")

    # ─── Table 1: external baselines ───────────────────────────────────────
    print("\n=== Table 1: External baselines (DisasterLex + 4 baselines) ===")
    for model_short, model_label in MODELS:
        # DisasterLex row first (Full from ablation matrix)
        print(_format_cell(table[model_short]["full"], model_label, "DisasterLex"))
        for cond in BASELINES:
            print(_format_cell(table[model_short][cond], model_label, COND_LABEL[cond]))
        print("  \\midrule")


if __name__ == "__main__":
    main()
