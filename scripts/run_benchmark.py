"""
DisasterLex benchmark runner.

Evaluates the four-stage pipeline (extract → classify → plan → execute) on the
75-case dev split, with one of six condition flags (full, no-plan, no-routing,
react, text-rag, lightrag).

Usage:
    python scripts/run_benchmark.py --ablation full --parallel 3
    python scripts/run_benchmark.py --ablation no-plan --parallel 3
    python scripts/run_benchmark.py --ablation text-rag --parallel 3
    python scripts/run_benchmark.py --cases draft_k13 draft_m18
"""
from __future__ import annotations

import argparse
import concurrent.futures
import json
import logging
import os
import re
import sys
from datetime import datetime
from pathlib import Path
from typing import Any

from dotenv import load_dotenv

load_dotenv()
logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parent.parent
_DEFAULT_LLM_MODEL = os.getenv("LLM_MODEL", "google/gemini-3.1-flash-lite-preview")
CONFIG_DIR = PROJECT_ROOT / "configs"
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

# ── Config paths ──────────────────────────────────────────────────────────────

INCIDENT_BENCHMARK_PATH = CONFIG_DIR / "benchmark" / "test.json"

# Legacy ReAct benchmark paths (--mode react)
TIER1_PATH = CONFIG_DIR / "benchmark" / "benchmark_concept_matching.json"
TIER2_PATH = CONFIG_DIR / "benchmark" / "benchmark_sql.json"
TIER3_LEGACY_PATH = CONFIG_DIR / "benchmark" / "benchmark_e2e.json"
TIER3_GROUNDED_PATH = CONFIG_DIR / "benchmark" / "benchmark_e2e_grounded.json"


# ── Tier 1: Concept Matching ──────────────────────────────────────────────────

def run_tier1(cases: list[dict], verbose: bool = False) -> dict:
    """
    Tests context_graph.match_concepts() against known synonym→concept pairs.
    Deterministic — no LLM or DB calls.
    """
    from src.graph.context_graph import get_context_graph

    graph = get_context_graph()
    results = []
    passed = 0

    for case in cases:
        cid = case["id"]
        query = case["query"]
        expected = set(case["expected_concepts"])
        must_include_all = case.get("must_include_all", True)

        try:
            matched = graph.match_concepts(query)
            returned_ids = {m["id"] for m in matched}

            if must_include_all:
                ok = expected.issubset(returned_ids)
            else:
                ok = bool(expected & returned_ids)

            # Precision / recall per case
            tp = len(expected & returned_ids)
            fp = len(returned_ids - expected)
            fn = len(expected - returned_ids)
            precision = tp / (tp + fp) if (tp + fp) > 0 else (1.0 if not expected else 0.0)
            recall = tp / (tp + fn) if (tp + fn) > 0 else 1.0
            f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0

            if ok:
                passed += 1

            result = {
                "id": cid,
                "query": query,
                "expected": sorted(expected),
                "returned": sorted(returned_ids),
                "passed": ok,
                "precision": round(precision, 3),
                "recall": round(recall, 3),
                "f1": round(f1, 3),
            }
            if verbose or not ok:
                logger.info(f"[T1] {cid}: {'PASS' if ok else 'FAIL'} | expected={sorted(expected)} returned={sorted(returned_ids)}")

        except Exception as e:
            result = {"id": cid, "query": query, "passed": False, "error": str(e)}
            logger.error(f"[T1] {cid}: ERROR — {e}")

        results.append(result)

    n = len(cases)
    pass_rate = passed / n if n > 0 else 0.0
    avg_f1 = sum(r.get("f1", 0.0) for r in results) / n if n > 0 else 0.0

    print(f"\nTier 1 — Concept Matching: {passed}/{n} passed ({pass_rate:.0%}) | avg F1={avg_f1:.3f}")
    return {"pass_rate": round(pass_rate, 3), "avg_f1": round(avg_f1, 3), "passed": passed, "total": n, "cases": results}


# ── Tier 2: SQL Agent ─────────────────────────────────────────────────────────

def _check_result(check: dict, state: dict) -> tuple[bool, str]:
    """Validate the SQL agent result against the specified check type."""
    result = state.get("result") or []
    columns = state.get("columns") or []
    error = state.get("error")

    if error and not result:
        return False, f"execution error: {error}"

    check_type = check.get("type")

    if check_type == "non_empty":
        ok = len(result) > 0
        return ok, "" if ok else "result is empty"

    if check_type == "row_count":
        expected_n = check["value"]
        actual_n = len(result)
        ok = actual_n == expected_n
        return ok, "" if ok else f"expected {expected_n} rows, got {actual_n}"

    if check_type == "value_range":
        col_idx = check.get("column", 0)
        if not result:
            return False, "no rows returned"
        try:
            # column can be an index or a name
            if isinstance(col_idx, int):
                val = float(result[0][col_idx])
            else:
                col_idx_num = columns.index(col_idx) if col_idx in columns else 0
                val = float(result[0][col_idx_num])
            ok = check["min"] <= val <= check["max"]
            return ok, "" if ok else f"value {val} outside range [{check['min']}, {check['max']}]"
        except (IndexError, ValueError, TypeError) as e:
            return False, f"value check error: {e}"

    if check_type == "column_present":
        required = set(check.get("columns", []))
        returned = set(columns)
        missing = required - returned
        ok = len(missing) == 0
        return ok, "" if ok else f"missing columns: {missing}"

    return True, "unknown check type (skipped)"


def run_tier2(cases: list[dict], verbose: bool = False) -> dict:
    """
    Tests text_to_sql_agent.run() on each question.
    Checks: SQL execution, keyword presence, result correctness.
    """
    sys.path.insert(0, str(PROJECT_ROOT))
    from src.agent.text_to_sql_agent import run as sql_run

    results = []
    exec_success = 0
    result_check_passed = 0
    keyword_passed = 0

    for case in cases:
        cid = case["id"]
        question = case["question"]
        expected_tables = case.get("expected_tables", [])
        expected_keywords = case.get("expected_sql_keywords", [])
        result_check = case.get("result_check", {"type": "non_empty"})

        try:
            state = sql_run(question)
            sql = state.get("sql", "") or ""
            error = state.get("error")
            executed = bool(state.get("result") is not None or not error)

            if executed and not error:
                exec_success += 1

            # Keyword check
            sql_upper = sql.upper()
            kw_hits = [kw for kw in expected_keywords if kw.upper() in sql_upper]
            kw_ok = len(kw_hits) == len(expected_keywords)
            if kw_ok:
                keyword_passed += 1

            # Table check (subset of keyword check)
            table_hits = [t for t in expected_tables if t.upper().replace("-", "_") in sql_upper.replace("-", "_") or t in sql]

            # Result check
            rc_ok, rc_msg = _check_result(result_check, state)
            if rc_ok:
                result_check_passed += 1

            overall_pass = executed and (not error) and kw_ok and rc_ok

            result = {
                "id": cid,
                "question": question,
                "passed": overall_pass,
                "sql_executed": executed,
                "sql_error": error,
                "generated_sql": sql,
                "keyword_check": {"expected": expected_keywords, "hits": kw_hits, "passed": kw_ok},
                "table_check": {"expected": expected_tables, "found": table_hits},
                "result_check": {"spec": result_check, "passed": rc_ok, "message": rc_msg},
                "retry_count": state.get("retry_count", 0),
                "total_rows": state.get("total_rows", 0),
                "difficulty": case.get("difficulty", "unknown"),
            }
            if verbose or not overall_pass:
                status = "PASS" if overall_pass else "FAIL"
                logger.info(f"[T2] {cid} ({case.get('difficulty','?')}): {status} | exec={executed} kw={kw_ok} rc={rc_ok} | {rc_msg or ''}")

        except Exception as e:
            result = {"id": cid, "question": question, "passed": False, "error": str(e)}
            logger.error(f"[T2] {cid}: EXCEPTION — {e}")

        results.append(result)

    n = len(cases)
    print(
        f"\nTier 2 — SQL Agent: "
        f"exec={exec_success}/{n} ({exec_success/n:.0%}) | "
        f"keywords={keyword_passed}/{n} ({keyword_passed/n:.0%}) | "
        f"result_check={result_check_passed}/{n} ({result_check_passed/n:.0%})"
    )
    return {
        "sql_exec_rate": round(exec_success / n, 3),
        "keyword_rate": round(keyword_passed / n, 3),
        "result_check_rate": round(result_check_passed / n, 3),
        "total": n,
        "cases": results,
    }


# ── Tier 3: End-to-End Pipeline ───────────────────────────────────────────────

def _strip_json_fences(raw: str) -> str:
    cleaned = raw.strip()
    cleaned = re.sub(r"^```(?:json)?\s*", "", cleaned)
    cleaned = re.sub(r"\s*```$", "", cleaned)
    return cleaned


def _load_json_response(response: Any, fallback: dict[str, Any]) -> dict[str, Any]:
    try:
        raw = _strip_json_fences(str(response.content))
        return json.loads(raw)
    except (json.JSONDecodeError, Exception) as e:
        logger.warning(
            "Judge parse error: %s | raw=%s",
            e,
            str(getattr(response, "content", "N/A"))[:200],
        )
        return fallback | {"reason": f"parse error: {e}"}


def _normalize_text(value: str) -> str:
    return re.sub(r"[^a-z0-9]+", " ", value.lower()).strip()


def _make_llm(model_name: str) -> Any:
    from langchain_openai import ChatOpenAI
    from src.config import provider_default_headers, provider_request_kwargs, resolve_provider

    base_url, api_key = resolve_provider(model_name)
    kwargs = dict(
        model=model_name,
        openai_api_key=api_key,
        openai_api_base=base_url,
        temperature=0.0,
        timeout=120,
    )
    kwargs.update(provider_request_kwargs(model_name, base_url))
    return ChatOpenAI(**kwargs, default_headers=provider_default_headers(base_url, "Disaster Benchmark Evaluator"))


def _judge_answer_legacy(judge_llm: Any, prompt_template: str, case: dict, system_answer: str) -> dict:
    """Call the legacy Tier 3 LLM judge against prose gold answers."""
    from langchain_core.messages import HumanMessage

    rubric_text = "\n".join(f"- {r}" for r in case.get("rubric", []))
    prompt = (
        prompt_template
        .replace("{question}", case["question"])
        .replace("{gold_answer}", case["gold_answer"])
        .replace("{rubric}", rubric_text)
        .replace("{system_answer}", system_answer)
    )
    response = judge_llm.invoke([HumanMessage(content=prompt)])
    return _load_json_response(
        response,
        {"score": 1, "rubric_hits": [], "rubric_misses": case.get("rubric", [])},
    )


def _empty_claims() -> dict[str, Any]:
    return {
        "entities": [],
        "ordered_entities": [],
        "numeric_claims": [],
        "boolean_claims": [],
        "causal_relation_tokens": [],
        "causal_claims": [],
        "recommendations": [],
    }


_STRUCTURED_FACT_LINE_RE = re.compile(
    r"^-+\s*subject:\s*(?P<subject>.+?)\s+metric:\s*(?P<metric>.+?)\s+value:\s*(?P<value>.+?)\s*$",
    re.IGNORECASE,
)


def _parse_numeric_literal(raw: str) -> float | None:
    cleaned = raw.strip().rstrip(".").replace(",", "")
    if cleaned.endswith("%"):
        cleaned = cleaned[:-1].strip()
    if not cleaned:
        return None
    try:
        return float(cleaned)
    except ValueError:
        return None


def _extract_structured_numeric_claims(system_answer: str) -> list[dict[str, Any]]:
    claims: list[dict[str, Any]] = []
    in_block = False
    saw_fact = False
    for raw_line in system_answer.splitlines():
        line = raw_line.strip()
        if not in_block:
            if line.lower().startswith("structured facts:"):
                in_block = True
            continue

        if not line:
            continue
        if line.startswith("("):
            continue

        match = _STRUCTURED_FACT_LINE_RE.match(line)
        if match:
            saw_fact = True
            value = _parse_numeric_literal(match.group("value"))
            if value is None:
                continue
            claims.append(
                {
                    "subject": match.group("subject").strip(),
                    "metric": match.group("metric").strip(),
                    "value": value,
                }
            )
            continue

        if saw_fact:
            break

    return claims


def _merge_numeric_claims(
    preferred: list[dict[str, Any]],
    fallback: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    merged: list[dict[str, Any]] = []
    seen: set[tuple[str, str, str]] = set()
    for claim in preferred + fallback:
        key = (
            _normalize_text(str(claim.get("subject", ""))),
            _normalize_text(str(claim.get("metric", ""))),
            str(claim.get("value")),
        )
        if key in seen:
            continue
        seen.add(key)
        merged.append(claim)
    return merged


def _collect_extraction_targets(case: dict) -> dict[str, Any]:
    targets = {
        "entities": [],
        "metrics": [],
        "booleans": [],
        "causal_relation_tokens": [],
    }
    for check in case.get("checks", []):
        kind = check.get("kind")
        if kind in {"entity_set_match", "ordered_topk_match"}:
            targets["entities"].extend(str(value) for value in check.get("expected", []))
            if check.get("claim_field") == "causal_relation_tokens":
                targets["causal_relation_tokens"].extend(str(value) for value in check.get("expected", []))
        elif kind == "numeric_value_match":
            targets["entities"].append(str(check.get("subject", "")))
            targets["metrics"].append({
                "id": str(check.get("metric", "")),
                "subject": str(check.get("subject", "")),
            })
        elif kind == "boolean_match":
            targets["booleans"].append(
                {
                    "name": check.get("name", ""),
                    "description": check.get("description", ""),
                }
            )
    targets["entities"] = sorted({value for value in targets["entities"] if value})
    # Deduplicate metrics by id, preserving subject hint
    seen_metric_ids: set[str] = set()
    deduped_metrics = []
    for m in targets["metrics"]:
        mid = m["id"] if isinstance(m, dict) else m
        if mid and mid not in seen_metric_ids:
            seen_metric_ids.add(mid)
            deduped_metrics.append(m)
    targets["metrics"] = sorted(deduped_metrics, key=lambda m: m["id"] if isinstance(m, dict) else m)
    targets["causal_relation_tokens"] = sorted(
        {value for value in targets["causal_relation_tokens"] if value}
    )
    return targets


def _extract_case_claims(extractor_llm: Any, case: dict, system_answer: str) -> dict[str, Any]:
    from langchain_core.messages import HumanMessage

    targets = _collect_extraction_targets(case)
    structured_numeric_claims = _extract_structured_numeric_claims(system_answer)
    prompt = f"""
Extract structured claims from the system answer for a disaster benchmark case.

Question: {case["question"]}
Category: {case.get("category", "unknown")}
System answer:
{system_answer}

Normalization targets:
- entities: {json.dumps(targets["entities"])}
- metrics: {json.dumps(targets["metrics"])}
- booleans: {json.dumps(targets["booleans"])}
- causal_relation_tokens: {json.dumps(targets["causal_relation_tokens"])}

Return valid JSON only with this schema:
{{
  "entities": ["..."],
  "ordered_entities": ["..."],
  "numeric_claims": [
    {{"subject": "...", "metric": "...", "value": 0}}
  ],
  "boolean_claims": [
    {{"name": "...", "value": true}}
  ],
  "causal_relation_tokens": ["source|REL|target"],
  "causal_claims": [
    {{"source": "...", "relation": "...", "target": "..."}}
  ],
  "recommendations": ["..."]
}}

Rules:
- Extract only claims explicitly present or clearly implied by the system answer.
- Normalize entity labels and metric ids to the provided targets when they fit exactly.
- If a field has no claims, return an empty list.
- Do not invent numeric values.
- If the answer contains a STRUCTURED FACTS block, treat those lines as authoritative and
  preserve their subject / metric / value triples exactly when they match the targets.
- For statewide or dataset-wide aggregate numeric claims with no specific entity subject
  (e.g. "the dataset has X", "statewide total is X", "Texas has X hexes"), use "dataset"
  as the subject in numeric_claims.
- For numeric_claims, each metric entry in the targets includes an "id" and "subject". Use
  the subject hint to select the correct numeric value when the answer reports multiple
  numbers — pick the value most closely associated with that subject/entity. Use the metric
  "id" as the metric field in your output.
- For boolean targets, decide semantically from the provided description even if the answer
  does not use the literal target name. When the answer materially supports a target,
  emit a boolean_claim using the EXACT provided "name".
- In causal_relation_tokens, use ONLY 3-part tokens "source|REL|target" — do NOT add a
  4th condition segment. Use ONLY these exact canonical concept IDs: impervious, runoff,
  flood_occurrence, flood_severity, elevation, stream_dist, drainage, hand, vulnerability,
  community_resilience, power_disruption, shelter_operations, hospital_operations,
  emergency_response_capacity. Do not rephrase or expand them.
  Examples of mapping natural language to canonical tokens:
  - "Impervious surface coverage increases stormwater runoff" → ["impervious|INCREASES|runoff"]
  - "Higher elevation reduces flood occurrence, but low drainage worsens flood severity"
    → ["elevation|REDUCES|flood_occurrence", "drainage|INCREASES|flood_severity"]
  - "Floods cascade to power disruption, degrading hospital operations"
    → ["flood_occurrence|INCREASES|power_disruption", "power_disruption|INCREASES|hospital_operations"]
- Split each distinct recommended action into a separate entry in "recommendations".
  Do not bundle multiple actions into one string.
- "ordered_entities": populate this with any ranked/prioritized/ordered list of named
  geographic entities (counties, zip codes, states) mentioned in the answer — in the
  order they are mentioned. This includes "gap counties", "top counties", "priority
  counties", "at-risk areas", "ranked hexes", etc. Use exact names from the entities
  target list when they match. Include ALL entities from the answer that match the target
  list, even if only briefly mentioned.
""".strip()
    response = extractor_llm.invoke([HumanMessage(content=prompt)])
    claims = _load_json_response(response, _empty_claims())
    claims = _empty_claims() | claims
    claims["numeric_claims"] = _merge_numeric_claims(
        structured_numeric_claims,
        claims.get("numeric_claims", []),
    )
    return claims


def _get_scorer_commit_sha() -> str:
    """Return the current git HEAD SHA of the scorer/repo, or empty string if not in git."""
    import subprocess
    try:
        out = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=Path(__file__).resolve().parent.parent,
            capture_output=True, text=True, timeout=5,
        )
        sha = (out.stdout or "").strip()
        if sha:
            # Mark whether the working tree is dirty so reruns from a dirty
            # checkout are obviously not from the tagged commit.
            try:
                dirty = subprocess.run(
                    ["git", "status", "--porcelain"],
                    cwd=Path(__file__).resolve().parent.parent,
                    capture_output=True, text=True, timeout=5,
                )
                if (dirty.stdout or "").strip():
                    sha = sha + "-dirty"
            except Exception:
                pass
            return sha
    except Exception:
        pass
    return ""


def _derive_baseline_routing_state(
    extractor_llm: Any, case: dict, system_answer: str
) -> dict[str, Any]:
    """Post-hoc operational-frame classification for baselines without explicit routing.

    Baselines (ReAct, Text-RAG, LightRAG, CHESS) do not run the orchestrator's
    classify_cluster / extract_context_and_criticality nodes, so their
    routing_state is otherwise empty. To evaluate Tier R fairly — measuring
    whether the baseline's ANSWER aligned with the correct operational frame —
    we ask the same extractor model to classify the produced answer into
    cluster / query_type / criticality. This is not a measurement of the
    baseline's routing capability (it has none); it is a measurement of
    operational-frame alignment in the answer text. The paper should disclose
    this distinction.

    Returns a dict with keys {cluster, query_type, criticality} suitable for
    pipeline_state_match deterministic checks.
    """
    from langchain_core.messages import HumanMessage
    try:
        from src.agent.prompt_registry import CLUSTERS, get_cluster_names
        clusters = get_cluster_names()
        cluster_to_qts: dict[str, list[str]] = {}
        for cname in clusters:
            cluster_obj = CLUSTERS.get(cname)
            if cluster_obj and getattr(cluster_obj, "query_types", None):
                cluster_to_qts[cname] = list(cluster_obj.query_types.keys())
            else:
                cluster_to_qts[cname] = []
    except Exception:
        clusters = ["damage_assessment_response", "infrastructure_mitigation", "life_safety_operations"]
        cluster_to_qts = {c: [] for c in clusters}

    qt_lines = "\n".join(
        f"  {c}: {', '.join(qts) if qts else '(no specific subtypes)'}"
        for c, qts in cluster_to_qts.items()
    )

    truncated_answer = (system_answer or "")[:2000]
    prompt = f"""Classify the following disaster-response system answer into operational-frame fields.
This is a post-hoc classification of an answer produced by a baseline system that does
not have an explicit routing step. Pick the cluster/query_type that best matches the
PRIMARY operational frame the answer addresses.

Original user question:
{case.get('question', '')}

System answer (first 2000 chars):
{truncated_answer}

Cluster vocabulary:
- life_safety_operations: active rescue, evacuation, medical triage, SAR
- damage_assessment_response: post-event damage counts, economic loss, recovery planning, resource gap
- infrastructure_mitigation: protecting infrastructure (flood barriers, hardening, surge exposure)

Query types per cluster:
{qt_lines}

Criticality scale (1-5):
1-2: informational / preparedness
3: imminent / active response
4-5: life-safety mission (evacuation, shutdowns)

Return JSON only:
{{"cluster": "<one of the cluster names>", "query_type": "<one of the query types in that cluster>", "criticality": <int 1-5>}}
""".strip()
    try:
        response = extractor_llm.invoke([HumanMessage(content=prompt)])
        parsed = _load_json_response(response, {})
    except Exception:
        parsed = {}

    cluster = str(parsed.get("cluster", "")).strip()
    query_type = str(parsed.get("query_type", "")).strip()
    try:
        criticality = int(parsed.get("criticality", 3))
    except (TypeError, ValueError):
        criticality = 3
    if criticality < 1 or criticality > 5:
        criticality = 3
    return {
        "cluster": cluster,
        "query_type": query_type,
        "criticality": criticality,
        "_derivation": "post_hoc_answer_classification",
    }


def _find_numeric_claim(claims: dict[str, Any], subject: str, metric: str) -> float | None:
    subject_norm = _normalize_text(subject)
    metric_norm = _normalize_text(metric)

    def _try_float(v):
        try:
            return float(v)
        except (TypeError, ValueError):
            return None

    def _strip_token(text: str, token: str) -> str:
        if not token:
            return text
        stripped = re.sub(rf"\b{re.escape(token)}\b", " ", text)
        return re.sub(r"\s+", " ", stripped).strip()

    def _subject_matches(claim_subject: str) -> bool:
        # Exact, substring, or county-suffix-stripped match.
        # E.g., gold "kerr" should match claim "kerr_county" and vice versa.
        a = _strip_token(subject_norm, "county")
        b = _strip_token(claim_subject, "county")
        if a == b:
            return True
        return a in b or b in a

    # Pass 1: exact subject + exact metric match.
    for claim in claims.get("numeric_claims", []):
        claim_subject = _normalize_text(str(claim.get("subject", "")))
        claim_metric = _normalize_text(str(claim.get("metric", "")))
        if claim_subject == subject_norm and claim_metric == metric_norm:
            return _try_float(claim.get("value"))

    # Pass 2: subject substring match, exact metric match.
    for claim in claims.get("numeric_claims", []):
        claim_subject = _normalize_text(str(claim.get("subject", "")))
        claim_metric = _normalize_text(str(claim.get("metric", "")))
        if claim_metric == metric_norm and _subject_matches(claim_subject):
            return _try_float(claim.get("value"))

    # Pass 3: subject-tolerant + metric token-set Jaccard similarity ≥ 0.4.
    # Strips the subject token from both gold and claim metric (handles
    # "tornado_eal_potter" gold vs "tornado_eal" claim, or "high_fire_hexes_kerr"
    # gold vs "high_fire_risk_hexes" claim, and "hospital_flood_risk" gold vs
    # "high_risk_hospital_hex_count" claim).
    # Jaccard 0.4 is permissive on naming. The per-check `tolerance` field on
    # the gold check is the backstop: a lexically-similar claim with a wrong
    # value will still fail the tolerance comparison in _score_numeric_check,
    # so loose matching cannot create false-positive correct-value passes
    # unless an unrelated claim happens to also equal the gold value (rare).
    gold_metric_stripped = _strip_token(_strip_token(metric_norm, subject_norm), "county")
    gold_tokens = set(gold_metric_stripped.split()) if gold_metric_stripped else set()
    if gold_tokens:
        best_match = None
        best_jaccard = 0.0
        for claim in claims.get("numeric_claims", []):
            claim_subject = _normalize_text(str(claim.get("subject", "")))
            if not _subject_matches(claim_subject):
                continue
            claim_metric = _normalize_text(str(claim.get("metric", "")))
            claim_metric_stripped = _strip_token(
                _strip_token(claim_metric, subject_norm), "county"
            )
            claim_tokens = set(claim_metric_stripped.split())
            if not claim_tokens:
                continue
            inter = len(gold_tokens & claim_tokens)
            union = len(gold_tokens | claim_tokens)
            if union == 0:
                continue
            jaccard = inter / union
            if jaccard >= 0.4 and jaccard > best_jaccard:
                best_jaccard = jaccard
                best_match = claim
        if best_match is not None:
            return _try_float(best_match.get("value"))

    return None


def _find_boolean_claim(claims: dict[str, Any], name: str) -> bool | None:
    target = _normalize_text(name)
    for claim in claims.get("boolean_claims", []):
        if _normalize_text(str(claim.get("name", ""))) == target:
            value = claim.get("value")
            if isinstance(value, bool):
                return value
    return None


def _semantic_boolean_fallback(
    extractor_llm: Any,
    case: dict,
    check: dict,
    system_answer: str,
) -> bool | None:
    from langchain_core.messages import HumanMessage

    prompt = f"""
Decide one boolean benchmark target semantically from the system answer.

Question: {case["question"]}
Boolean target name: {check.get("name", "")}
Boolean target description: {check.get("description", "")}
System answer:
{system_answer}

Return valid JSON only as {{"value": true}} or {{"value": false}}.

Rules:
- Judge against the description, not the literal target name.
- Return true only if the answer materially addresses the described point.
- Return false if the answer omits the point or contradicts it.
""".strip()
    response = extractor_llm.invoke([HumanMessage(content=prompt)])
    parsed = _load_json_response(response, {"value": None})
    value = parsed.get("value")
    return value if isinstance(value, bool) else None


def _score_entity_check(check: dict, claims: dict[str, Any]) -> tuple[float, dict[str, Any]]:
    expected = [str(value) for value in check.get("expected", [])]
    min_match = int(check.get("min_match", len(expected)))
    claim_field = str(check.get("claim_field", "entities"))
    actual_values = [str(value) for value in claims.get(claim_field, [])]
    expected_norm = {_normalize_text(value): value for value in expected}
    actual_norm = {_normalize_text(value): value for value in actual_values}
    matched_keys = [key for key in expected_norm if key in actual_norm]
    fraction = min(1.0, len(matched_keys) / max(1, min_match))
    return fraction, {
        "expected": expected,
        "actual": actual_values,
        "matched": [expected_norm[key] for key in matched_keys],
    }


def _score_ordered_check(check: dict, claims: dict[str, Any]) -> tuple[float, dict[str, Any]]:
    expected = [str(value) for value in check.get("expected", [])]
    actual = [str(value) for value in claims.get(check.get("claim_field", "ordered_entities"), [])]
    min_match = int(check.get("min_match", len(expected)))
    if check.get("order_sensitive", False):
        matched = [
            expected_value
            for index, expected_value in enumerate(expected)
            if index < len(actual) and _normalize_text(actual[index]) == _normalize_text(expected_value)
        ]
    else:
        expected_norm = {_normalize_text(value): value for value in expected}
        actual_norm = {_normalize_text(value) for value in actual}
        matched = [value for norm, value in expected_norm.items() if norm in actual_norm]
    fraction = min(1.0, len(matched) / max(1, min_match))
    return fraction, {"expected": expected, "actual": actual, "matched": matched}


def _score_numeric_check(check: dict, claims: dict[str, Any]) -> tuple[float, dict[str, Any]]:
    subject = str(check.get("subject", ""))
    metric = str(check.get("metric", ""))
    expected = float(check.get("expected", 0))
    tolerance = float(check.get("tolerance", 0))
    actual = _find_numeric_claim(claims, subject, metric)
    if actual is None:
        return 0.0, {"subject": subject, "metric": metric, "expected": expected, "actual": None}
    delta = abs(actual - expected)
    return (1.0 if delta <= tolerance else 0.0), {
        "subject": subject,
        "metric": metric,
        "expected": expected,
        "actual": actual,
        "tolerance": tolerance,
        "delta": delta,
    }


def _score_boolean_check(check: dict, claims: dict[str, Any]) -> tuple[float, dict[str, Any]]:
    name = str(check.get("name", ""))
    expected = bool(check.get("expected"))
    actual = _find_boolean_claim(claims, name)
    if actual is None:
        # Claim not found: extractor could not parse a boolean claim — score as failure
        score = 0.0
    else:
        score = 1.0 if actual is expected else 0.0
    return score, {"name": name, "expected": expected, "actual": actual}


def _score_recommendation_count(check: dict, claims: dict[str, Any]) -> tuple[float, dict[str, Any]]:
    min_count = int(check.get("min_count", 1))
    raw = [str(value) for value in claims.get("recommendations", []) if str(value).strip()]
    # Deduplicate by normalized text and filter trivially short strings
    seen: set[str] = set()
    recommendations: list[str] = []
    for r in raw:
        key = _normalize_text(r)
        if key not in seen and len(r) >= 15:
            seen.add(key)
            recommendations.append(r)
    fraction = min(1.0, len(recommendations) / max(1, min_count))
    return fraction, {"min_count": min_count, "actual_count": len(recommendations), "recommendations": recommendations}


def _evaluate_deterministic_check(check: dict, claims: dict[str, Any]) -> dict[str, Any]:
    kind = check.get("kind")
    if kind == "entity_set_match":
        fraction, details = _score_entity_check(check, claims)
    elif kind == "ordered_topk_match":
        fraction, details = _score_ordered_check(check, claims)
    elif kind == "numeric_value_match":
        fraction, details = _score_numeric_check(check, claims)
    elif kind == "boolean_match":
        fraction, details = _score_boolean_check(check, claims)
    elif kind == "recommendation_count":
        fraction, details = _score_recommendation_count(check, claims)
    elif kind == "pipeline_state_match":
        # Reads directly from routing_state dict — zero LLM involvement, 100% deterministic.
        # Requires the caller to pass routing_state; claims dict is ignored for this kind.
        routing_state = claims.get("__routing_state__", {})
        field = check.get("field", "")
        expected = check.get("expected")
        actual = routing_state.get(field)
        if isinstance(expected, list):
            ok = actual in expected
        else:
            ok = actual == expected
        fraction = 1.0 if ok else 0.0
        details = {"field": field, "expected": expected, "actual": actual}
    else:
        fraction, details = 0.0, {"error": f"Unsupported deterministic check kind: {kind}"}
    return {
        "id": check["id"],
        "kind": kind,
        "evaluator": "deterministic",
        "weight": float(check["weight"]),
        "score_fraction": round(fraction, 3),
        "weighted_score": round(float(check["weight"]) * fraction, 3),
        "passed": fraction >= 1.0,
        "details": details,
    }


def _score_reasoning_checks(
    reasoning_llm: Any,
    case: dict,
    system_answer: str,
    checks: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    from langchain_core.messages import HumanMessage

    if not checks:
        return []

    prompt = {
        "question": case["question"],
        "gold_facts": case.get("gold_facts", {}),
        "system_answer": system_answer,
        "checks": [
            {
                "id": check["id"],
                "kind": check["kind"],
                "prompt": check.get("prompt", ""),
                "required_points": check.get("required_points", []),
            }
            for check in checks
        ],
    }
    response = reasoning_llm.invoke(
        [
            HumanMessage(
                content=(
                    "Evaluate the system answer against each reasoning check. "
                    "Return valid JSON only as "
                    '{"checks": [{"id": "...", "score_fraction": 0.0, "reason": "..."}]}. '
                    "Score fractions must be between 0 and 1.\n\n"
                    + json.dumps(prompt, indent=2)
                )
            )
        ]
    )
    payload = _load_json_response(response, {"checks": []})
    by_id = {item.get("id"): item for item in payload.get("checks", []) if item.get("id")}
    results = []
    for check in checks:
        scored = by_id.get(check["id"], {})
        try:
            fraction = float(scored.get("score_fraction", 0.0))
        except (TypeError, ValueError):
            fraction = 0.0
        fraction = max(0.0, min(1.0, fraction))
        results.append(
            {
                "id": check["id"],
                "kind": check["kind"],
                "evaluator": "reasoning_llm",
                "weight": float(check["weight"]),
                "score_fraction": round(fraction, 3),
                "weighted_score": round(float(check["weight"]) * fraction, 3),
                "passed": fraction >= 0.5,
                "reason": scored.get("reason", ""),
            }
        )
    return results


def _compat_score(raw_score: float) -> int:
    return max(1, min(5, int(round(raw_score))))


def _pipeline_result_to_trace(pipeline_result: Any) -> tuple[str, set[str], list[dict[str, Any]], list[dict[str, Any]]]:
    if isinstance(pipeline_result, str):
        return pipeline_result, set(), [], []
    system_answer = pipeline_result.get("answer", "")
    tool_calls = pipeline_result.get("tool_calls", [])
    trace_errors = pipeline_result.get("errors", [])
    traced_tools = {
        call.get("tool", "")
        for call in tool_calls
        if call.get("tool") and not call.get("blocked")
    }
    return system_answer, traced_tools, tool_calls, trace_errors


def run_tier3_legacy(cases: list[dict], judge_model: str, prompt_template: str, verbose: bool = False) -> dict:
    from main import run_pipeline

    llm_judge = _make_llm(judge_model)
    results = []
    scores = []
    tool_correct = 0
    total_rubric_hits = 0
    total_rubric_items = 0
    skipped_disabled = 0

    for case in cases:
        if "DISABLED" in case:
            logger.info("Skipping disabled case %s", case["id"])
            skipped_disabled += 1
            continue
        cid = case["id"]
        question = case["question"]
        expected_tools = set(case.get("expected_tools", []))

        try:
            pipeline_result = run_pipeline(question, return_trace=True)
            system_answer, traced_tools, tool_calls, trace_errors = _pipeline_result_to_trace(pipeline_result)

            tool_match = expected_tools.issubset(traced_tools)
            if tool_match:
                tool_correct += 1

            judgment = _judge_answer_legacy(llm_judge, prompt_template, case, system_answer)
            score = int(judgment.get("score", 1))
            scores.append(score)

            rubric_hits = judgment.get("rubric_hits", [])
            rubric_misses = judgment.get("rubric_misses", [])
            total_rubric_hits += len(rubric_hits)
            total_rubric_items += len(case.get("rubric", []))

            result = {
                "id": cid,
                "category": case.get("category", "unknown"),
                "question": question,
                "system_answer": system_answer,
                "judge_score": score,
                "judge_reason": judgment.get("reason", ""),
                "rubric_hits": rubric_hits,
                "rubric_misses": rubric_misses,
                "expected_tools": sorted(expected_tools),
                "traced_tools": sorted(traced_tools),
                "tool_calls": tool_calls,
                "trace_errors": trace_errors,
                "tool_match": tool_match,
            }
            if verbose or score <= 2:
                logger.info("[T3 legacy] %s: score=%s/5 | tool_match=%s | %s", cid, score, tool_match, judgment.get("reason", "")[:80])

        except Exception as e:
            result = {"id": cid, "question": question, "judge_score": 1, "error": str(e)}
            scores.append(1)
            logger.error("[T3 legacy] %s: EXCEPTION — %s", cid, e)

        results.append(result)

    n = len(results)
    mean_score = sum(scores) / n if n > 0 else 0.0
    rubric_hit_rate = total_rubric_hits / total_rubric_items if total_rubric_items > 0 else 0.0
    tool_rate = tool_correct / n if n > 0 else 0.0

    print(
        f"\nTier 3 — End-to-End: "
        f"mean_score={mean_score:.2f}/5 | "
        f"rubric_hit_rate={rubric_hit_rate:.0%} | "
        f"tool_selection={tool_rate:.0%}"
    )
    return {
        "mean_judge_score": round(mean_score, 3),
        "rubric_hit_rate": round(rubric_hit_rate, 3),
        "tool_selection_rate": round(tool_rate, 3),
        "score_distribution": {str(i): scores.count(i) for i in range(1, 6)},
        "total": n,
        "skipped_disabled": skipped_disabled,
        "cases": results,
        "tier3_format": "legacy",
    }


def run_tier3_grounded(
    cases: list[dict],
    claim_extractor_model: str,
    reasoning_judge_model: str,
    verbose: bool = False,
) -> dict:
    from main import run_pipeline

    extractor_llm = _make_llm(claim_extractor_model)
    reasoning_llm = _make_llm(reasoning_judge_model)

    results = []
    scores = []
    fact_scores = []
    reasoning_scores = []
    tool_correct = 0
    total_check_fraction = 0.0
    total_checks = 0
    skipped_disabled = 0

    for case in cases:
        if "DISABLED" in case:
            logger.info("Skipping disabled case %s", case["id"])
            skipped_disabled += 1
            continue

        cid = case["id"]
        question = case["question"]
        expected_tools = set(case.get("expected_tools", []))
        deterministic_checks = [check for check in case.get("checks", []) if check.get("evaluator") == "deterministic"]
        reasoning_checks = [check for check in case.get("checks", []) if check.get("evaluator") == "reasoning_llm"]

        try:
            pipeline_result = run_pipeline(question, return_trace=True)
            system_answer, traced_tools, tool_calls, trace_errors = _pipeline_result_to_trace(pipeline_result)
            tool_match = expected_tools.issubset(traced_tools)
            if tool_match:
                tool_correct += 1

            extracted_claims = _extract_case_claims(extractor_llm, case, system_answer)
            deterministic_results = [
                _evaluate_deterministic_check(check, extracted_claims) for check in deterministic_checks
            ]
            reasoning_results = _score_reasoning_checks(
                reasoning_llm,
                case,
                system_answer,
                reasoning_checks,
            )

            all_check_results = deterministic_results + reasoning_results
            fact_score = round(sum(result["weighted_score"] for result in deterministic_results), 3)
            reasoning_score = round(sum(result["weighted_score"] for result in reasoning_results), 3)
            judge_score_raw = round(fact_score + reasoning_score, 3)
            judge_score = _compat_score(judge_score_raw)

            total_check_fraction += sum(result["score_fraction"] for result in all_check_results)
            total_checks += len(all_check_results)
            scores.append(judge_score)
            fact_scores.append(fact_score)
            reasoning_scores.append(reasoning_score)

            fact_checks_passed = [result for result in deterministic_results if result["passed"]]
            fact_checks_failed = [result for result in deterministic_results if not result["passed"]]
            reasoning_checks_result = reasoning_results

            failed_check_ids = [result["id"] for result in all_check_results if result["score_fraction"] < 1.0]
            judge_reason = (
                "All grounded checks passed."
                if not failed_check_ids
                else f"Incomplete grounded checks: {', '.join(failed_check_ids)}"
            )

            result = {
                "id": cid,
                "category": case.get("category", "unknown"),
                "question": question,
                "system_answer": system_answer,
                "judge_score": judge_score,
                "judge_score_raw": judge_score_raw,
                "fact_score": fact_score,
                "reasoning_score": reasoning_score,
                "judge_reason": judge_reason,
                "fact_checks_passed": fact_checks_passed,
                "fact_checks_failed": fact_checks_failed,
                "reasoning_checks": reasoning_checks_result,
                "extracted_claims": extracted_claims,
                "expected_tools": sorted(expected_tools),
                "traced_tools": sorted(traced_tools),
                "tool_calls": tool_calls,
                "trace_errors": trace_errors,
                "tool_match": tool_match,
            }
            if verbose or judge_score <= 2:
                logger.info(
                    "[T3 grounded] %s: score=%.2f/5 -> %s/5 | fact=%.2f reason=%.2f | %s",
                    cid,
                    judge_score_raw,
                    judge_score,
                    fact_score,
                    reasoning_score,
                    judge_reason,
                )

        except Exception as e:
            result = {"id": cid, "question": question, "judge_score": 1, "error": str(e)}
            scores.append(1)
            fact_scores.append(0.0)
            reasoning_scores.append(0.0)
            logger.error("[T3 grounded] %s: EXCEPTION — %s", cid, e)

        results.append(result)

    n = len(results)
    mean_score = sum(scores) / n if n > 0 else 0.0
    mean_fact_score = sum(fact_scores) / n if n > 0 else 0.0
    mean_reasoning_score = sum(reasoning_scores) / n if n > 0 else 0.0
    rubric_hit_rate = total_check_fraction / total_checks if total_checks > 0 else 0.0
    tool_rate = tool_correct / n if n > 0 else 0.0

    print(
        f"\nTier 3 — End-to-End: "
        f"mean_score={mean_score:.2f}/5 | "
        f"rubric_hit_rate={rubric_hit_rate:.0%} | "
        f"tool_selection={tool_rate:.0%}"
    )
    return {
        "mean_judge_score": round(mean_score, 3),
        "mean_fact_score": round(mean_fact_score, 3),
        "mean_reasoning_score": round(mean_reasoning_score, 3),
        "rubric_hit_rate": round(rubric_hit_rate, 3),
        "tool_selection_rate": round(tool_rate, 3),
        "score_distribution": {str(i): scores.count(i) for i in range(1, 6)},
        "total": n,
        "skipped_disabled": skipped_disabled,
        "cases": results,
        "tier3_format": "grounded",
    }


def run_tier3(
    cases: list[dict],
    judge_model: str | None = None,
    prompt_template: str = "",
    *,
    tier3_format: str = "grounded",
    claim_extractor_model: str | None = None,
    reasoning_judge_model: str | None = None,
    verbose: bool = False,
) -> dict:
    """Run Tier 3 in either legacy or grounded mode."""
    if tier3_format == "legacy":
        return run_tier3_legacy(
            cases,
            judge_model=judge_model or _DEFAULT_LLM_MODEL,
            prompt_template=prompt_template,
            verbose=verbose,
        )
    return run_tier3_grounded(
        cases,
        claim_extractor_model=claim_extractor_model or judge_model or _DEFAULT_LLM_MODEL,
        reasoning_judge_model=reasoning_judge_model or judge_model or _DEFAULT_LLM_MODEL,
        verbose=verbose,
    )


# ── Incident Mode Benchmark ───────────────────────────────────────────────────

_ABLATION_PRESETS: dict[str, dict] = {
    "full":        {},
    "no-tavily":   {"use_tavily": False},
    "no-routing":  {"use_routing": False},
    "no-plan":     {"use_planning": False},
    "react":       None,   # special: use run_pipeline() instead of orchestrator
    "llm-only":    {"use_kg": False, "use_tavily": False},   # DB queries still run; no KG
    # Reproducible benchmark preset: disables Tavily (non-deterministic external API)
    # Use this for paper-reportable runs to reduce cross-run variance.
    "benchmark":   {"use_tavily": False},
    # ── Graph-claim isolation baselines (paper main table) ──────────────────
    # concept-only: keep concept→table mapping (MAPS_TO) but disable causal
    # graph traversal (INCREASES/REDUCES/INDICATES). Tests whether the causal
    # graph adds anything on top of curated concept-table mappings.
    "concept-only": {"use_kg": False},
    # flat-schema: same agent, same tools, but inject the FULL DDCG schema
    # (all 36 tables, ~149 columns) into the SQL prompt instead of the
    # graph-retrieved subset. Tests whether the graph-mediated schema retrieval
    # adds anything over a flat schema dump on a small enough catalog.
    "flat-schema": {"_force_full_schema": True, "use_kg": False},
    # flat-ekg: inject the full curated EKG (concepts + edges) as static text
    # into the planner+executor system prompts instead of doing graph traversal
    # via tool calls. KG tool is disabled (use_kg=False); the agent sees the
    # entire causal graph up-front. Tests whether graph traversal contributes
    # anything beyond what a frontier LLM can extract from a flat-text dump
    # of the same EKG.
    "flat-ekg":    {"_flat_ekg_prompt": True, "use_kg": False},
    # lexical-link: same agent, same tools, but retrieve schema via lexical
    # (BM25-style) keyword scoring of table/column metadata against the user
    # query — no EKG concepts, no MAPS_TO. Tests whether the contribution is
    # just better schema search rather than the curated graph structure.
    "lexical-link": {"_lexical_schema_link": True, "use_kg": False},
    # ── ReAct family baselines ───────────────────────────────────────────────
    # Text-RAG hybrid baseline (conventional RAG + SQL tool):
    #   - ReAct single-agent (no 4-stage pipeline) — shares the react fallback
    #   - Tool set: [query_database, query_documents] — no KG, no MAPS_TO, no DDCG
    #   - query_documents retrieves top-5 chunks from TDIS corpus via
    #     sentence-transformer embeddings (see src/agent/text_rag.py)
    # Measures whether structured knowledge (EKG + DDCG) beats a commodity
    # chunk retriever pulling from the same source documents.
    "text-rag":    {"_pipeline_mode": "text-rag", "_disable_concept_routing": True,
                    "_use_react_fallback": True},
    # LightRAG hybrid baseline (auto-extracted KG + SQL tool):
    #   - ReAct single-agent (no 4-stage pipeline) — shares the react fallback
    #   - Tool set: [query_database, query_lightrag] — no curated EKG, no MAPS_TO,
    #     no DDCG concept-aware retrieval
    #   - query_lightrag retrieves entities/relations/text from a LightRAG
    #     index built over the same TDIS corpus the EKG was extracted from
    #     (see scripts/baselines/index_lightrag.py)
    # Sits between text-rag (flat chunks) and full (curated EKG + DDCG) on
    # the ladder of structure: tests whether automatic graph extraction
    # substitutes for manual curation. Appendix-only in the paper main table.
    "lightrag":    {"_pipeline_mode": "lightrag", "_disable_concept_routing": True,
                    "_use_react_fallback": True},
}


def _run_single_case(
    case: dict,
    ablation_flags: dict | None,
    extractor_llm: Any,
    reasoning_llm: Any,
    use_react_fallback: bool,
    verbose: bool,
) -> dict:
    """Execute a single benchmark case and return its result dict.

    This is extracted from the ``run_incident_benchmark`` loop so it can be
    called both serially and via ``concurrent.futures.ThreadPoolExecutor``.

    Returns a dict with the same structure that ``run_incident_benchmark``
    previously appended to ``case_results``, plus a sentinel key
    ``"__skipped__": True`` when the case is disabled.
    """
    if "DISABLED" in case:
        return {"__skipped__": True, "id": case["id"]}

    cid = case["id"]
    question = case["question"]

    # Per-case ablation overrides (e.g., Tier K forces use_tavily=False)
    case_ablation = {**(ablation_flags or {}), **(case.get("ablation_config") or {})}

    deterministic_checks = [c for c in case.get("checks", []) if c.get("evaluator") == "deterministic"]
    reasoning_checks = [c for c in case.get("checks", []) if c.get("evaluator") == "reasoning_llm"]

    # Reset the SQL-agent retry accumulator at case start. Each query_database
    # call inside the orchestrator's execute node adds its retry_count; we
    # drain the total at the end of this case to record real sql_retries.
    try:
        from src.agent.text_to_sql_agent import reset_retry_accumulator
        reset_retry_accumulator()
    except Exception:
        pass

    try:
        if use_react_fallback:
            from main import run_pipeline
            import concurrent.futures as _cf
            import threading as _threading
            _result_holder: list = []
            _exc_holder: list = []
            def _react_call():
                try:
                    _result_holder.append(run_pipeline(question, return_trace=True))
                except Exception as _e:
                    _exc_holder.append(_e)
            _t = _threading.Thread(target=_react_call, daemon=True)
            _t.start()
            react_timeout = 360 if os.environ.get("PIPELINE_MODE") in ("text-rag", "lightrag") else 240
            _t.join(timeout=react_timeout)
            if _t.is_alive():
                raise TimeoutError(f"react baseline timed out after {react_timeout}s")
            if _exc_holder:
                raise _exc_holder[0]
            pipeline_result = _result_holder[0]
            system_answer, traced_tools, tool_calls, trace_errors = _pipeline_result_to_trace(pipeline_result)
            # Baselines (react/text-rag/lightrag/CHESS) have no explicit routing
            # node; without populating routing_state, every Tier R pipeline_state_match
            # check returns 0 by structural disadvantage. Derive routing_state
            # post-hoc from the produced answer so Tier R measures operational-frame
            # alignment of the baseline's answer (not direct routing accuracy).
            routing_state = _derive_baseline_routing_state(
                extractor_llm, case, system_answer
            )
        else:
            from src.agent.orchestrator import run as run_orchestrator
            routing_state = run_orchestrator(
                question,
                ablation=case_ablation or None,
                return_state=True,
            )
            system_answer = routing_state.get("final_answer", "")
            tool_calls = []
            trace_errors = []

        # Inject routing_state into claims so pipeline_state_match can read it
        extracted_claims = _extract_case_claims(extractor_llm, case, system_answer)
        extracted_claims["__routing_state__"] = routing_state
        for check in deterministic_checks:
            if check.get("kind") != "boolean_match":
                continue
            name = str(check.get("name", ""))
            if not name or _find_boolean_claim(extracted_claims, name) is not None:
                continue
            inferred = _semantic_boolean_fallback(extractor_llm, case, check, system_answer)
            if inferred is None:
                continue
            extracted_claims.setdefault("boolean_claims", []).append(
                {"name": name, "value": inferred, "source": "semantic_fallback"}
            )

        deterministic_results = [
            _evaluate_deterministic_check(check, extracted_claims)
            for check in deterministic_checks
        ]
        reasoning_results = _score_reasoning_checks(
            reasoning_llm, case, system_answer, reasoning_checks
        )

        all_check_results = deterministic_results + reasoning_results
        fact_score = round(sum(r["weighted_score"] for r in deterministic_results), 3)
        reasoning_score = round(sum(r["weighted_score"] for r in reasoning_results), 3)
        judge_score_raw = round(fact_score + reasoning_score, 3)
        judge_score = _compat_score(judge_score_raw)

        failed_check_ids = [r["id"] for r in all_check_results if r["score_fraction"] < 1.0]
        judge_reason = (
            "All grounded checks passed."
            if not failed_check_ids
            else f"Incomplete: {', '.join(failed_check_ids)}"
        )

        # Drain the SQL-agent retry accumulator (real retry count summed across
        # all query_database calls during this case). Returns 0 for baselines
        # that don't use the SQL sub-pipeline.
        try:
            from src.agent.text_to_sql_agent import consume_retry_total
            sql_retries = consume_retry_total()
        except Exception:
            sql_retries = 0

        result = {
            "id": cid,
            "tier": case.get("tier", ""),
            "category": case.get("category", ""),
            "question": question,
            "ablation_config": case_ablation,
            "system_answer": system_answer,
            "routing_state": {k: v for k, v in routing_state.items() if k != "final_answer"},
            "judge_score": judge_score,
            "judge_score_raw": judge_score_raw,
            "fact_score": fact_score,
            "reasoning_score": reasoning_score,
            "judge_reason": judge_reason,
            "fact_checks_passed": [r for r in deterministic_results if r["passed"]],
            "fact_checks_failed": [r for r in deterministic_results if not r["passed"]],
            "reasoning_checks": reasoning_results,
            "extracted_claims": {k: v for k, v in extracted_claims.items() if k != "__routing_state__"},
            "tool_calls": tool_calls,
            "trace_errors": trace_errors,
            "sql_retries": sql_retries,
            # Routing accuracy fields — consumed by run_incident_benchmark aggregator
            "__deterministic_results__": deterministic_results,
        }

        if verbose or judge_score <= 2:
            logger.info(
                "[incident] %s: score=%.2f/5 -> %s/5 | fact=%.2f reason=%.2f | %s",
                cid, judge_score_raw, judge_score, fact_score, reasoning_score, judge_reason,
            )

    except Exception as e:
        result = {
            "id": cid,
            "question": question,
            "judge_score": 1,
            "error": str(e),
            "__deterministic_results__": [],
        }
        logger.error("[incident] %s: EXCEPTION — %s", cid, e)

    return result


def run_incident_benchmark(
    cases: list[dict],
    ablation_name: str = "full",
    claim_extractor_model: str | None = None,
    reasoning_judge_model: str | None = None,
    verbose: bool = False,
    parallel: int = 1,
) -> dict:
    """Run the incident mode benchmark against the orchestrator pipeline.

    Each case may specify ``ablation_config`` to override per-case ablation
    settings (e.g., Tier K cases always run with ``use_tavily=False``).

    The ``pipeline_state_match`` check kind reads directly from the
    orchestrator's returned PipelineState — no LLM extraction needed.

    Args:
        cases: List of case dicts loaded from the benchmark JSON.
        ablation_name: Which ablation preset to apply (default ``"full"``).
        claim_extractor_model: OpenRouter model for claim extraction.
        reasoning_judge_model: OpenRouter model for reasoning judgement.
        verbose: Log per-case results even when score > 2.
        parallel: Number of cases to run concurrently via
            ``ThreadPoolExecutor``.  Default ``1`` (serial).  ``ChatOpenAI``
            makes stateless HTTP calls and is safe for concurrent use.
    """
    raw_flags = _ABLATION_PRESETS.get(ablation_name, {}) or {}
    # Extract side-channel meta flags (not passed to build_orchestrator).
    if raw_flags.pop("_disable_concept_routing", False):
        os.environ["DISABLE_CONCEPT_ROUTING"] = "1"
        print("[ABLATION] Context graph DISABLED — KG causal retrieval off, "
              "MAPS_TO routing off, DDCG catalog omitted from prompt. "
              "SQL agent must use DuckDB introspection + hardcoded hints.")
    else:
        # Ensure clean state if a prior run set this
        os.environ.pop("DISABLE_CONCEPT_ROUTING", None)
    # flat-schema: bypass concept matching, dump full DDCG into the SQL prompt
    if raw_flags.pop("_force_full_schema", False):
        os.environ["FORCE_FULL_SCHEMA"] = "1"
        print("[ABLATION] flat-schema — concept matching skipped; "
              "full DDCG (all 36 tables) injected into the SQL prompt.")
    else:
        os.environ.pop("FORCE_FULL_SCHEMA", None)
    # flat-ekg: inject the entire curated EKG as static text into the
    # planner+executor system prompts (no graph traversal tool calls).
    if raw_flags.pop("_flat_ekg_prompt", False):
        os.environ["FLAT_EKG_PROMPT"] = "1"
        print("[ABLATION] flat-ekg — full curated EKG injected into planner+"
              "executor system prompts; query_knowledge_graph tool disabled.")
    else:
        os.environ.pop("FLAT_EKG_PROMPT", None)
    # lexical-link: bypass concept matching, retrieve via BM25-style keyword
    # scoring of table/column metadata against the user query
    if raw_flags.pop("_lexical_schema_link", False):
        os.environ["LEXICAL_SCHEMA_LINK"] = "1"
        print("[ABLATION] lexical-link — concept matching skipped; "
              "schema retrieved by lexical keyword scoring of table/column metadata.")
    else:
        os.environ.pop("LEXICAL_SCHEMA_LINK", None)
    pipeline_mode = raw_flags.pop("_pipeline_mode", None)
    if pipeline_mode:
        os.environ["PIPELINE_MODE"] = pipeline_mode
        print(f"[ABLATION] PIPELINE_MODE={pipeline_mode} — swapping agent tools / prompt.")
        # Invalidate any cached agent so the next run_pipeline() rebuilds it.
        try:
            import main as _main_mod
            _main_mod._reset_agent()
        except Exception:
            pass
    else:
        os.environ.pop("PIPELINE_MODE", None)
    force_react = raw_flags.pop("_use_react_fallback", False)
    ablation_flags = raw_flags
    use_react_fallback = ablation_name == "react" or force_react

    # LLM objects are created once and shared across threads.  ChatOpenAI is
    # stateless between invocations, so sharing is safe.
    extractor_llm = _make_llm(claim_extractor_model or _DEFAULT_LLM_MODEL)
    reasoning_llm = _make_llm(reasoning_judge_model or _DEFAULT_LLM_MODEL)

    # Preserve original case order by indexing futures/results by case id.
    ordered_ids: list[str] = [case["id"] for case in cases]
    result_by_id: dict[str, dict] = {}
    skipped = 0

    if parallel > 1:
        with concurrent.futures.ThreadPoolExecutor(max_workers=parallel) as executor:
            future_to_id = {
                executor.submit(
                    _run_single_case,
                    case,
                    ablation_flags,
                    extractor_llm,
                    reasoning_llm,
                    use_react_fallback,
                    verbose,
                ): case["id"]
                for case in cases
            }
            done_count = 0
            for future in concurrent.futures.as_completed(future_to_id):
                cid = future_to_id[future]
                done_count += 1
                try:
                    result_by_id[cid] = future.result()
                    r = result_by_id[cid]
                    _js = r.get('judge_score') or 0
                    _fs = r.get('fact_score') or 0
                    score_str = f"judge={_js:.2f} fact={_fs:.2f}" if not r.get("__skipped__") else "skipped"
                    print(f"  [{done_count}/{len(cases)}] {cid}: {score_str}", flush=True)
                except Exception as e:
                    # Catch unexpected executor-level failures
                    logger.error("[incident] %s: EXECUTOR EXCEPTION — %s", cid, e)
                    result_by_id[cid] = {
                        "id": cid,
                        "judge_score": 1,
                        "error": str(e),
                        "__deterministic_results__": [],
                    }
                    print(f"  [{done_count}/{len(cases)}] {cid}: ERROR — {e}", flush=True)
    else:
        for i, case in enumerate(cases, 1):
            result = _run_single_case(
                case, ablation_flags, extractor_llm, reasoning_llm, use_react_fallback, verbose
            )
            result_by_id[case["id"]] = result
            if not result.get("__skipped__"):
                _js = result.get('judge_score') or 0
                _fs = result.get('fact_score') or 0
                score_str = f"judge={_js:.2f} fact={_fs:.2f}"
            else:
                score_str = "skipped"
            print(f"  [{i}/{len(cases)}] {case['id']}: {score_str}", flush=True)

    # Reconstruct results in original case order and aggregate metrics.
    results: list[dict] = []
    scores: list[float] = []
    fact_scores: list[float] = []
    reasoning_scores_list: list[float] = []
    routing_correct = 0
    total_routing_checks = 0
    sql_retry_counts: list[int] = []

    for cid in ordered_ids:
        result = result_by_id.get(cid, {"id": cid, "judge_score": 1, "error": "missing result", "__deterministic_results__": []})
        if result.get("__skipped__"):
            skipped += 1
            continue

        det_results = result.pop("__deterministic_results__", [])

        judge_score = result.get("judge_score", 1)
        fact_score = result.get("fact_score", 0.0)
        reasoning_score = result.get("reasoning_score", 0.0)

        scores.append(judge_score)
        fact_scores.append(fact_score)
        reasoning_scores_list.append(reasoning_score)
        # Real per-case retry total (drained from text_to_sql_agent's thread-local
        # accumulator in _run_single_case). Defaults to 0 for baselines that do
        # not invoke the SQL sub-pipeline.
        sql_retry_counts.append(int(result.get("sql_retries", 0)))

        for r in det_results:
            if r.get("kind") == "pipeline_state_match":
                total_routing_checks += 1
                if r.get("passed"):
                    routing_correct += 1

        results.append(result)

    n = len(results)
    mean_score = sum(scores) / n if n > 0 else 0.0
    mean_fact = sum(fact_scores) / n if n > 0 else 0.0
    mean_reasoning = sum(reasoning_scores_list) / n if n > 0 else 0.0
    routing_accuracy = routing_correct / total_routing_checks if total_routing_checks > 0 else None
    mean_sql_retries = sum(sql_retry_counts) / len(sql_retry_counts) if sql_retry_counts else 0.0

    print(
        f"\nIncident Benchmark ({ablation_name}): "
        f"mean={mean_score:.2f}/5 | fact={mean_fact:.2f} | reasoning={mean_reasoning:.2f} | "
        f"routing_accuracy={routing_accuracy:.0%} | sql_retries={mean_sql_retries:.1f}"
        if routing_accuracy is not None
        else f"\nIncident Benchmark ({ablation_name}): "
        f"mean={mean_score:.2f}/5 | fact={mean_fact:.2f} | reasoning={mean_reasoning:.2f}"
    )
    return {
        "ablation": ablation_name,
        "mean_judge_score": round(mean_score, 3),
        "mean_fact_score": round(mean_fact, 3),
        "mean_reasoning_score": round(mean_reasoning, 3),
        "routing_accuracy": round(routing_accuracy, 3) if routing_accuracy is not None else None,
        "mean_sql_retries": round(mean_sql_retries, 2),
        "score_distribution": {str(i): scores.count(i) for i in range(1, 6)},
        "total": n,
        "skipped": skipped,
        "extractor_model": claim_extractor_model or _DEFAULT_LLM_MODEL,
        "reasoning_judge_model": reasoning_judge_model or _DEFAULT_LLM_MODEL,
        "scorer_commit_sha": _get_scorer_commit_sha(),
        "cases": results,
    }


# ── Ablation Comparison ────────────────────────────────────────────────────────

_ABLATION_HYPOTHESES = [
    ("full", "no-routing",  "full >= no-routing on pipeline_state_match checks"),
    ("full", "no-plan",     "full >= no-plan on Tier M multi-table numeric checks"),
    ("full", "no-tavily",   "full >= no-tavily on Tier M reasoning checks"),
    ("full", "text-rag",    "full >= text-rag overall (structured CG vs chunk RAG)"),
    ("full", "react",       "full >= react overall"),
]


def compare_ablations(result_files: list[str]) -> dict:
    """Compare multiple ablation result JSONs and flag hypothesis violations.

    Args:
        result_files: Paths to incident benchmark result JSONs, one per ablation config.

    Returns:
        Dict with pairwise score diffs and hypothesis_violation flags.
    """
    loaded: dict[str, dict] = {}
    for path in result_files:
        with open(path) as f:
            data = json.load(f)
        name = data.get("ablation", Path(path).stem)
        loaded[name] = data

    comparisons = []
    for a, b, hypothesis in _ABLATION_HYPOTHESES:
        if a not in loaded or b not in loaded:
            continue
        score_a = loaded[a]["mean_judge_score"]
        score_b = loaded[b]["mean_judge_score"]
        diff = round(score_a - score_b, 3)
        violated = diff < 0   # a < b means the hypothesis a >= b is violated
        comparisons.append({
            "hypothesis": hypothesis,
            "config_a": a,
            "config_b": b,
            "score_a": score_a,
            "score_b": score_b,
            "diff_a_minus_b": diff,
            "hypothesis_violation": violated,
        })
        status = "VIOLATED" if violated else "confirmed"
        print(f"  [{status}] {a}={score_a:.3f} vs {b}={score_b:.3f} (diff={diff:+.3f}) — {hypothesis}")

    all_scores = {name: data["mean_judge_score"] for name, data in loaded.items()}
    return {"scores": all_scores, "comparisons": comparisons}


# ── Overall score ─────────────────────────────────────────────────────────────

def compute_overall(t1: dict | None, t2: dict | None, t3: dict | None) -> float:
    """Weighted overall score [0, 1] across available tiers."""
    components = []
    if t1:
        components.append(t1["pass_rate"])
    if t2:
        # Weight exec and result check equally
        components.append((t2["sql_exec_rate"] + t2["result_check_rate"]) / 2)
    if t3:
        components.append(t3["mean_judge_score"] / 5.0)
    return round(sum(components) / len(components), 3) if components else 0.0


# ── Main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(description="Run DisasterContextGraph benchmark")
    parser.add_argument("--tier", choices=["1", "2", "3", "all"], default="all",
                        help="Which tier(s) to run (default: all)")
    parser.add_argument(
        "--tier3-format",
        choices=["grounded", "legacy"],
        default="grounded",
        help="Which Tier 3 benchmark format to use (default: grounded)",
    )
    parser.add_argument(
        "--mode",
        choices=["react", "incident"],
        default="incident",
        help="incident: 4-node incident pipeline benchmark (default); react: legacy ReAct Tier 1/2/3",
    )
    parser.add_argument(
        "--ablation",
        choices=list(_ABLATION_PRESETS.keys()),
        default="full",
        help="Ablation configuration for incident mode (default: full)",
    )
    parser.add_argument(
        "--compare",
        nargs="+",
        metavar="FILE",
        default=None,
        help="Compare multiple ablation result JSONs and report hypothesis violations",
    )
    parser.add_argument("--output", type=str, default=None,
                        help="Output JSON path (default: auto-generated from mode/ablation/timestamp)")
    parser.add_argument("--cases", nargs="+", metavar="CASE_ID", default=None,
                        help="Run only specific case IDs (incident mode only, e.g. incident_k02 incident_m06)")
    parser.add_argument("--verbose", "-v", action="store_true",
                        help="Print per-case results")
    parser.add_argument(
        "--benchmark",
        type=str,
        default=None,
        metavar="PATH",
        help="Path to benchmark JSON (default: configs/benchmark/test.json). Incident mode only.",
    )
    parser.add_argument(
        "--parallel",
        type=int,
        default=1,
        metavar="N",
        help="Run N cases in parallel threads (default: 1, serial). Incident mode only.",
    )
    parser.add_argument(
        "--extractor-model",
        type=str,
        default=None,
        metavar="MODEL",
        help="Override claim extractor model (e.g. minimax/minimax-m2.7). Incident mode only.",
    )
    parser.add_argument(
        "--pipeline-model",
        type=str,
        default=None,
        metavar="MODEL",
        help="Override pipeline LLM model (default: LLM_MODEL env var). "
             "Set before any src.* imports so the config singleton picks it up. "
             "Example: qwen/qwen3-vl-30b-a3b-instruct",
    )
    args = parser.parse_args()

    # Override pipeline model BEFORE any src.* imports — cfg singleton reads LLM_MODEL at import time.
    if args.pipeline_model:
        os.environ["LLM_MODEL"] = args.pipeline_model
        print(f"[INFO] Pipeline model overridden to: {args.pipeline_model}")

    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    _pipeline_model_tag = ""
    if args.pipeline_model:
        # e.g. qwen/qwen3-vl-30b-a3b-instruct → qwen3-vl-30b
        _slug = args.pipeline_model.split("/")[-1]
        _pipeline_model_tag = "_" + _slug[:20].rstrip("-")
    _effective_model = args.pipeline_model or os.getenv("LLM_MODEL", _DEFAULT_LLM_MODEL)
    _model_subdir = _effective_model.rsplit("/", 1)[-1] if _effective_model else "_unclassified"
    if args.output:
        output_path = Path(args.output)
    elif args.mode == "incident":
        output_path = CONFIG_DIR / "benchmark" / "results" / _model_subdir / f"incident_{args.ablation}{_pipeline_model_tag}_{timestamp}.json"
    else:
        output_path = CONFIG_DIR / "benchmark" / "results" / _model_subdir / f"react_{timestamp}.json"

    # ── Compare mode ──────────────────────────────────────────────────────────
    if args.compare:
        print("\n" + "=" * 60)
        print("ABLATION COMPARISON")
        print("=" * 60)
        comparison = compare_ablations(args.compare)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w") as f:
            json.dump(comparison, f, indent=2)
        print(f"\nComparison saved to: {output_path}")
        return

    # ── Incident mode ─────────────────────────────────────────────────────────
    if args.mode == "incident":
        print("\n" + "=" * 60)
        print(f"INCIDENT BENCHMARK — ablation: {args.ablation}")
        print("=" * 60)
        incident_path = Path(args.benchmark) if args.benchmark else INCIDENT_BENCHMARK_PATH
        if not incident_path.exists():
            print(f"ERROR: {incident_path} not found. Run: python scripts/build_incident_benchmark.py")
            return
        with open(incident_path) as f:
            incident_data = json.load(f)
        cases = incident_data["cases"]
        if args.cases:
            cases = [c for c in cases if c["id"] in args.cases]
            if not cases:
                print(f"ERROR: No cases matched {args.cases}")
                return
            print(f"Running {len(cases)} selected case(s): {[c['id'] for c in cases]}")
        result = run_incident_benchmark(
            cases,
            ablation_name=args.ablation,
            claim_extractor_model=args.extractor_model or incident_data.get("claim_extractor_model", _DEFAULT_LLM_MODEL),
            # --extractor-model overrides reasoning judge too — both must use the same cross-model
            # to avoid self-evaluation bias (pipeline model must not judge its own reasoning).
            reasoning_judge_model=args.extractor_model or incident_data.get("reasoning_judge_model", _DEFAULT_LLM_MODEL),
            verbose=args.verbose,
            parallel=args.parallel,
        )
        result["pipeline_model"] = os.getenv("LLM_MODEL", _DEFAULT_LLM_MODEL)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w") as f:
            json.dump(result, f, indent=2)
        print(f"\nResults saved to: {output_path}")
        return

    run_tiers = {"1", "2", "3"} if args.tier == "all" else {args.tier}

    t1_result = t2_result = t3_result = None

    # ── Tier 1 ────────────────────────────────────────────────────────────────
    if "1" in run_tiers:
        print("\n" + "=" * 60)
        print("TIER 1: Concept Matching")
        print("=" * 60)
        with open(TIER1_PATH) as f:
            t1_data = json.load(f)
        t1_result = run_tier1(t1_data["cases"], verbose=args.verbose)

    # ── Tier 2 ────────────────────────────────────────────────────────────────
    if "2" in run_tiers:
        print("\n" + "=" * 60)
        print("TIER 2: SQL Agent Accuracy")
        print("=" * 60)
        with open(TIER2_PATH) as f:
            t2_data = json.load(f)
        t2_result = run_tier2(t2_data["cases"], verbose=args.verbose)

    # ── Tier 3 ────────────────────────────────────────────────────────────────
    if "3" in run_tiers:
        print("\n" + "=" * 60)
        print("TIER 3: End-to-End Pipeline Quality")
        print("=" * 60)
        tier3_path = TIER3_GROUNDED_PATH if args.tier3_format == "grounded" else TIER3_LEGACY_PATH
        with open(tier3_path) as f:
            t3_data = json.load(f)
        t3_result = run_tier3(
            t3_data["cases"],
            judge_model=t3_data.get("judge_model", _DEFAULT_LLM_MODEL),
            prompt_template=t3_data.get("judge_prompt_template", ""),
            tier3_format=args.tier3_format,
            claim_extractor_model=t3_data.get("claim_extractor_model"),
            reasoning_judge_model=t3_data.get("reasoning_judge_model"),
            verbose=args.verbose,
        )

    # ── Summary ───────────────────────────────────────────────────────────────
    overall = compute_overall(t1_result, t2_result, t3_result)
    print(f"\n{'=' * 60}")
    print(f"OVERALL SCORE: {overall:.3f}")
    print(f"{'=' * 60}")

    output = {
        "timestamp": timestamp,
        "overall_score": overall,
        "tier1": t1_result,
        "tier2": t2_result,
        "tier3": t3_result,
    }

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(output, f, indent=2)
    print(f"\nResults saved to: {output_path}")


if __name__ == "__main__":
    main()
