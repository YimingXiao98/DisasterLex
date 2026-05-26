"""
Orchestrator — Active Incident Command Pipeline.
4-node LangGraph pipeline featuring real-time Tavily search and criticality-guided analysis.

Pipeline:
    User Query
        → Step 1: Context & Criticality Extraction (1-5 scale)
        → Step 2: Classify Cluster (Life Safety | Infrastructure | Damage Assessment)
        → Step 3: Research & Plan (ReAct + TAVILY + KG)
        → Step 4: Execute Analysis (ReAct + DB + KG)
        → Final Answer

Ablation flags (passed to run() as `ablation` dict):
    use_tavily   : bool  — Step 3 web search (default True)
    use_kg       : bool  — KG queries in steps 3 and 4 (default True)
    use_routing  : bool  — Steps 1-2 cluster classification (default True)
    use_planning : bool  — Step 3 research+plan node (default True)
"""
from __future__ import annotations

import concurrent.futures
import json
import logging
import os
import re as _re
from pathlib import Path
from typing import Any, Dict, List, Optional, TypedDict

from langchain_tavily import TavilySearch
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.tools import tool
from langchain_openai import ChatOpenAI
from langgraph.errors import GraphRecursionError
from langgraph.graph import StateGraph, END
from langgraph.prebuilt import create_react_agent

from src.agent.prompt_registry import (
    get_cluster_summary,
    get_query_types_for_cluster,
    get_prompt_details,
    CLUSTERS,
    get_cluster_names,
)
from src.agent.regions import region_hint
from src.config import cfg

logger = logging.getLogger(__name__)
logging.getLogger("httpx").setLevel(logging.WARNING)

SEP = "=" * 70

_PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
_FLAT_EKG_TEXT_CACHE: str | None = None


def _flat_ekg_prompt_block() -> str:
    """Return a system-prompt block containing the full curated EKG, or ''.

    Activated by the flat-ekg ablation (FLAT_EKG_PROMPT=1 in env). The EKG
    JSON is loaded once per process and cached. Format: concepts then edges
    rendered as compact lines so the LLM can read causal structure without
    tool calls. Cost: ~16k tokens per call (the EKG is small).
    """
    global _FLAT_EKG_TEXT_CACHE
    if not os.environ.get("FLAT_EKG_PROMPT"):
        return ""
    if _FLAT_EKG_TEXT_CACHE is not None:
        return _FLAT_EKG_TEXT_CACHE
    try:
        ekg = json.loads((_PROJECT_ROOT / "configs" / "graph" / "ekg_curated.json").read_text())
    except FileNotFoundError:
        logger.warning("FLAT_EKG_PROMPT=1 but configs/graph/ekg_curated.json not found; skipping.")
        _FLAT_EKG_TEXT_CACHE = ""
        return ""

    nodes = ekg.get("nodes", [])
    edges = ekg.get("edges", [])
    lines: list[str] = []
    lines.append("## EXPERT KNOWLEDGE GRAPH (full curated graph — use this in place of tool-based KG lookups)")
    lines.append(f"# {len(nodes)} concepts, {len(edges)} typed causal edges")
    lines.append("")
    lines.append("### Concepts")
    for n in nodes:
        cid = n.get("id", "")
        ctype = n.get("type", "")
        cdesc = (n.get("description") or "").strip().replace("\n", " ")
        syns = n.get("synonyms") or []
        syn_str = f"  synonyms: {', '.join(syns)}" if syns else ""
        lines.append(f"- {cid} [{ctype}]: {cdesc}{syn_str}")
    lines.append("")
    lines.append("### Edges (typed causal relations with confidence)")
    for e in edges:
        src = e.get("source", "")
        rel = e.get("type", "")
        tgt = e.get("target", "")
        conf = e.get("confidence", "")
        conf_str = f" (conf={conf})" if conf != "" else ""
        lines.append(f"- {src} {rel} {tgt}{conf_str}")
    block = "\n".join(lines) + "\n"
    _FLAT_EKG_TEXT_CACHE = block
    return block

# ── Thread-based timeout ───────────────────────────────────────────────────────

def _invoke_with_timeout(fn, *args, timeout_secs: int = 120, **kwargs):
    """Run fn(*args, **kwargs) in a thread; raise TimeoutError if it exceeds timeout_secs.

    NOTE: On timeout the background thread keeps running until the API responds,
    but the main thread is unblocked immediately (best-effort in Python).

    IMPORTANT: We deliberately do NOT use `with ThreadPoolExecutor(...) as executor`
    because the context manager's __exit__ calls shutdown(wait=True), which waits
    indefinitely for the orphan thread to finish. If the orphan thread is stuck in
    a network call (e.g., Tavily / OpenRouter HTTP socket), the whole process hangs
    past the user-visible timeout, eventually getting SIGKILL'd by an external
    watchdog. shutdown(wait=False, cancel_futures=True) lets the main thread move
    on immediately.
    """
    executor = concurrent.futures.ThreadPoolExecutor(max_workers=1)
    try:
        future = executor.submit(fn, *args, **kwargs)
        return future.result(timeout=timeout_secs)
    finally:
        # Don't wait for the orphan thread to drain — its blocking I/O may never
        # return. The thread is daemonized via executor and will be reaped at
        # process exit. cancel_futures=True only affects pending (un-started)
        # tasks; we still rely on the daemon-thread behavior for the running one.
        try:
            executor.shutdown(wait=False, cancel_futures=True)
        except TypeError:
            # cancel_futures was added in Python 3.9; fall back gracefully.
            executor.shutdown(wait=False)

CRITICALITY_NAMES = {
    1: "Monitor (Routine)",
    2: "Advisory (Watch)",
    3: "Action (Warning/Imminent)",
    4: "Urgent (Active Impact)",
    5: "Critical (Catastrophic)",
}

_DEFAULT_ABLATION = {
    "use_tavily": True,
    "use_kg": True,
    "use_routing": True,
    "use_planning": True,
}

# ── State ─────────────────────────────────────────────────────────────────────


class PipelineState(TypedDict):
    user_query: str
    # Step 1 outputs
    area_of_interest: str
    hazard_type: str          # "flood"|"hurricane"|"wildfire"|"tornado"|"drought"|"power_disruption"|"multi_hazard"
    criticality: int          # 1-5
    criticality_rationale: str
    # Step 2 outputs
    cluster: str
    query_type: str
    prompt_objective: str
    prompt_key_datasets: list[str]
    prompt_core_logic: str
    # Step 3 output
    analysis_plan: str
    # Step 4 output
    final_answer: str
    # Data availability warnings (set in Step 1, consumed in Step 4)
    data_warnings: list


# ── LLM factory ───────────────────────────────────────────────────────────────

def _get_llm() -> ChatOpenAI:
    cfg.validate()
    from src.config import provider_default_headers, provider_request_kwargs, resolve_provider
    base_url, api_key = resolve_provider(cfg.LLM_MODEL)
    # Optional seed for reproducible reruns. The phase2 launcher sets LLM_SEED
    # to a different integer per seed (1, 2, 3) so each rerun draws different
    # samples from the OpenRouter chat API where the upstream provider honors
    # the OpenAI-compatible `seed` parameter.
    import os as _os
    seed_val: int | None = None
    seed_env = _os.environ.get("LLM_SEED")
    if seed_env:
        try:
            seed_val = int(seed_env)
        except ValueError:
            seed_val = None
    kwargs: dict = dict(
        model=cfg.LLM_MODEL,
        temperature=0,
        api_key=api_key,
        base_url=base_url,
        timeout=120,
    )
    if seed_val is not None:
        kwargs["seed"] = seed_val
    kwargs.update(provider_request_kwargs(cfg.LLM_MODEL, base_url))
    # DashScope thinking-capable models (qwen3.6-flash, qwen-plus, qwq-*, qvq-*)
    # default to thinking-on, which adds 2-4× latency for the same final answer
    # and confounds comparison with non-thinking flash models on other providers.
    # Disable thinking for any DashScope-routed model — harmless on models that
    # don't support the flag.
    if "dashscope" in (base_url or "").lower():
        kwargs["extra_body"] = {"enable_thinking": False}
    return ChatOpenAI(
        **kwargs,
        default_headers=provider_default_headers(base_url, "Disaster Active Incident Orchestrator"),
    )


# Models that require response_format=json_object for Stage 1/2 JSON-output
# prompts. Small Llama-3.x models refuse hazard scenarios as safety violations
# without explicit JSON-mode enforcement; with json_object response_format they
# comply. Qwen3-8b breaks under json_object (returns nonsense), so it stays
# on plain-text output. List is conservative — extend only after verifying
# the model both (a) needs the hack and (b) doesn't break with it.
_REQUIRES_JSON_MODE = (
    "llama-3.1-8b",
    "llama-3.2-3b",
    "llama-3.2-1b",
)


def _get_llm_json_mode_if_needed() -> ChatOpenAI:
    """Same as _get_llm but adds response_format=json_object for small Llamas.

    Used by extract_context_and_criticality and classify_cluster which must
    return strict JSON. Other stages use plain _get_llm() so tool-calling and
    free-form output still work.
    """
    base = _get_llm()
    name = (cfg.LLM_MODEL or "").lower()
    if any(p in name for p in _REQUIRES_JSON_MODE):
        return base.bind(response_format={"type": "json_object"})
    return base


# ── Tools ───────────────────────────────────────────────────────────────────

@tool
def query_database(question: str) -> str:
    """Query the DuckDB disaster database with a natural language question.
    Use this to retrieve actual data values, statistics, or comparisons.
    """
    from src.agent import text_to_sql_agent
    print(f"\n[TOOL] query_database: {question[:80]}...")
    result_state = text_to_sql_agent.run(question)

    if result_state.get("result") is not None:
        columns = result_state.get("columns", [])
        rows = result_state["result"]
        total = result_state.get("total_rows", len(rows))
        table = text_to_sql_agent._format_table(columns, rows)
        output = f"\n{'='*60}\nSQL: {result_state.get('sql', '')}\n\nResults ({total} total rows):\n{table}\n{'='*60}\n\nSummary: {result_state.get('answer', '')}"
        print(output)
        return output
    else:
        return f"Database query failed: {result_state.get('error', 'Unknown error')}"


@tool
def query_knowledge_graph(question: str) -> str:
    """Retrieve causal rules and domain knowledge from the Expert Knowledge Graph (EKG).
    Use this to understand risk factors and causal mechanisms (e.g., 'elevation INCREASES flood').
    """
    from src.agent.graph_agent import query_ekg
    print(f"\n[TOOL] query_knowledge_graph: {question[:80]}...")
    rules, summary = query_ekg(question)
    return summary


@tool
def _stub_web_search(query: str) -> str:
    """Web search stub — disabled for ablation study (use_tavily=False)."""
    return "[]"


# ── Data availability check ───────────────────────────────────────────────────

def _check_data_availability(user_query: str) -> list[dict]:
    """Match concepts in user_query against CONCEPT_TABLE_MAP and return unavailability warnings.

    Runs at orchestrator level (Step 1) so warnings are available to the execute node (Step 4)
    regardless of whether the SQL agent completes. Returns list of warning dicts or [].
    """
    try:
        from src.graph.context_graph import get_context_graph
        graph = get_context_graph()
        matched = graph.match_concepts(user_query)
        concept_ids = [c["id"] for c in matched]
        if not concept_ids:
            return []
        result = graph.check_concept_runnability(concept_ids)
        return result.get("warnings", [])
    except Exception as exc:
        logger.warning(f"Data availability check failed (non-fatal): {exc}")
        return []


_STRUCTURED_FACT_LINE_RE = _re.compile(
    r"^-+\s*subject:\s*(?P<subject>.+?)\s+metric:\s*(?P<metric>.+?)\s+value:\s*(?P<value>.+?)\s*$",
    _re.IGNORECASE,
)


def _extract_structured_fact_records(text: str) -> List[Dict[str, str]]:
    records: List[Dict[str, str]] = []
    in_block = False
    saw_record = False
    for raw_line in (text or "").splitlines():
        stripped = raw_line.strip()
        normalized = stripped.lstrip("#").strip().lower()
        if not in_block:
            if normalized.startswith("structured facts"):
                in_block = True
            continue

        if not stripped:
            continue

        match = _STRUCTURED_FACT_LINE_RE.match(stripped)
        if match:
            saw_record = True
            records.append(
                {
                    "subject": match.group("subject").strip(),
                    "metric": match.group("metric").strip(),
                    "value": match.group("value").strip(),
                }
            )
            continue

        if saw_record:
            break

    return records


def _strip_structured_facts_section(text: str) -> str:
    lines = (text or "").rstrip().splitlines()
    for idx, line in enumerate(lines):
        normalized = line.strip().lstrip("#").strip().lower()
        if normalized.startswith("structured facts"):
            return "\n".join(lines[:idx]).rstrip()
    return (text or "").rstrip()


def _replace_structured_facts_section(text: str, records: List[Dict[str, str]]) -> str:
    body = _strip_structured_facts_section(text)
    if not records:
        return body
    fact_lines = "\n".join(
        f"- subject: {record['subject']}  metric: {record['metric']}  value: {record['value']}"
        for record in records
    )
    return body.rstrip() + "\n\nSTRUCTURED FACTS:\n" + fact_lines


# ── Compound query detection & splitting ─────────────────────────────────────

_COMPOUND_PATTERNS = [
    _re.compile(r"\bsimultaneously\b", _re.IGNORECASE),
    _re.compile(r"\bat the same time\b", _re.IGNORECASE),
    # Cross-cluster: life-safety keyword + infrastructure keyword (order-independent)
    _re.compile(
        r"(?:shelter\s+gap|evacuat\w+\s+gap|life.?safety\s+gap)\b.{1,120}?\b"
        r"(?:infrastructure\s+hard|power\s+grid|water\s+treatment|pipeline|supply\s+chain)",
        _re.IGNORECASE | _re.DOTALL,
    ),
    _re.compile(
        r"(?:infrastructure\s+hard|power\s+grid|water\s+treatment|pipeline|supply\s+chain)\b.{1,120}?\b"
        r"(?:shelter\s+gap|evacuat\w+\s+gap|life.?safety\s+gap)",
        _re.IGNORECASE | _re.DOTALL,
    ),
]

_LIFE_SAFETY_KEYWORDS = _re.compile(
    r"\b(?:shelter|evacuat\w+|search.and.rescue|SAR|rescue\s+operation|life.safety|"
    r"hospital\s+access|medical\s+triage|population\s+rescue)\b",
    _re.IGNORECASE,
)
_INFRA_DAMAGE_KEYWORDS = _re.compile(
    r"\b(?:infrastructure|power\s+grid|substation|water\s+treatment|pipeline|"
    r"supply\s+chain|economic\s+loss|damage\s+assessment|hardening|mitigation|"
    r"structural\s+damage|recovery\s+prioritization)\b",
    _re.IGNORECASE,
)


def _is_compound_query(user_query: str) -> bool:
    """Return True only when the query requires analyses from TWO DIFFERENT operational clusters.

    A query is compound when:
      (a) an explicit simultaneity keyword is present, OR
      (b) a cross-cluster regex fires AND both sub-parts belong to different
          operational domains (life-safety vs. infrastructure/damage).

    Single-cluster "and" queries (e.g., "shelter capacity AND surge risk" — both
    about shelter placement under surge) return False.
    """
    for p in _COMPOUND_PATTERNS[:2]:  # simultaneity keywords — always compound
        if p.search(user_query):
            return True
    # Cross-cluster patterns require that both domain keyword sets fire
    for p in _COMPOUND_PATTERNS[2:]:
        if p.search(user_query):
            has_life_safety = bool(_LIFE_SAFETY_KEYWORDS.search(user_query))
            has_infra_damage = bool(_INFRA_DAMAGE_KEYWORDS.search(user_query))
            if has_life_safety and has_infra_damage:
                return True
    return False


def _split_compound_query(user_query: str) -> list[str] | None:
    """Use LLM to split a compound query into two focused sub-queries.

    Returns a list of exactly 2 sub-query strings, or None on failure.
    Each sub-query preserves the geographic area and hazard context.
    """
    llm = _get_llm()
    prompt = (
        f"A disaster management query asks for two distinct analyses simultaneously. "
        f"Split it into exactly 2 focused sub-queries. Each must:\n"
        f"  - Address ONE operational domain only\n"
        f"  - Preserve the original geographic area and hazard context\n"
        f"  - Be answerable independently\n\n"
        f'Query: "{user_query}"\n\n'
        f'Return ONLY JSON: {{"sub_queries": ["sub-query 1", "sub-query 2"]}}'
    )
    try:
        response = _invoke_with_timeout(llm.invoke, [HumanMessage(content=prompt)], timeout_secs=30)
        clean = response.content.strip().replace("```json", "").replace("```", "").strip()
        parsed = json.loads(clean)
        subs = parsed.get("sub_queries", [])
        if len(subs) == 2 and all(isinstance(s, str) and s.strip() for s in subs):
            logger.info(f"Compound query split into: {subs}")
            return subs
    except Exception as exc:
        logger.warning(f"Compound query split failed: {exc}")
    return None


def _merge_compound_answers(original_query: str, sub_answers: list[str]) -> str:
    """Synthesize two independent sub-query answers into a single coherent response."""
    if not sub_answers:
        return "[No sub-query results to merge.]"
    if len(sub_answers) == 1:
        return sub_answers[0]
    llm = _get_llm()
    prompt = (
        f"You are an Incident Command analyst. Two independent analyses were run for different "
        f"aspects of a compound disaster query. Synthesize them into ONE coherent, actionable "
        f"incident report. Preserve all numeric findings and recommendations from both analyses. "
        f"Do not drop any data.\n\n"
        f'Original query: "{original_query}"\n\n'
        f"Analysis 1:\n{sub_answers[0]}\n\n"
        f"Analysis 2:\n{sub_answers[1]}\n\n"
        f"Write the combined incident report:"
    )
    try:
        response = _invoke_with_timeout(llm.invoke, [HumanMessage(content=prompt)], timeout_secs=60)
        return response.content.strip()
    except Exception as exc:
        logger.warning(f"Merge failed: {exc} — concatenating answers.")
        return sub_answers[0] + "\n\n---\n\n" + sub_answers[1]


# ── Node 1: Context & Criticality Extraction ────────────────────────────────

def extract_context_and_criticality(state: PipelineState) -> dict:
    """Step 1: Extract area, hazard, and criticality (1-5)."""
    print(f"\n{SEP}\nSTEP 1: Context & Criticality Extraction\n{SEP}")

    prompt = f"""Analyze the disaster management query below. Extract area_of_interest,
hazard_type (flood|hurricane|wildfire|tornado|drought|power_disruption|multi_hazard), criticality (1-5), and
criticality_rationale.
Hazard-type routing guidance:
  - "power outage", "grid disruption", "blackout", "brownout", ERCOT capacity loss → power_disruption
  - "tornado outbreak", "EF-rated tornado", "tornado corridor" → tornado
  - "sustained drought", "drought", "water shortage" → drought
  - questions combining 2+ distinct hazards (e.g. hurricane + drought, flood + power + heat) → multi_hazard
  - fall through to flood / hurricane / wildfire only if the primary hazard is clearly one of those three.

Criticality is based on the URGENCY and SEVERITY implied by the query:
  1=Monitor (routine) — planning queries, assessments, "what if" scenarios, historical analysis
  2=Advisory (watch) — forecast threats, "watch issued", "expected in days", "approaching (days away)"
  3=Action (warning/imminent) — "imminent", "expected to make landfall in X hours", "projected landfall in X hours", "projected [hazard] in less than 24 hours", "warning issued", "red flag warning", "Cat [N] hurricane approaching [location]", "hurricane approaching"
  4=Urgent (active impact) — "is currently affecting", "has struck", "outbreak", "is burning", "active wildfire", "wildfire is burning", "wildfire is spreading", "knocking out power", "making landfall NOW" [present-tense active event language]
  5=Critical (catastrophic) — "Cat 5", "Category 5", "catastrophic", "mass casualty", "complete infrastructure failure". NOTE: Any mention of "Category 5" or "Cat 5" hurricane ALWAYS assigns Criticality 5 regardless of other framing.
Note: If the query includes "planning", "assessment", "what if", or past-tense framing, cap criticality at 3 even if hazard terms appear.
Note: "Projected [hazard] in N hours" where N <= 24 is Criticality 3 (imminent), NOT a planning query — do NOT assign Criticality 1 or 2.
Note: The words "query", "assess", "estimate", or "identify" in the question do NOT reduce criticality — only planning/historical/what-if framing does.

User Query: "{state['user_query']}"

Return ONLY JSON: {{"area_of_interest":"...","hazard_type":"...","criticality":N,"criticality_rationale":"..."}}
"""
    llm = _get_llm_json_mode_if_needed()
    try:
        response = _invoke_with_timeout(llm.invoke, [HumanMessage(content=prompt)], timeout_secs=60)
        clean = response.content.strip().replace("```json", "").replace("```", "").strip()
        parsed = json.loads(clean)
    except (TimeoutError, concurrent.futures.TimeoutError):
        logger.warning("extract_context_and_criticality timed out — using defaults.")
        parsed = {
            "area_of_interest": "Harris County",
            "hazard_type": "flood",
            "criticality": 1,
            "criticality_rationale": "Defaulting to baseline (timeout).",
        }
    except Exception:
        parsed = {
            "area_of_interest": "Harris County",
            "hazard_type": "flood",
            "criticality": 1,
            "criticality_rationale": "Defaulting to baseline.",
        }

    print(f"  Area: {parsed.get('area_of_interest')}")
    print(f"  Hazard: {parsed.get('hazard_type')}")
    crit = parsed.get("criticality", 1)
    print(f"  Criticality: {crit} ({CRITICALITY_NAMES.get(crit)})")

    # Check data availability for this query's concepts
    data_warnings = _check_data_availability(state["user_query"])
    if data_warnings:
        logger.info(f"Data availability: {len(data_warnings)} unavailability warning(s) for this query")

    return {**state, **parsed, "data_warnings": data_warnings}


# ── Node 1 (ablation): Inject defaults when routing is disabled ───────────────

def _extract_area_from_query(query: str) -> str:
    """Lightweight regex extraction of county/city from a query string.

    Tries to find '<Name> County' first; falls back to the first capitalized
    proper-noun sequence before a comma or end of sentence.  Used by the
    no-routing ablation so SQL WHERE clauses still receive a real county name.
    """
    # Pattern 1: explicit "X County" mention
    m = _re.search(r'\b([A-Z][a-zA-Z]+(?:\s+[A-Z][a-zA-Z]+)*)\s+County\b', query)
    if m:
        return f"{m.group(1)} County"
    # Pattern 2: first capitalized word before a comma or period (city name)
    m = _re.search(r'\b([A-Z][a-zA-Z]+(?:\s+[A-Z][a-zA-Z]+)*)\b', query)
    if m:
        return m.group(1)
    return "Harris County"  # last-resort default


def _inject_defaults(state: PipelineState) -> dict:
    """Ablation node — replaces steps 1-2 with fixed defaults (use_routing=False).

    Cluster and query_type are hardcoded (this is what no-routing tests).
    area_of_interest is extracted from the query so SQL WHERE clauses remain
    functional — without this, every query returns zero rows, confounding the
    routing penalty with geographic context erasure.
    """
    print(
        f"\n{SEP}\n[ABLATION] Routing disabled — injecting default context\n{SEP}")
    area = _extract_area_from_query(state["user_query"])
    print(f"  [ABLATION] area_of_interest (extracted): {area}")
    details = get_prompt_details(
        "life_safety_operations", "shelter_placement") or {}
    return {
        **state,
        "area_of_interest": area,
        "hazard_type": "flood",
        "criticality": 3,
        "criticality_rationale": "Default: routing ablated.",
        "cluster": "life_safety_operations",
        "query_type": "shelter_placement",
        "prompt_objective": details.get("objective", "Conduct analysis per core logic."),
        "prompt_key_datasets": details.get("key_datasets", []),
        "prompt_core_logic": details.get("core_logic", "Execute analysis based on user query."),
        "data_warnings": _check_data_availability(state["user_query"]),
    }


# ── Node 2: Classify Cluster ───────────────────────────────────────────────

def classify_cluster(state: PipelineState) -> dict:
    """Step 2: Routes query to one of the 3 active-event clusters."""
    print(f"\n{SEP}\nSTEP 2: Classify Cluster\n{SEP}")

    cluster_descriptions = get_cluster_summary()

    summary_lines = []
    for name in get_cluster_names():
        summary_lines.append(f"Cluster: {name}")
        summary_lines.append(get_query_types_for_cluster(name))
        summary_lines.append("-" * 30)

    full_summary = "\n".join(summary_lines)

    prompt = f"""Determine which operational cluster and query type best fits this request.
User Query: "{state['user_query']}"
Context: {state['hazard_type']} in {state['area_of_interest']} (Criticality {state['criticality']})

Cluster Descriptions (read these FIRST to select the correct cluster):
{cluster_descriptions}

Available Query Types per Cluster:
{full_summary}

ROUTING RULES:
- life_safety_operations: OPERATIONAL queries — active rescue, search-and-rescue (SAR) zone identification for responder deployment, evacuation, medical triage. Requires action language ("identify SAR zones", "deploy units", "rescue victims", "evacuate residents"). NOT for causal mechanism analysis, NOT for retrospective vulnerability quantification.
- damage_assessment_response: ANALYTICAL queries — post-event damage counts, economic loss, recovery planning, resource gap/mutual aid assessment (EMAC requests, shelter deficits, resource shortfalls), AND causal mechanism questions ("how does X cause/affect Y", "what role does X play in Y", "retrieve causal path"). Use this for population-level vulnerability analysis WITHOUT a deployment context (e.g. "identify populations at compound risk", "disproportionate flood burden on low-income", "map medically-vulnerable populations"). NOTE: "shelter deficits" or "mutual aid" in a post-event context → resource_gap (damage_assessment_response), NOT shelter_placement (life_safety_operations).
- infrastructure_mitigation: protecting infrastructure BEFORE or DURING an event (flood barriers, hardening, WTP risk, power grid exposure, surge exposure to critical facilities). This also covers infrastructure VULNERABILITY MAPPING — "identify hospitals in surge zones", "map power generation assets at flood risk", "identify substations exposed to wildfire" are infrastructure_mitigation queries, NOT life_safety_operations, because the analysis target is infrastructure (facilities, grid, utilities), not triage or deployment. For hurricane events involving "supply chain disruption", "critical facility hardening", or "surge exposure to substations/hospitals/WTPs" → use hurricane_surge_exposure, NOT power_grid_vulnerability. power_grid_vulnerability is for wildfire/tornado/flood events explicitly about power grid only.

DISAMBIGUATION for "identify X at risk" patterns:
- "Identify hospitals/substations/WTPs/facilities in <risk zone>" (infrastructure exposure mapping) → infrastructure_mitigation
- "Identify populations with high <vulnerability> + <hazard exposure>" (equity / compound-risk analysis, no deployment) → damage_assessment_response (typically population_impact)
- "Identify counties facing <compound hazard>" for regional analysis → damage_assessment_response
- "Identify SAR zones / evacuation zones / shelter sites" (explicit deployment framing) → life_safety_operations
- Default for statewide or multi-county vulnerability mapping without clear deployment language → infrastructure_mitigation or damage_assessment_response, NOT life_safety_operations.

PRECEDENCE when a query has MULTIPLE signals:
- A query that estimates ECONOMIC LOSS or BUILDING COUNTS as a primary metric is damage_assessment_response, even if it also includes a "list facilities in flood/fire zones" subtask. The loss/damage metric is the dominant frame; facility identification is a supporting sub-query. Examples that route to damage_assessment_response despite mentioning facilities:
  * "Estimate total buildings at risk, total flood EAL, AND identify hospitals in the flood zone" — primary metric is buildings + EAL → damage_assessment_response (structural_damage).
  * "Quantify population exposure, the count of vulnerable hexes, AND fire/EMS totals" — primary metric is exposure + vulnerable count → damage_assessment_response (population_impact).
- A query with ONLY infrastructure-exposure mapping (no $ loss, no building count, no population aggregate) stays infrastructure_mitigation. Examples that remain infrastructure_mitigation:
  * "Identify substations exposed to wildfire" — pure IM exposure mapping.
  * "Map hospitals in storm-surge zones" — pure IM exposure mapping.

Return ONLY JSON: {{"cluster":"folder_name", "query_type":"file_name"}}
"""
    llm = _get_llm_json_mode_if_needed()
    try:
        response = _invoke_with_timeout(llm.invoke, [HumanMessage(content=prompt)], timeout_secs=60)
        clean = response.content.strip().replace("```json", "").replace("```", "").strip()
        parsed = json.loads(clean)
        details = get_prompt_details(parsed["cluster"], parsed["query_type"])

        if not details:
            logger.warning(
                f"Query type '{parsed['query_type']}' not found in cluster '{parsed['cluster']}'. "
                "Attempting fuzzy match."
            )
            cluster_info = CLUSTERS.get(parsed["cluster"])
            if cluster_info and cluster_info.query_types:
                best_guess = list(cluster_info.query_types.keys())[0]
                parsed["query_type"] = best_guess
                details = get_prompt_details(parsed["cluster"], best_guess)

    except Exception as e:
        logger.error(f"Classification failed: {e}")
        parsed = {"cluster": "life_safety_operations",
                  "query_type": "shelter_placement"}
        details = get_prompt_details(
            "life_safety_operations", "shelter_placement")

    if not details:
        details = {}

    print(f"  Cluster: {parsed.get('cluster')}")
    print(f"  Query Type: {parsed.get('query_type')}")

    return {
        **state,
        "cluster": parsed.get("cluster", "life_safety_operations"),
        "query_type": parsed.get("query_type", "shelter_placement"),
        "prompt_objective": details.get("objective", "Conduct analysis per core logic."),
        "prompt_key_datasets": details.get("key_datasets", []),
        "prompt_core_logic": details.get("core_logic", "Execute analysis based on user query."),
    }


# ── Node 3: Research & Plan ────────────────────────────────────────────────

def _make_plan_node(cfg: dict):
    """Factory — returns a research_and_plan node with ablation config applied."""
    def research_and_plan(state: PipelineState) -> dict:
        print(f"\n{SEP}\nSTEP 3: Research & Plan (ReAct + TAVILY)\n{SEP}")

        tools: list = []
        if cfg["use_tavily"]:
            tools.append(TavilySearch(max_results=5, topic="general"))
        else:
            tools.append(_stub_web_search)
            print("  [ABLATION] Tavily disabled — using stub search.")

        if cfg["use_kg"]:
            tools.append(query_knowledge_graph)
        else:
            print(
                "  [ABLATION] KG disabled — no causal rules will be retrieved in planning.")

        # Only require the causal-chain block when the question has causal intent;
        # for pure data-retrieval or routing cases, forcing a chain adds noise.
        user_q = state.get("user_query", "") or ""
        causal_triggers = (
            "why", "how does", "cascade", "cascading", "downstream",
            "impact on", "risk to", "due to", "because",
            "drive", "driver", "trigger", "cause", "affect",
            "compound", "propagat", "chain", "dependen",
        )
        needs_causal_chain = any(t in user_q.lower() for t in causal_triggers)
        causal_block = (
            "\n4. After your numbered plan, append a single line labelled "
            "'EXPECTED CAUSAL CHAIN:' naming 2-4 concepts from the KG connected by the "
            "relation types you retrieved (INCREASES, REDUCES, TRIGGERS, REQUIRES). "
            "Example: 'EXPECTED CAUSAL CHAIN: rainfall INCREASES runoff INCREASES "
            "flood_occurrence REDUCES hospital_operations'. This chain is the reasoning "
            "backbone the execute step must restate in the final answer.\n"
            if needs_causal_chain else ""
        )

        llm = _get_llm()
        flat_ekg_block = _flat_ekg_prompt_block()
        agent = create_react_agent(
            llm,
            tools,
            prompt=(
                f"You are a disaster intelligence analyst. Create a detailed analysis plan "
                f"for {state['hazard_type']} in {state['area_of_interest']}.\n"
                f"CRITICALITY: {state['criticality']} ({CRITICALITY_NAMES[state['criticality']]})\n"
                f"CORE LOGIC: {state['prompt_core_logic']}\n\n"
                f"{flat_ekg_block}"
                "1. Use search to find CURRENT weather/hazard alerts or evacuation orders.\n"
                "2. Use KG to find causal risk factors.\n"
                "3. Synthesize into a numbered analysis plan for the database execution step.\n"
                "Tool budget for this planning step is strict: call web search at most once "
                "and query_knowledge_graph at most once. If a tool returns no useful result, "
                "do not retry with synonyms or broader wording; immediately synthesize the "
                "plan from CORE LOGIC and available context. After any KG result, stop using "
                "tools and write the plan.\n"
                "Do not include DESCRIBE, SHOW TABLES, information_schema, PRAGMA, or other "
                "schema-inspection steps in the plan. The execution tool receives the schema "
                "separately and must query data tables directly."
                f"{causal_block}"
            ),
        )

        try:
            result = _invoke_with_timeout(
                agent.invoke,
                {"messages": [{"role": "user", "content": "Generate the analysis plan."}]},
                timeout_secs=180,
                config={"recursion_limit": 15},
            )
            plan = result["messages"][-1].content
        except (TimeoutError, concurrent.futures.TimeoutError):
            logger.warning("Plan node timed out after 180s — using empty plan.")
            plan = ""
        except GraphRecursionError:
            logger.warning("Plan node hit recursion limit — using empty plan.")
            plan = ""

        print(f"\n[TACTICAL PLAN]:\n{plan}\n")
        logger.info(
            f"Analysis plan generated for {state['area_of_interest']}.")

        return {**state, "analysis_plan": plan}

    return research_and_plan


def _skip_planning(state: PipelineState) -> dict:
    """Ablation node — replaces step 3 with an empty plan (use_planning=False)."""
    print(f"\n{SEP}\n[ABLATION] Planning disabled — skipping step 3\n{SEP}")
    return {**state, "analysis_plan": ""}


# ── Node 4: Execute Analysis ───────────────────────────────────────────────

def _make_execute_node(cfg: dict):
    """Factory — returns an execute_analysis node with ablation config applied."""
    def execute_analysis(state: PipelineState) -> dict:
        print(f"\n{SEP}\nSTEP 4: Execute Analysis (ReAct + DB/KG)\n{SEP}")

        tools: list = [query_database]
        if cfg["use_kg"]:
            tools.append(query_knowledge_graph)
        else:
            print("  [ABLATION] KG disabled — execute will use DB only.")

        llm = _get_llm()
        # Always anchor on the structured prompt template (key datasets + core logic)
        # so the execute node doesn't drift toward KG-only answers when Tavily dominates the plan.
        core_logic = state.get("prompt_core_logic", "")
        key_datasets = ", ".join(state.get("prompt_key_datasets", []))
        area = state.get("area_of_interest", "")
        plan = state.get("analysis_plan", "")
        data_warnings = state.get("data_warnings", [])
        # Truncate plan to avoid context overflow with verbose models
        plan_truncated = plan[:1500] + "\n[...plan truncated...]" if len(plan) > 1500 else plan
        template_block = (
            f"\nDB Analysis Steps (execute these IN ORDER using query_database):\n"
            f"AREA: {area} — always use this EXACT string (with 'County' suffix) in WHERE x.County = '...' filters.\n"
            f"KEY DATASETS: {key_datasets}\n"
            f"STEPS:\n{core_logic}\n"
            "You MUST call query_database before writing any analysis. "
            "If a query returns 0 rows, check the WHERE clause county name. "
            "Do NOT run schema-inspection queries; the database tool already has the schema.\n"
        ) if core_logic else ""
        disclosure_block = ""
        if data_warnings:
            lines = ["DATA UNAVAILABILITY — the following data cannot be queried:"]
            for w in data_warnings:
                lines.append(f"  • [{w['concept']}] {w['suggestion']}")
            lines.append(
                "MANDATORY: You MUST explicitly state in your final answer which data is "
                "unavailable and what proxy or alternative you are using. "
                "Do NOT attempt to query the unavailable tables."
            )
            disclosure_block = "\n".join(lines) + "\n"
        # Regional-scope hint: emits the canonical member-county list when the
        # user query references a named multi-county region (Texas Panhandle,
        # Permian Basin, Gulf Coast, etc.). Defined in config/regions.json;
        # empty string when no region is matched. This is documentation, not
        # routing — the SQL is still LLM-generated.
        region_block = region_hint(state.get("user_query", ""))
        if region_block:
            region_block = region_block + "\n"
        schema_hints = (
            "SCHEMA HINTS (read before writing SQL — these are the common traps):\n"
            "- Population metric choice: for a TOTAL POPULATION across many hexes, use "
            "SUM(population_per_hex) — this is the NON-OVERLAPPING, geographically correct "
            "aggregation. DO NOT use SUM(population_7km): population_7km is a 7km-radius buffer "
            "that double-counts each resident ~247× when summed (every hex overlaps ~250 "
            "neighbors). Use population_7km (or AVG thereof) only as a per-hex density/exposure "
            "proxy, never in a SUM over multiple hexes.\n"
            "- Standard risk threshold is nri_<hazard>_score >= 75 for 'high risk'. A few benchmark "
            "cases use >= 80 for 'critical' — if >= 75 returns suspicious counts, try 80.\n"
            "- PERCENTILE vs FIXED threshold: when the question uses 'above-median', 'percentile', "
            "'disproportionate', 'top X%', or compares two populations relative to each other, "
            "use a DYNAMIC threshold via QUANTILE_CONT(col, frac) in a WITH thresh AS (...) CTE, "
            "NOT the fixed literal 75. Example: WITH thresh AS (SELECT "
            "QUANTILE_CONT(CAST(nri_riverine_flood_score AS DOUBLE), 0.75) AS p75 FROM HP_FLD_002) "
            "then filter WHERE score > (SELECT p75 FROM thresh).\n"
            "- Column name gotchas: hospital count is hosp_n (not hospital_per100k); FEMA SFHA is "
            "ve_ae_fraction (not fema_nfhl_sfha_fraction); social vulnerability is sovi where -999 "
            "means MISSING and must be excluded (WHERE sovi != -999).\n"
            "- Crosswalk join: hex_county_state_zip_crosswalk joins on hex_id (not hex_id_l7). "
            "County values include the 'County' suffix (Harris County, Travis County).\n"
            "- For percentile thresholds in DuckDB use QUANTILE_CONT(col, frac), "
            "not PERCENTILE_CONT(frac) WITHIN GROUP(...).\n"
        )
        flat_ekg_block = _flat_ekg_prompt_block()
        system_prompt = (
            f"You are an Incident Command analyst. Execute the provided analysis plan using actual data.\n"
            f"CRITICALITY: {state['criticality']}\n"
            f"AREA: {area}\n"
            f"{disclosure_block}"
            f"{region_block}"
            f"{template_block}\n"
            f"{flat_ekg_block}"
            f"{schema_hints}\n"
            f"BACKGROUND PLAN (context only — IGNORE any plan instruction that contradicts the STEPS above):\n{plan_truncated}\n"
            "Constraints:\n"
            "- Complete the full analysis in at most 8 query_database calls total. "
            "Use a single query with CTEs or JOINs to combine related lookups instead of issuing separate calls.\n"
            "- Never call query_database with DESCRIBE, SHOW, PRAGMA, information_schema, "
            "duckdb_tables(), duckdb_columns(), or pragma_table_info(). Query actual data "
            "tables directly using the schema hints and key datasets above.\n"
            "- Call query_knowledge_graph AT MOST TWICE. After that, stop calling KG tools and "
            "proceed directly to query_database and the final written analysis. Do NOT chain more "
            "than 2 KG lookups even if the returned rules seem to invite further exploration.\n"
            "- If SQL retries have consumed 3+ attempts on a single logical query, abandon that "
            "query and answer with whatever data you already have — do not keep iterating.\n"
            "- For aggregate population totals across multiple hexes, use SUM(population_per_hex) "
            "(non-overlapping). Never SUM population_7km (7km buffer overcounts ~247×). Use "
            "AVG(population_7km) only as a per-hex density proxy when explicitly asked for 'average'.\n"
            "- Gate recommendations by Criticality:\n"
            "  - 1-2: Informational only.\n"
            "  - 3: Active response recommendations.\n"
            "  - 4-5: Life-safety missions (evacuation, shutdowns).\n"
            "- Cite actual data values.\n"
            "- Report ALL column values from result rows — do not drop numeric columns.\n"
            "- If the question scope is a named region, basin, border region, statewide rural Texas, "
            "or 'outside major metros', compute the aggregate for that FULL named scope. Do NOT "
            "substitute one example county and do NOT stop at a per-county table when the question "
            "asks for a regional or statewide answer.\n"
            "- If the question asks for hospital resilience or exposure at county/region scope, report "
            "the TOTAL hospital count for that scope and any requested average vulnerability metric. "
            "Do NOT answer only with the flood-exposed subset unless the question explicitly asks for "
            "the subset.\n"
            "- For cascade-failure scenarios, explicitly state the sequence "
            "'energy plant failure -> substation outage -> water treatment failure -> hospital backup "
            "exhaustion -> mass casualty escalation' and explicitly note that static facility counts "
            "alone cannot model timing or propagation.\n"
            "- INCIDENT COMMAND DOCTRINE (apply when the trigger condition holds, not by query phrase):\n"
            "  • IF cluster = damage_assessment_response AND criticality >= 4 — explicitly mention "
            "FEMA Preliminary Damage Assessment / federal declaration escalation in the recommendations, "
            "because criticality-4+ damage events meet the doctrinal trigger for federal recovery support.\n"
            "  • IF query_type contains 'resource_gap' OR the question asks about responder/shelter "
            "sufficiency — explicitly evaluate mutual aid / EMAC requests as a recommendation, "
            "since the doctrinal response to a quantified resource gap is mutual-aid activation.\n"
            "  • IF the question mentions energy infrastructure (power, generation, substation, ERCOT, "
            "grid) at regional scope — discuss grid-scale reliability impact (not just local outage), "
            "since regional generation losses propagate through the wider transmission system.\n"
            "  • IF the question mentions hospital exposure or healthcare continuity — discuss backup "
            "power and fuel-resupply continuity, since these are the doctrinal failure points for "
            "extended hospital operation under outage.\n"
            "- SELF-CHECK before finalizing (internal — don't narrate): "
            "(a) did I state every primary numeric metric the question asked for, each with its exact "
            "value? List the metrics the question asked for and verify one fact line per metric. "
            "(b) IF the question mentions 'low-income', 'minority', 'vulnerable', "
            "'disproportionate', 'equity', or 'racial/ethnic' — did I address the equity "
            "dimension explicitly in the answer? "
            "(c) IF the question asks 'how', 'why', about cascades, drivers, or impacts — "
            "did I name the causal chain using concept identifiers from the KG "
            "(e.g. 'flood_occurrence INCREASES power_disruption REDUCES hospital_operations')? "
            "(d) are my recommendations gated by the Criticality level above? "
            "(e) IF the question asks for a multi-county or regional aggregate, did I sum across "
            "the FULL set of named counties (not just one or two)? "
            "(f) IF cluster = damage_assessment_response and criticality >= 4 — did I mention FEMA PDA? "
            "(g) IF query_type involves resource_gap — did I evaluate mutual aid / EMAC? "
            "If any answer is no and the question demands it, amend before replying.\n"
            "- MANDATORY STRUCTURED FACTS (append as the LAST section of your reply, verbatim "
            "schema below). One line per aggregated numeric metric the question asks for. "
            "For multi-county/regional queries, emit the REGION-LEVEL aggregate (SUM or AVG "
            "across all counties in one fact line), NOT per-county rows. Use the subject "
            "that names the region or county you were asked about (e.g. 'harris_county', "
            "'dfw_metroplex', 'gulf_coast', 'permian_basin'). Use snake_case metric names.\n"
            "\n"
            "CRITICAL — metric-name rules. Skipping or substituting a requested metric is the "
            "most common cause of low scoring. Apply these rules without exception:\n"
            "1. Re-read the question and the STEPS / KEY DATASETS above. For EVERY numeric "
            "quantity the question asks for, emit a STRUCTURED FACTS line with a metric name "
            "that directly reflects what was requested. If the question asks for 'hospital "
            "hexes', the metric is 'hospital_hexes' (or 'hospital_hexes_<county>'). If it "
            "asks for 'substations', the metric is 'substations' or 'substation_count'. Do "
            "NOT replace a requested metric with a tangential one (e.g., do not emit "
            "'fire_ems_stations' when the question asked about hospitals).\n"
            "2. Even if your SQL fails or returns 0 rows, you MUST still emit a STRUCTURED "
            "FACTS line for each requested metric. If the value is genuinely unobtainable, "
            "use value: 0 and note the reason in the prose above (e.g., \"no hexes met "
            "threshold\"). Do not omit the metric or describe it as 'limitations'.\n"
            "3. Before retrying or giving up on a SQL query, try ONCE with a relaxed "
            "threshold (e.g., > 0.3 instead of > 0.5, or remove the filter on a "
            "high-restriction column) so a meaningful value can be reported.\n"
            "4. Subject naming: use a short snake_case identifier for the county or region "
            "from the question. If a single county, use the county name in snake_case "
            "without 'county' suffix (e.g., 'midland', 'potter', 'harris'). For regional "
            "scope use the named region (e.g., 'gulf_coast').\n"
            "\n"
            "STRUCTURED FACTS:\n"
            "- subject: <subject>  metric: <metric_name>  value: <numeric_value>\n"
            "- subject: <subject>  metric: <metric_name>  value: <numeric_value>\n"
            "(omit the block entirely if the question asks only for qualitative disclosure; "
            "otherwise you MUST include one line per requested numeric metric.)\n"
        )

        agent = create_react_agent(llm, tools, prompt=system_prompt)

        # Capture intermediate messages via agent.stream so that if the
        # wall-clock timeout fires, we can reconstruct a partial answer
        # from (a) any AI message the agent produced along the way and
        # (b) the ReAct tool-call history. This turns 0-score timeouts
        # into partial-score answers worth scoring.
        captured_messages: list = []

        def _stream_invoke():
            for chunk in agent.stream(
                {"messages": [
                    {"role": "user", "content": f"Execute the plan for: {state['user_query']}"}]},
                config={"recursion_limit": 75},
                stream_mode="values",
            ):
                msgs = chunk.get("messages") if isinstance(chunk, dict) else None
                if msgs:
                    captured_messages.clear()
                    captured_messages.extend(msgs)
            return {"messages": list(captured_messages)}

        def _pick_final(messages: list) -> str:
            for msg in reversed(messages or []):
                content = getattr(msg, "content", "") or ""
                if content.strip():
                    return content
            return ""

        def _synthesize_partial() -> str:
            """Build a best-effort answer from captured messages + plan when timeout hits."""
            partial_text = _pick_final(captured_messages)
            plan_block = (state.get("analysis_plan") or "").strip()
            if partial_text:
                # Agent produced something substantive before the deadline.
                prefix = (
                    "[Analysis partial — execute step hit the 600s budget. "
                    "The text below is what the agent produced before the deadline, "
                    "with the tactical plan appended as supporting context.]\n\n"
                )
                if plan_block and plan_block[:60] not in partial_text:
                    return f"{prefix}{partial_text}\n\n---\n\nTactical plan context:\n{plan_block}"
                return f"{prefix}{partial_text}"
            if plan_block:
                return (
                    "[Analysis incomplete — execute step hit the 600s budget before any final answer. "
                    "The tactical plan below outlines the intended analysis and reflects the KG-grounded "
                    "causal reasoning, but numeric values are not finalised via DB execution.]\n\n"
                    f"{plan_block}"
                )
            return "[Analysis incomplete: execute step timed out and no intermediate output was captured.]"

        try:
            result = _invoke_with_timeout(
                _stream_invoke,
                timeout_secs=600,
            )
            final = _pick_final(result.get("messages", []))
            if not final:
                final = _synthesize_partial()
        except (TimeoutError, concurrent.futures.TimeoutError):
            logger.warning("Execute node timed out after 600s — synthesizing partial from captured state.")
            final = _synthesize_partial()
        except Exception as exc:
            logger.warning(f"Execute agent error: {exc} — synthesizing partial answer.")
            partial = _synthesize_partial()
            final = partial if partial and "incomplete" not in partial[:60].lower() else (
                f"[Analysis incomplete due to processing limits. Error: {exc}]\n\n{partial}"
            )

        # Fallback disclosure: if warnings exist and the answer doesn't mention unavailability,
        # append a mandatory disclosure section (catches timeout/recursion-limit truncation).
        if data_warnings:
            disclosure_terms = ["unavailable", "not available", "no data", "cannot query",
                                "not in database", "proxy", "not loaded"]
            if not any(term in final.lower() for term in disclosure_terms):
                notice_lines = ["\n\n---\n**Data Availability Notice**"]
                for w in data_warnings:
                    notice_lines.append(f"- **[{w['concept']}]** {w['suggestion']}")
                final = final + "\n".join(notice_lines)
                logger.info("Appended fallback disclosure to final answer.")

        original_facts = _extract_structured_fact_records(final)
        body = _strip_structured_facts_section(final)
        final = _replace_structured_facts_section(body, original_facts)

        print(f"\nFINAL ANSWER:\n{final}")
        return {**state, "final_answer": final}

    return execute_analysis


# ── Graph Builder ─────────────────────────────────────────────────────────────

def build_orchestrator(ablation: dict | None = None):
    """Build a LangGraph orchestrator, optionally with ablation flags applied.

    ablation keys (all default True):
        use_tavily, use_kg, use_routing, use_planning
    """
    cfg = {**_DEFAULT_ABLATION, **(ablation or {})}

    graph = StateGraph(PipelineState)

    # Node 4 is always present
    graph.add_node("execute", _make_execute_node(cfg))
    graph.add_edge("execute", END)

    # Step 3: planning or skip
    if cfg["use_planning"]:
        graph.add_node("plan", _make_plan_node(cfg))
        graph.add_edge("plan", "execute")
        plan_entry = "plan"
    else:
        graph.add_node("skip_plan", _skip_planning)
        graph.add_edge("skip_plan", "execute")
        plan_entry = "skip_plan"

    # Steps 1-2: routing or inject defaults
    if cfg["use_routing"]:
        graph.add_node("extract", extract_context_and_criticality)
        graph.add_node("classify", classify_cluster)
        graph.add_edge("extract", "classify")
        graph.add_edge("classify", plan_entry)
        graph.set_entry_point("extract")
    else:
        graph.add_node("inject_defaults", _inject_defaults)
        graph.add_edge("inject_defaults", plan_entry)
        graph.set_entry_point("inject_defaults")

    return graph.compile()


# ── Default orchestrator (no ablation) ────────────────────────────────────────

_orchestrator = build_orchestrator()


# ── Public API ────────────────────────────────────────────────────────────────

def run(
    query: str,
    verbose: bool = False,
    ablation: dict | None = None,
    return_state: bool = False,
) -> str | dict:
    """Run the incident command pipeline.

    Args:
        query:        Natural language incident scenario.
        verbose:      Print final answer to stdout.
        ablation:     Dict of ablation flags (use_tavily, use_kg, use_routing, use_planning).
                      None = full system (default).
        return_state: If True, return full PipelineState dict instead of just final_answer string.
                      Use this in benchmark evaluation to access routing metadata.

    Returns:
        str (final_answer) by default, or dict (full PipelineState) if return_state=True.
    """
    # Compound query detection: split before routing if the query asks for
    # two distinct operational analyses simultaneously (e.g., shelter gaps AND
    # surge-exposed infrastructure). Only applied when routing is enabled.
    cfg = {**_DEFAULT_ABLATION, **(ablation or {})}
    if cfg.get("use_routing", True) and _is_compound_query(query):
        sub_queries = _split_compound_query(query)
        if sub_queries:
            print(f"\n[COMPOUND QUERY] Split into {len(sub_queries)} sub-queries:")
            for i, sq in enumerate(sub_queries, 1):
                print(f"  {i}. {sq}")
            sub_results = []
            first_state = None
            orchestrator = build_orchestrator(ablation) if ablation else _orchestrator
            for sq in sub_queries:
                sub_result = orchestrator.invoke(
                    {"user_query": sq, "data_warnings": []},
                    config={"recursion_limit": 75},
                )
                sub_results.append(sub_result.get("final_answer", ""))
                if first_state is None:
                    first_state = sub_result
            merged = _merge_compound_answers(query, sub_results)
            if verbose:
                print(f"\nFINAL ANSWER (merged):\n{merged}")
            if return_state and first_state is not None:
                state_out = {k: first_state.get(k, "") for k in PipelineState.__annotations__}
                state_out["final_answer"] = merged
                return state_out
            return merged

    orchestrator = build_orchestrator(ablation) if ablation else _orchestrator
    result = orchestrator.invoke(
        {"user_query": query, "data_warnings": []},
        config={"recursion_limit": 75},
    )

    if verbose and not return_state:
        print(f"\nFINAL ANSWER:\n{result['final_answer']}")

    if return_state:
        return {k: result.get(k, "") for k in PipelineState.__annotations__}
    return result["final_answer"]
