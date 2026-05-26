"""
Graph Agent — retrieves relevant causal rules from the Expert Knowledge Graph (EKG).

Given a user question, the agent:
  1. Matches key concepts/keywords from the question against Concept nodes in Neo4j.
  2. Retrieves 1-hop causal edges via Cypher.
  3. Optionally finds multi-hop causal paths between source and target concepts.
  4. Returns the relevant subgraph (matched rules) as structured data
     and a natural language summary.

The EKG is stored in Neo4j (loaded via scripts/load_neo4j.py).
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional, Tuple

from src.graph.context_graph import ContextGraph, get_context_graph

logger = logging.getLogger(__name__)


class GraphAgent:
    """
    Retrieves relevant causal rules from the EKG given a natural language query.

    Uses Neo4j (via ContextGraph) for synonym-based concept matching and
    Cypher-based causal edge retrieval.

    Usage:
        agent = GraphAgent()
        rules, summary = agent.query("What is the flood risk for a property with low elevation?")
    """

    def __init__(self, graph: ContextGraph | None = None):
        self.graph = graph or get_context_graph()
        logger.info("GraphAgent initialized (Neo4j-backed)")

    # ── Concept matching ──────────────────────────────────────────────────

    def _extract_concepts(self, question: str) -> List[Dict[str, Any]]:
        """
        Match a question against Concept nodes using synonym lists stored
        in Neo4j.

        Returns a list of matched concept dicts (id, type, description,
        matched_term).
        """
        matched = self.graph.match_concepts(question)

        if not matched:
            logger.info("No specific concepts matched for graph query")

        logger.info(
            f"Concepts extracted from '{question}': "
            f"{[c['id'] for c in matched]}"
        )
        return matched

    # ── Edge retrieval ────────────────────────────────────────────────────

    def _get_related_edges(
        self, concept_ids: List[str]
    ) -> List[Dict[str, Any]]:
        """
        Retrieve all causal edges where any of the concept_ids appears
        as source or target (1-hop via Cypher).
        """
        return self.graph.retrieve_causal_rules(concept_ids)

    # ── Formatting ────────────────────────────────────────────────────────

    # Hub concepts tend to be connected to many edges most of which are not
    # relevant to any specific cascade question. We deprioritise edges that
    # only touch these as an endpoint (see _rank_and_prune_edges).
    _HUB_CONCEPTS = {"flood_occurrence", "flood_severity", "community_resilience"}

    # Place / event / organisation tokens that indicate *specific* (not
    # textbook) evidence. Edges carrying these get a ranking bonus.
    _SPECIFICITY_TOKENS = (
        "harvey", "uri", "ike", "imelda", "allison",
        "harris county", "galveston", "travis", "brazoria", "nueces",
        "austin", "houston", "montgomery", "fort bend",
        "colorado river", "buffalo bayou", "ercot", "nfip",
        "sfp", "fema", "eoc",
    )

    @staticmethod
    def _format_rule(edge: Dict[str, Any]) -> str:
        """Format a single causal edge as a natural-language sentence."""
        source = edge.get("source", "?").replace("_", " ")
        target = edge.get("target", "?").replace("_", " ")
        etype = edge.get("type", "")
        condition = (edge.get("condition") or "").strip()
        evidence = (edge.get("evidence") or "").strip()

        verb = {
            "INCREASES": "increases",
            "REDUCES": "reduces",
            "INDICATES": "indicates the level of",
            "TRIGGERS": "triggers",
            "PRECEDES": "precedes",
            "REQUIRES": "requires",
            "MITIGATES": "mitigates",
            "AMPLIFIES_IF": "amplifies (conditional)",
        }.get(etype, etype.lower().replace("_", " "))

        cond_phrase = f" (when {condition})" if condition else ""
        core = f"{source.capitalize()}{cond_phrase} {verb} {target}"

        if evidence:
            # Strip provenance prefixes like "TDIS: " that add noise
            ev = evidence.split(": ", 1)[-1] if ": " in evidence[:30] else evidence
            ev = ev.rstrip(".")[:240]
            return f"{core}. Evidence: {ev}."
        return f"{core}."

    def _rank_and_prune_edges(
        self,
        edges: List[Dict[str, Any]],
        concept_ids: List[str],
        question: str,
        max_edges: int = 8,
    ) -> List[Dict[str, Any]]:
        """
        Deduplicate, score, and cap retrieved edges so only the most relevant
        rules are shown to the downstream agent.

        Ranking tiers (higher = earlier in output):
          +100  both endpoints are matched concepts (cascade within scope)
          +50   at least one matched endpoint that is not a hub concept
          +20   evidence mentions a specific place / event / organisation
          +10×  term overlap between endpoints and question text
          +     evidence length bonus (up to ~10)
        """
        if not edges:
            return edges

        # Dedup by (source, target, type) — keep the longer-evidenced variant
        by_key: Dict[tuple, Dict[str, Any]] = {}
        for e in edges:
            key = (e.get("source"), e.get("target"), e.get("type"))
            existing = by_key.get(key)
            if existing is None or len(e.get("evidence") or "") > len(existing.get("evidence") or ""):
                by_key[key] = e
        unique = list(by_key.values())

        concept_set = set(concept_ids)
        q_lower = question.lower()

        def score(e: Dict[str, Any]) -> float:
            src = e.get("source", "")
            tgt = e.get("target", "")
            src_m = src in concept_set
            tgt_m = tgt in concept_set

            s = 0.0
            if src_m and tgt_m:
                s += 100  # cascade edge fully inside question's concept scope
            nonhub = (src_m and src not in self._HUB_CONCEPTS) or (
                tgt_m and tgt not in self._HUB_CONCEPTS
            )
            if nonhub:
                s += 50

            # Term overlap with the question text itself
            for token in (src.replace("_", " "), tgt.replace("_", " ")):
                if token and token in q_lower:
                    s += 10

            # Specificity (place/event/organisation in evidence)
            ev = (e.get("evidence") or "").lower()
            if any(tok in ev for tok in self._SPECIFICITY_TOKENS):
                s += 20

            # Mild bonus for longer evidence (proxy for actual quote vs label)
            s += min(len(ev), 500) / 50
            return s

        unique.sort(key=score, reverse=True)
        return unique[:max_edges]

    def _build_summary(
        self,
        concept_ids: List[str],
        edges: List[Dict[str, Any]],
    ) -> str:
        """Render the (already ranked) edges as a compact numbered list."""
        if not edges:
            return "No relevant domain knowledge rules found for this query."

        lines = [
            f"Expert Knowledge Graph — {len(edges)} most relevant causal rules "
            f"(ranked by question-fit):",
        ]
        for i, edge in enumerate(edges, 1):
            lines.append(f"  {i}. {self._format_rule(edge)}")
        return "\n".join(lines)

    def _get_concept_descriptions(
        self, concept_ids: List[str]
    ) -> Dict[str, str]:
        """Fetch concept descriptions from Neo4j."""
        if not concept_ids:
            return {}
        query = """
        MATCH (c:Concept)
        WHERE c.id IN $ids
        RETURN c.id AS id, c.description AS description
        """
        with self.graph.driver.session() as session:
            result = session.run(query, ids=concept_ids)
            return {
                record["id"]: record["description"] or record["id"]
                for record in result
            }

    # ── Main query interface ──────────────────────────────────────────────

    def query(
        self, question: str, max_edges: int = 5
    ) -> Tuple[List[Dict[str, Any]], str]:
        """
        Main interface: given a natural language question, return matched
        causal rules and a summary.

        Edges are ranked by question-fit and capped at ``max_edges`` to
        avoid flooding strong LLMs with redundant textbook rules. See
        ``_rank_and_prune_edges`` for the ranking logic.

        Returns:
            (matched_edges, summary_text)
        """
        matched_concepts = self._extract_concepts(question)
        concept_ids = [c["id"] for c in matched_concepts]
        edges = self._get_related_edges(concept_ids)
        edges = self._rank_and_prune_edges(
            edges, concept_ids, question, max_edges=max_edges
        )
        summary = self._build_summary(concept_ids, edges)

        logger.info(
            f"Graph query: matched {len(concept_ids)} concepts, "
            f"surfaced {len(edges)} ranked rules (cap {max_edges})"
        )
        return edges, summary

    def query_causal_paths(
        self,
        question: str,
        target_concepts: List[str] | None = None,
        max_hops: int = 3,
    ) -> Tuple[List[Dict[str, Any]], str]:
        """
        Find multi-hop causal paths from question-matched concepts to
        target concepts (e.g., flood_occurrence, flood_severity).

        This is the key advantage of Neo4j over file-based JSON: real
        graph traversal instead of 1-hop edge matching.

        Parameters
        ----------
        question        : Natural language question.
        target_concepts : Target concept IDs (defaults to outcome concepts).
        max_hops        : Maximum path length.

        Returns
        -------
        (paths, summary_text)
        """
        if target_concepts is None:
            target_concepts = ["flood_occurrence", "flood_severity"]

        matched_concepts = self._extract_concepts(question)
        source_ids = [c["id"] for c in matched_concepts]

        # Exclude targets from sources to avoid trivial self-paths
        source_ids = [s for s in source_ids if s not in target_concepts]

        if not source_ids:
            return [], "No source concepts matched for path finding."

        paths = self.graph.retrieve_causal_paths(
            source_ids, target_concepts, max_hops=max_hops
        )

        if not paths:
            summary = (
                f"No causal paths found from {source_ids} to "
                f"{target_concepts} within {max_hops} hops."
            )
            return [], summary

        # Build summary
        lines = [
            f"Found {len(paths)} causal path(s) from matched concepts to "
            f"{target_concepts}:",
            "",
        ]
        for i, path in enumerate(paths, 1):
            node_chain = " → ".join(path.get("node_ids", []))
            hops = path.get("hops", "?")
            lines.append(f"  Path {i} ({hops} hops): {node_chain}")
            for edge in path.get("edges", []):
                etype = edge.get("type", "?")
                cond = edge.get("condition", "")
                cond_str = f" ({cond})" if cond else ""
                lines.append(f"    {etype}{cond_str}")
            lines.append("")

        summary = "\n".join(lines)
        return paths, summary

    def query_by_features(
        self, feature_values: Dict[str, Any]
    ) -> Tuple[List[Dict[str, Any]], str]:
        """
        Given a dict of feature_name → value, find all applicable EKG rules.
        Used when features are already known (e.g., from SQL Agent results).

        Returns:
            (matched_edges, summary_text)
        """
        # Only keep feature names that exist as Concept nodes
        query = """
        MATCH (c:Concept)
        WHERE c.id IN $ids
        RETURN c.id AS id
        """
        with self.graph.driver.session() as session:
            result = session.run(query, ids=list(feature_values.keys()))
            valid_ids = [record["id"] for record in result]

        edges = self._get_related_edges(valid_ids)
        summary = self._build_summary(valid_ids, edges)
        return edges, summary

    def get_schema_for_concepts(
        self, question: str
    ) -> List[Dict[str, Any]]:
        """
        Bridge method: given a question, find relevant concepts and then
        traverse MAPS_TO to find the relevant DataTables with their schemas.

        This is useful for concept-aware schema retrieval in the SQL Agent.

        Returns list of table dicts with columns and join rules.
        """
        matched_concepts = self._extract_concepts(question)
        concept_ids = [c["id"] for c in matched_concepts]
        return self.graph.retrieve_schema_for_concepts(concept_ids)


# ── Module-level convenience functions ───────────────────────────────────────

_agent: Optional[GraphAgent] = None


def get_graph_agent() -> GraphAgent:
    """Get or create the singleton GraphAgent."""
    global _agent
    if _agent is None:
        _agent = GraphAgent()
    return _agent


def query_ekg(question: str) -> Tuple[List[Dict[str, Any]], str]:
    """Convenience function: query the EKG with a natural language question."""
    return get_graph_agent().query(question)
