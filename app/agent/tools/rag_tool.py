import json
import logging
import re
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional
from app.core.config import settings
from app.rag.store import vector_store
from app.schemas.catalog import CatalogItem

logger = logging.getLogger(__name__)


def search_data_catalog(
    query: str,
    top_k: Optional[int] = None,
    min_similarity: Optional[float] = None,
) -> str:
    """Tool for AI Agent: Search LookML semantic catalog for metric definitions,
    calculation formulas, dimensions, and join structures filtered by cosine similarity.
    """
    try:
        items: List[CatalogItem] = vector_store.search(
            query,
            top_k=top_k if top_k is not None else settings.RAG_TOP_K,
            min_similarity=min_similarity if min_similarity is not None else settings.RAG_MIN_SIMILARITY,
        )
        if not items:
            threshold = min_similarity if min_similarity is not None else settings.RAG_MIN_SIMILARITY
            return f"No catalog items matched query: '{query}' above similarity threshold ({threshold})."

        formatted_docs = []
        for i, item in enumerate(items, 1):
            formula = f" | SQL: {item.sql_formula}" if item.sql_formula else ""
            table = f" | Table: {item.table_name}" if item.table_name else ""
            score_info = (
                f" | score: {item.similarity_score:.3f} (distance: {item.distance:.3f})"
                if item.similarity_score is not None and item.distance is not None
                else ""
            )
            formatted_docs.append(
                f"[{i}] {item.name} ({item.item_type} in {item.view}{table}{formula}{score_info})\n"
                f"    Description: {item.description}"
            )

        return "\n\n".join(formatted_docs)
    except Exception as e:
        logger.error("RAG tool failed: %s", e)
        return f"Error querying semantic catalog: {str(e)}"


def get_table_schema(table_name: str) -> str:
    """Retrieve detailed schema for a specific lakehouse table or LookML derived table (PDT),
    including row grain, primary key, row count stats, date coverage, and column types.
    """
    try:
        from app.agent.tools.duckdb_tool import duckdb_tool

        clean_name = table_name.strip().strip("`").strip("'").strip('"')
        tbl_id = f"table.{clean_name}"
        view_id = f"view.{clean_name}"

        tbl_item = vector_store.get_by_id(tbl_id) or vector_store.get_by_id(view_id)
        if tbl_item and tbl_item.description:
            return f"[Table Schema: {clean_name}] {tbl_item.description}"

        # Fallback to inspecting live DuckDB table if present
        available = duckdb_tool.get_available_tables()
        matched = next((t for t in available if t.lower() == clean_name.lower()), None)
        if matched:
            cols = duckdb_tool.get_table_columns(matched)
            return f"[Table Schema: {matched}] Available columns: {', '.join(cols)}"

        return (
            f"Table '{table_name}' was not found in the lakehouse catalog. "
            f"Available tables: {', '.join(available)}."
        )
    except Exception as e:
        logger.error("get_table_schema failed: %s", e)
        return f"Error retrieving schema for '{table_name}': {str(e)}"


SEMANTIC_ITEM_TYPES = ["explore", "metric", "dimension"]


@dataclass
class Retrieval:
    """What Chroma retrieved for one question: Looker-style, explores first, then fields within them."""

    explores: List[str] = field(default_factory=list)
    measures: List[CatalogItem] = field(default_factory=list)
    dimensions: List[CatalogItem] = field(default_factory=list)
    glossary: List[Dict[str, Any]] = field(default_factory=list)
    golden_queries: List[Dict[str, Any]] = field(default_factory=list)
    # One entry per filter value the router found: {"value", "source", "matches": [{field, values, explores}]}
    filter_values: List[Dict[str, Any]] = field(default_factory=list)
    # How each explore was ranked: {"score", "parts": {"own", "best_measure", "best_dimension", "filter_bonus"}}
    explore_scores: Dict[str, Dict[str, Any]] = field(default_factory=dict)
    # Glossary term -> {"match": "keyword" | "vector", "score"}; keyword matches have no similarity score
    glossary_scores: Dict[str, Dict[str, Any]] = field(default_factory=dict)

    def scored(self) -> Dict[str, Any]:
        """The retrieval result with similarity scores, for the answer's trace (not for the prompt)."""
        def item(i: CatalogItem) -> Dict[str, Any]:
            return {"name": i.id, "score": _round(i.similarity_score)}

        return {
            "explores": [{"name": e, **self.explore_scores.get(e, {})} for e in self.explores],
            "measures": [item(i) for i in self.measures],
            "dimensions": [item(i) for i in self.dimensions],
            "filter_values": self.filter_values,
            "glossary": [{"term": g.get("term"), **self.glossary_scores.get(g.get("term"), {})} for g in self.glossary],
            "golden_queries": [{"question": g["question"], "score": g.get("score")} for g in self.golden_queries],
        }


def _round(score: Optional[float]) -> Optional[float]:
    return round(score, 3) if score is not None else None


# Weak matches still help to vote for an explore, but only fields above this floor reach the prompt
_FIELD_POOL = 15
_INSTRUCTION_MIN_SIMILARITY = 0.35


def retrieve(
    question: str, measure_phrase: str = "", dimension_phrase: str = "", filter_values: Optional[List[str]] = None
) -> Retrieval:
    """Two-level retrieval for the agent's prompt.

    1. Measures and dimensions are searched separately (each with its own phrase), so neither type
       crowds the other out.
    2. Explores are ranked by their own similarity plus the similarity of the matching fields they
       contain, and the top ones are kept.
    3. Only fields that belong to a kept explore are returned, up to the per-type budget.
    4. Glossary entries (by term, synonym or meaning) and golden queries are retrieved for the question.
    5. Each filter value is resolved to the dimensions that contain it (glossary first, then a value
       search in the data), and explores that contain it get a boost.
    """
    glossary_scores: Dict[str, Dict[str, Any]] = {}
    glossary = retrieve_glossary(question, scores=glossary_scores)
    resolved = [resolve_filter_value(v, glossary) for v in (filter_values or []) if v.strip()]
    measures = vector_store.search(
        measure_phrase or question, top_k=_FIELD_POOL, min_similarity=settings.RAG_MIN_SIMILARITY, item_types=["metric"]
    )
    # Without a group-by in the question no dimension search runs: filter fields come from the value
    # resolution and time fields are listed per explore, so extra dimensions would only add tokens.
    dimensions = (
        vector_store.search(
            dimension_phrase, top_k=_FIELD_POOL, min_similarity=settings.RAG_MIN_SIMILARITY, item_types=["dimension"]
        )
        if dimension_phrase.strip()
        else []
    )
    explore_hits = vector_store.search(question, top_k=settings.RETRIEVAL_EXPLORES + 1, min_similarity=0.0, item_types=["explore"])

    # An explore scores its own similarity plus its best measure, its best dimension (half weight) and
    # half a point for every filter value found in it. Best match, not a sum, so an explore with many
    # look-alike measures doesn't win on volume.
    scores: Dict[str, float] = defaultdict(float)
    parts: Dict[str, Dict[str, float]] = defaultdict(
        lambda: {"own": 0.0, "best_measure": 0.0, "best_dimension": 0.0, "filter_bonus": 0.0}
    )
    for item in explore_hits:
        scores[item.name] += item.similarity_score or 0.0
        parts[item.name]["own"] = item.similarity_score or 0.0
    for weight, part, hits in ((1.0, "best_measure", measures), (0.5, "best_dimension", dimensions)):
        best: Dict[str, float] = {}
        for item in hits:
            for explore in item.explores:
                best[explore] = max(best.get(explore, 0.0), item.similarity_score or 0.0)
        for explore, score in best.items():
            scores[explore] += weight * score
            parts[explore][part] = weight * score
    for value in resolved:
        for explore in {e for m in value["matches"] for e in m["explores"]}:
            scores[explore] += 0.5
            parts[explore]["filter_bonus"] += 0.5
    explores = [e for e, _ in sorted(scores.items(), key=lambda kv: kv[1], reverse=True)][: settings.RETRIEVAL_EXPLORES]
    for value in resolved:
        value["matches"] = [m for m in value["matches"] if set(m["explores"]) & set(explores)][:3]

    def within(items: List[CatalogItem], budget: int) -> List[CatalogItem]:
        return [i for i in items if set(i.explores) & set(explores)][:budget]

    return Retrieval(
        explores=explores,
        measures=within(measures, settings.RETRIEVAL_MEASURES),
        dimensions=within(dimensions, settings.RETRIEVAL_DIMENSIONS),
        glossary=glossary,
        golden_queries=[
            {"question": item.name, "query": json.loads(item.sql_formula or "{}"), "score": _round(item.similarity_score)}
            for item in vector_store.search(
                question,
                top_k=settings.RETRIEVAL_GOLDEN_QUERIES,
                min_similarity=_INSTRUCTION_MIN_SIMILARITY,
                item_types=["golden_query"],
            )
        ],
        filter_values=resolved,
        explore_scores={
            e: {"score": _round(scores[e]), "parts": {k: _round(v) for k, v in parts[e].items()}} for e in explores
        },
        glossary_scores=glossary_scores,
    )


def resolve_filter_value(value: str, glossary: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Which dimensions hold a value the user mentioned: a glossary mapping, else a search of the data."""
    from app.semantic.service import semantic_layer

    lowered = value.lower()
    for entry in glossary:
        words = [entry.get("term", "")] + entry.get("synonyms", [])
        if entry.get("field") and entry.get("value") and any(w.lower() == lowered or w.lower() in lowered for w in words if w):
            fields = []
            for explore in semantic_layer.model.explores:
                for f in semantic_layer.model.searchable_fields(explore):
                    if f.name == entry["field"]:
                        fields.append((f.id, explore))
            explores_by_field: Dict[str, List[str]] = defaultdict(list)
            for field_id, explore in fields:
                explores_by_field[field_id].append(explore)
            return {
                "value": value,
                "source": "glossary",
                "matches": [{"field": f, "values": [entry["value"]], "explores": e} for f, e in explores_by_field.items()],
            }
    return {"value": value, "source": "data", "matches": semantic_layer.search_values_everywhere(value, per_field=3)}


def retrieve_glossary(text: str, scores: Optional[Dict[str, Dict[str, Any]]] = None) -> List[Dict[str, Any]]:
    """Hybrid glossary lookup: entries whose term or synonym appears in the text, then close vector matches.
    `scores`, if given, receives how each returned entry matched: {"match": "keyword" | "vector", "score"}.
    """
    from app.semantic.service import semantic_layer

    lowered = text.lower()
    found: Dict[str, Dict[str, Any]] = {}
    how: Dict[str, Dict[str, Any]] = {}
    for entry in semantic_layer.instructions.glossary:
        words = [entry.term] + entry.synonyms
        if any(re.search(rf"\b{re.escape(w.lower())}\b", lowered) for w in words):
            found[entry.term] = entry.model_dump()
            how[entry.term] = {"match": "keyword", "score": None}
    for item in vector_store.search(
        text, top_k=settings.RETRIEVAL_GLOSSARY, min_similarity=_INSTRUCTION_MIN_SIMILARITY, item_types=["glossary"]
    ):
        found.setdefault(item.name, json.loads(item.sql_formula or "{}"))
        how.setdefault(item.name, {"match": "vector", "score": _round(item.similarity_score)})
    kept = list(found.values())[: settings.RETRIEVAL_GLOSSARY]
    if scores is not None:
        scores.update({e["term"]: how[e["term"]] for e in kept if e.get("term") in how})
    return kept


def search_semantic_catalog(query: str, top_k: Optional[int] = None) -> str:
    """Tool for AI Agent: find the LookML explores, measures and dimensions that fit a question.

    Returns field names in the view.field form the semantic query compiler expects, with the
    explores each field can be queried in. Hidden fields are never indexed.
    """
    try:
        items: List[CatalogItem] = vector_store.search(
            query,
            top_k=top_k if top_k is not None else settings.RETRIEVAL_MEASURES + settings.RETRIEVAL_DIMENSIONS,
            min_similarity=settings.RAG_MIN_SIMILARITY,
            item_types=SEMANTIC_ITEM_TYPES,
        )
        if not items:
            return f"No explores or fields matched '{query}'. Try get_explores, then get_measures or get_dimensions on the most likely explore."

        lines = []
        for item in items:
            if item.item_type == "explore":
                lines.append(f"[explore] {item.name}: {item.description}")
                continue
            kind = "measure" if item.item_type == "metric" else "dimension"
            where = f" | explores: {', '.join(item.explores)}" if item.explores else ""
            lines.append(f"[{kind}] {item.view}.{item.name}{where}\n    {item.description}")
        return "\n".join(lines)
    except Exception as e:
        logger.error("Semantic catalog search failed: %s", e)
        return f"Error searching the semantic catalog: {str(e)}"
