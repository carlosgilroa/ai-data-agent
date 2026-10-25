import logging
from typing import List, Optional
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
