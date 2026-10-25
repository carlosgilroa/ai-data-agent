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
    """Tool for AI Agent: Search data dictionary and LookML catalog for definitions,
    metric formulas, table schemas, and business rules filtered by cosine similarity.
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
        matched_ids = {item.id for item in items}

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

        # Attach physical/derived table column schema ONLY for the single top-1 referenced table
        top_table = None
        for item in items:
            if item.table_name:
                top_table = item.table_name
                break
            elif item.view:
                top_table = item.view
                break

        if top_table:
            tbl_id = f"table.{top_table}"
            view_id = f"view.{top_table}"
            if tbl_id not in matched_ids and view_id not in matched_ids:
                tbl_item = vector_store.get_by_id(tbl_id) or vector_store.get_by_id(view_id)
                if tbl_item and tbl_item.description:
                    formatted_docs.append(f"[Table Schema: {top_table}] {tbl_item.description}")

        return "\n\n".join(formatted_docs)
    except Exception as e:
        logger.error("RAG tool failed: %s", e)
        return f"Error querying semantic catalog: {str(e)}"
