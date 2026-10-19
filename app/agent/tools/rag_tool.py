import logging
from typing import Dict, Any, List
from app.rag.store import vector_store
from app.schemas.catalog import CatalogItem

logger = logging.getLogger(__name__)


def search_data_catalog(query: str, top_k: int = 5) -> str:
    """Tool for AI Agent: Search data dictionary and LookML catalog for definitions,

    metric formulas, table schemas, and business rules.
    """
    try:
        items: List[CatalogItem] = vector_store.search(query, top_k=top_k)
        if not items:
            return f"No catalog items matched query: '{query}'. Check available tables: master_records, fx_rates, mart_monthly_pnl."

        formatted_docs = []
        for i, item in enumerate(items, 1):
            formula = f" | SQL: {item.sql_formula}" if item.sql_formula else ""
            table = f" | Table: {item.table_name}" if item.table_name else ""
            formatted_docs.append(
                f"[{i}] {item.name} ({item.item_type} in {item.view}{table}{formula})\n"
                f"    Description: {item.description}"
            )

        return "\n\n".join(formatted_docs)
    except Exception as e:
        logger.error("RAG tool failed: %s", e)
        return f"Error querying semantic catalog: {str(e)}"
