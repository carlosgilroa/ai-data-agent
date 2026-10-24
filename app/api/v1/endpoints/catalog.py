from typing import Dict, Any, Optional
from fastapi import APIRouter, Query
from app.schemas.catalog import CatalogListResponse, CatalogSearchResponse
from app.rag.store import vector_store
from app.rag.indexer import indexer

router = APIRouter()


@router.get("/catalog", response_model=CatalogListResponse, tags=["Semantic Catalog"])
def list_catalog() -> CatalogListResponse:
    """List all indexed LookML measures, dimensions, and data lake tables."""
    items = vector_store.list_all()
    if not items:
        # Auto-index if store is currently empty
        res = indexer.run_indexing()
        items = res.get("items", [])
    return CatalogListResponse(total=len(items), items=items)


@router.post("/catalog/index", response_model=Dict[str, Any], tags=["Semantic Catalog"])
def reindex_catalog() -> Dict[str, Any]:
    """Wipe ChromaDB and trigger a clean re-indexing of LookML views, models, and DuckDB parquet lakehouse."""
    result = indexer.run_indexing()
    return {
        "status": "success",
        "total_indexed": result.get("total_indexed", 0),
        "metrics_count": result.get("metrics", 0),
        "dimensions_count": result.get("dimensions", 0),
        "explores_count": result.get("explores", 0),
        "joins_count": result.get("joins", 0),
        "tables_count": result.get("tables", 0),
    }


@router.get("/catalog/search", response_model=CatalogSearchResponse, tags=["Semantic Catalog"])
def search_catalog(
    q: str = Query(..., description="Natural language search term for metrics or dimensions"),
    top_k: int = Query(5, ge=1, le=20, description="Maximum number of semantic results"),
    min_similarity: Optional[float] = Query(None, ge=0.0, le=1.0, description="Minimum cosine similarity score threshold"),
) -> CatalogSearchResponse:
    """Semantically search the LookML business metrics and database columns with similarity filtering."""
    matches = vector_store.search(query=q, top_k=top_k, min_similarity=min_similarity)
    return CatalogSearchResponse(
        query=q,
        total_matches=len(matches),
        results=matches
    )
