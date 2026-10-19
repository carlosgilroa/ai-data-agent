from fastapi import APIRouter
from app.core.config import settings
from app.schemas.query import HealthResponse
from app.agent.tools.duckdb_tool import duckdb_tool
from app.rag.store import vector_store

router = APIRouter()


@router.get("/health", response_model=HealthResponse, tags=["Health"])
def get_health() -> HealthResponse:
    """Check system health, DuckDB parquet lakehouse connectivity, and ChromaDB status."""
    tables = duckdb_tool.get_available_tables()
    return HealthResponse(
        status="healthy" if tables else "degraded",
        version=settings.VERSION,
        duckdb_connected=len(tables) > 0,
        tables_available=tables,
        vector_store_ready=vector_store.count() > 0 or vector_store.chroma_available,
        gemini_configured=bool(settings.GEMINI_API_KEY and settings.GEMINI_API_KEY != "your_gemini_api_key_here")
    )
