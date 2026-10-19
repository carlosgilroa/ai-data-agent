from contextlib import asynccontextmanager
import logging
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from app.core.config import settings
from app.api.v1.router import api_router
from app.rag.indexer import indexer

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s"
)
logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan event handler for startup initialization and shutdown."""
    logger.info("Initializing AI Data Agent semantic catalog and Lakehouse connections...")
    try:
        indexer.run_indexing()
        logger.info("Semantic catalog auto-indexing completed successfully.")
    except Exception as e:
        logger.error("Error during startup indexing: %s", e)
    yield
    logger.info("Shutting down AI Data Agent service.")


app = FastAPI(
    title=settings.PROJECT_NAME,
    version=settings.VERSION,
    description="""
### AI-Native Semantic Query Agent & RAG Microservice
Connects LookML semantic metrics with local DuckDB Parquet data lakehouse,
enabling natural language questions with verified metric definitions,
two-tier financial accounting logic, and sub-second SQL execution.

* **Swagger Docs**: `/docs`
* **OpenAPI Schema**: `/openapi.json`
* **Health Check**: `/api/v1/health`
* **Semantic Catalog**: `/api/v1/catalog`
* **Agent Query Interface**: `/api/v1/agent/query`
    """,
    lifespan=lifespan
)

# CORS configuration
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include API v1 router
app.include_router(api_router, prefix=settings.API_V1_STR)


@app.get("/", tags=["Root"])
def root():
    return {
        "service": settings.PROJECT_NAME,
        "version": settings.VERSION,
        "docs_url": "/docs",
        "health_check": f"{settings.API_V1_STR}/health",
        "catalog_endpoint": f"{settings.API_V1_STR}/catalog",
        "query_endpoint": f"{settings.API_V1_STR}/agent/query"
    }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "app.main:app",
        host=settings.HOST,
        port=settings.PORT,
        reload=settings.DEBUG
    )
