from fastapi import APIRouter
from app.api.v1.endpoints import health, catalog, agent

api_router = APIRouter()

api_router.include_router(health.router)
api_router.include_router(catalog.router)
api_router.include_router(agent.router, prefix="/agent")
