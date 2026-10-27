from typing import Any, Dict

from fastapi import APIRouter, HTTPException

from app.semantic.compiler import SemanticQuery, SemanticQueryError
from app.semantic.service import semantic_layer

router = APIRouter()


@router.post("/semantic/query", response_model=Dict[str, Any], tags=["Semantic Layer"])
def run_semantic_query(query: SemanticQuery) -> Dict[str, Any]:
    """Compile a LookML semantic query to SQL and run it, without calling Gemini.

    Useful to check a metric by hand: the response carries the generated SQL, the field
    definitions used and the rows.
    """
    semantic_layer.ensure_fresh()
    try:
        return semantic_layer.run(query)
    except SemanticQueryError as e:
        raise HTTPException(status_code=400, detail=str(e))


@router.get("/semantic/explores", response_model=Dict[str, Any], tags=["Semantic Layer"])
def list_explores() -> Dict[str, Any]:
    """List the LookML explores the agent can query."""
    semantic_layer.ensure_fresh()
    return {
        "explores": [
            {"name": e.name, "label": e.label, "description": e.description}
            for e in semantic_layer.model.explores.values()
        ]
    }


@router.get("/semantic/explores/{explore}", response_model=Dict[str, Any], tags=["Semantic Layer"])
def describe_explore(explore: str) -> Dict[str, Any]:
    """List the visible dimensions and measures of one explore."""
    semantic_layer.ensure_fresh()
    if explore not in semantic_layer.model.explores:
        raise HTTPException(status_code=404, detail=f"Unknown explore '{explore}'.")
    return {
        "explore": explore,
        "fields": [semantic_layer.field_info(f) for f in semantic_layer.model.explore_fields(explore)],
    }
