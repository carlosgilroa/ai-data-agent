import time
import logging
from fastapi import APIRouter, HTTPException
from app.schemas.query import QueryRequest, QueryResponse, DirectSQLRequest, DirectSQLResponse
from app.agent.runner import agent_runner
from app.agent.tools.duckdb_tool import duckdb_tool

logger = logging.getLogger(__name__)

router = APIRouter()


@router.post("/query", response_model=QueryResponse, tags=["AI Data Agent"])
def query_agent(request: QueryRequest) -> QueryResponse:
    """Submit a natural language financial or data engineering question to the AI Agent.

    The agent retrieves LookML semantics, formulates LookML-aligned SQL, executes against DuckDB,
    and returns an analytical answer with full traceability.
    """
    session_label = request.session_id or "default"
    logger.info(">>> [Session: %s] USER QUERY: %s", session_label, request.query)
    try:
        response = agent_runner.run_query(request)
        logger.info(
            "<<< [Session: %s] INTENT: %s | MODEL: %s | EXEC_TIME: %.2fms | SQL: %s",
            session_label,
            response.intent,
            response.model_used,
            response.execution_time_ms,
            response.sql,
        )
        logger.info("<<< [Session: %s] AGENT ANSWER: %s", session_label, response.answer)
        return response
    except (RuntimeError, ValueError) as e:
        logger.warning("Query error [Session: %s]: %s", session_label, e)
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.error("Query unexpected error [Session: %s]: %s", session_label, e, exc_info=True)
        raise HTTPException(status_code=500, detail=f"Agent execution failed: {str(e)}")


@router.get("/session/{session_id}", tags=["AI Data Agent"])
def get_session(session_id: str):
    """Retrieve active conversational dialogue turns for a specific session ID."""
    history = agent_runner.sessions.get(session_id, [])
    dialogue = []
    for msg in history:
        dialogue.append({
            "type": msg.__class__.__name__,
            "content": msg.content
        })
    return {
        "session_id": session_id,
        "turn_count": len(history) // 2,
        "message_count": len(history),
        "dialogue": dialogue
    }


@router.delete("/session/{session_id}", tags=["AI Data Agent"])
def reset_session(session_id: str):
    """Reset the multi-turn conversational memory for a specific session ID."""
    cleared = agent_runner.clear_session(session_id)
    return {
        "session_id": session_id,
        "cleared": cleared,
        "message": f"Session '{session_id}' memory cleared successfully." if cleared else f"Session '{session_id}' was not active."
    }


@router.post("/sql", response_model=DirectSQLResponse, tags=["Data Lakehouse"])
def execute_sql(request: DirectSQLRequest) -> DirectSQLResponse:
    """Directly execute a read-only DuckDB SQL query against the local Parquet lakehouse.

    Permitted tables: master_records, fx_rates, mart_monthly_pnl, mart_expense_summary.
    """
    start = time.time()
    try:
        result = duckdb_tool.run_query(request.sql)
        duration_ms = round((time.time() - start) * 1000, 2)
        return DirectSQLResponse(
            sql=request.sql,
            row_count=result.get("row_count", 0),
            columns=result.get("columns", []),
            data=result.get("data", []),
            execution_time_ms=duration_ms
        )
    except ValueError as ve:
        raise HTTPException(status_code=400, detail=str(ve))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Database execution error: {str(e)}")
