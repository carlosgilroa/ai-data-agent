import time
import logging
from pathlib import Path
from fastapi import APIRouter, HTTPException
from app.schemas.query import QueryRequest, QueryResponse, DirectSQLRequest, DirectSQLResponse
from app.agent.runner import agent_runner
from app.agent.tools.duckdb_tool import duckdb_tool
from app.core.config import settings

logger = logging.getLogger(__name__)


def _log_full_trace(response: QueryResponse) -> None:
    """Append the whole response (steps, tools, tokens) as one JSON line, for reviewing test sessions."""
    if not settings.QUERY_LOG_PATH:
        return
    try:
        path = Path(settings.QUERY_LOG_PATH)
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("a", encoding="utf-8") as f:
            f.write(response.model_dump_json() + "\n")
    except OSError as e:
        logger.warning("Could not write query log: %s", e)

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
        usage = response.token_usage or {}
        logger.info(
            "<<< [Session: %s] TOOLS: %s | LLM_CALLS: %s | TOKENS in/out/total: %s/%s/%s",
            session_label,
            [s.tool for s in response.thought_steps],
            usage.get("llm_calls"),
            usage.get("input_tokens"),
            usage.get("output_tokens"),
            usage.get("total_tokens"),
        )
        logger.info("<<< [Session: %s] AGENT ANSWER: %s", session_label, response.answer)
        _log_full_trace(response)
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
