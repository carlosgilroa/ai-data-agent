import time
from fastapi import APIRouter, HTTPException
from app.schemas.query import QueryRequest, QueryResponse, DirectSQLRequest, DirectSQLResponse
from app.agent.runner import agent_runner
from app.agent.tools.duckdb_tool import duckdb_tool

router = APIRouter()


@router.post("/query", response_model=QueryResponse, tags=["AI Data Agent"])
def query_agent(request: QueryRequest) -> QueryResponse:
    """Submit a natural language financial or data engineering question to the AI Agent.

    The agent retrieves LookML semantics, formulates LookML-aligned SQL, executes against DuckDB,
    and returns an analytical answer with full traceability.
    """
    try:
        response = agent_runner.run_query(request)
        return response
    except (RuntimeError, ValueError) as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Agent execution failed: {str(e)}")


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
