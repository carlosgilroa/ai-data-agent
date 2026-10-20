import pytest
from fastapi.testclient import TestClient
from app.main import app
from app.agent.runner import agent_runner, search_data_catalog, run_sql_query
from app.schemas.query import QueryRequest
from app.core.config import settings

client = TestClient(app)


def test_agent_tools_execution():
    """Verify first-class LangChain tools execute and return valid string outputs."""
    # Test catalog search tool
    cat_res = search_data_catalog.invoke({"query": "spend"})
    assert isinstance(cat_res, str)
    assert len(cat_res) > 0

    # Test SQL execution tool (pure Python DuckDB without pandas)
    sql_res = run_sql_query.invoke({"sql": "SELECT 1 AS test_col, 'abc' AS label"})
    assert isinstance(sql_res, str)
    assert "test_col" in sql_res
    assert "abc" in sql_res


def test_agent_unconfigured_api_key_raises_error():
    """Verify that when GEMINI_API_KEY is empty, agent_runner raises an explicit RuntimeError

    instead of silently returning a heuristic response.
    """
    original_key = settings.GEMINI_API_KEY
    try:
        settings.GEMINI_API_KEY = ""
        agent_runner.api_key = ""
        agent_runner.llm_with_tools = None

        with pytest.raises(RuntimeError, match="GEMINI_API_KEY is not configured"):
            agent_runner.run_query(QueryRequest(query="What is my net cashflow by month?"))
    finally:
        settings.GEMINI_API_KEY = original_key
        agent_runner.api_key = original_key


def test_api_endpoint_returns_400_when_key_missing():
    """Verify that POST /api/v1/agent/query returns HTTP 400 with a descriptive error

    when GEMINI_API_KEY is not configured.
    """
    original_key = settings.GEMINI_API_KEY
    try:
        settings.GEMINI_API_KEY = ""
        agent_runner.api_key = ""
        agent_runner.llm_with_tools = None

        response = client.post("/api/v1/agent/query", json={"query": "What is my net cashflow by month?"})
        assert response.status_code == 400
        data = response.json()
        assert "GEMINI_API_KEY is not configured" in data.get("detail", "")
    finally:
        settings.GEMINI_API_KEY = original_key
        agent_runner.api_key = original_key
