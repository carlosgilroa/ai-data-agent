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


def test_session_clear_endpoint():
    """Verify DELETE /api/v1/agent/session/{session_id} clears conversational memory."""
    from langchain_core.messages import HumanMessage, AIMessage

    test_session = "test-session-suite-1"
    # Seed session
    agent_runner.sessions[test_session] = [
        HumanMessage(content="What were my expenses in January 2026?"),
        AIMessage(content="Your total expenses were $2,500,000 COP.")
    ]
    assert test_session in agent_runner.sessions

    # Call delete endpoint
    response = client.delete(f"/api/v1/agent/session/{test_session}")
    assert response.status_code == 200
    data = response.json()
    assert data["session_id"] == test_session
    assert data["cleared"] is True
    assert test_session not in agent_runner.sessions

    # Call delete endpoint again on non-existent session
    response2 = client.delete(f"/api/v1/agent/session/{test_session}")
    assert response2.status_code == 200
    data2 = response2.json()
    assert data2["cleared"] is False


def test_intent_router_route_config():
    """Verify two-step IntentRouter maps each intent to the expected model, step budget, and tools."""
    from app.agent.router import intent_router

    m_oos, steps_oos, tools_oos = intent_router.get_route_config("out_of_scope")
    assert m_oos == settings.GEMINI_ROUTER_MODEL
    assert steps_oos == 0
    assert tools_oos == []

    m_cap, steps_cap, tools_cap = intent_router.get_route_config("capabilities")
    assert m_cap == settings.GEMINI_ROUTER_MODEL
    assert steps_cap == 1
    assert tools_cap == ["search_data_catalog"]

    m_sim, steps_sim, tools_sim = intent_router.get_route_config("simple_query")
    assert m_sim == settings.GEMINI_SIMPLE_MODEL
    assert steps_sim == 4
    assert tools_sim == ["search_data_catalog", "run_sql_query"]

    m_cmp, steps_cmp, tools_cmp = intent_router.get_route_config("complex_query")
    assert m_cmp == settings.GEMINI_MODEL
    assert steps_cmp == 6
    assert tools_cmp == ["search_data_catalog", "run_sql_query"]


def test_rag_tool_single_top_table_schema_attachment():
    """Verify search_data_catalog only attaches the schema for the single top-1 referenced table,

    preventing context overload from dumping multiple competing table schemas.
    """
    res = search_data_catalog.invoke({"query": "monthly expense pnl summary"})
    assert isinstance(res, str)
    # Count occurrences of schema footers
    schema_count = res.count("[Table Schema:")
    assert schema_count <= 1, f"Expected at most 1 table schema attached, found {schema_count}"


def test_rag_tool_no_match_clean_message():
    """Verify search_data_catalog provides a clean response without hardcoded table names when no items match."""
    from app.agent.tools.rag_tool import search_data_catalog as raw_search
    res = raw_search("completely_unrelated_query_xyz_987", min_similarity=0.99)
    assert "No catalog items matched" in res
    assert "master_records" not in res
    assert "mart_expense_summary" not in res
    assert "pdt_category_mom_velocity" not in res


def test_run_sql_query_docstring_is_generalized():
    """Verify run_sql_query tool docstring does not leak hardcoded table names into LLM tool bindings."""
    doc = run_sql_query.__doc__ or ""
    assert "master_records" not in doc
    assert "mart_expense_summary" not in doc
    assert "mart_monthly_pnl" not in doc
    assert "dim_accounts" not in doc



