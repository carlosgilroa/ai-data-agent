import pytest
from fastapi.testclient import TestClient
from app.main import app
from app.agent.runner import (
    agent_runner,
    search_data_catalog,
    get_table_schema,
    get_dimension_values,
    run_sql_query,
    TOOL_REGISTRY,
)
from app.schemas.query import QueryRequest
from app.core.config import settings

from app.rag.indexer import indexer

client = TestClient(app)


@pytest.fixture(scope="module", autouse=True)
def setup_catalog():
    """Ensure semantic catalog is indexed with table grains and stats before running agent tests."""
    indexer.run_indexing()


def test_agent_tools_execution():
    """Verify all 4 first-class LangChain tools execute and return valid string outputs."""
    # 1. Test catalog search tool (decoupled metrics & dimensions)
    cat_res = search_data_catalog.invoke({"query": "spend"})
    assert isinstance(cat_res, str)
    assert len(cat_res) > 0

    # 2. Test schema retrieval tool
    schema_res = get_table_schema.invoke({"table_name": "mart_expense_summary"})
    assert isinstance(schema_res, str)
    assert "[Table Schema: mart_expense_summary]" in schema_res
    assert "Available columns" in schema_res

    # 3. Test dimension values tool
    dim_res = get_dimension_values.invoke({
        "table_name": "mart_expense_summary",
        "column_name": "category_name",
        "pattern": "transport",
    })
    assert isinstance(dim_res, str)
    assert "Public transport" in dim_res or "values" in dim_res

    # 4. Test SQL execution tool (pure Python DuckDB without pandas)
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

    all_tools = ["search_data_catalog", "get_table_schema", "get_dimension_values", "run_sql_query"]

    m_oos, steps_oos, tools_oos = intent_router.get_route_config("out_of_scope")
    assert m_oos == settings.GEMINI_ROUTER_MODEL
    assert steps_oos == 0
    assert tools_oos == []

    m_cap, steps_cap, tools_cap = intent_router.get_route_config("capabilities")
    assert m_cap == settings.GEMINI_ROUTER_MODEL
    assert steps_cap == 2
    assert tools_cap == ["search_data_catalog", "get_table_schema"]

    m_sim, steps_sim, tools_sim = intent_router.get_route_config("simple_query")
    assert m_sim == settings.GEMINI_SIMPLE_MODEL
    assert steps_sim == 6
    assert tools_sim == all_tools

    m_cmp, steps_cmp, tools_cmp = intent_router.get_route_config("complex_query")
    assert m_cmp == settings.GEMINI_MODEL
    assert steps_cmp == 8
    assert tools_cmp == all_tools


def test_search_data_catalog_is_decoupled_from_schemas():
    """Verify search_data_catalog returns clean semantic items without dumping table schema footers."""
    res = search_data_catalog.invoke({"query": "monthly expense pnl summary"})
    assert isinstance(res, str)
    # Schemas are now decoupled into get_table_schema
    assert "[Table Schema:" not in res


def test_get_table_schema_tool():
    """Verify get_table_schema retrieves schema with grain, row count stats, and date coverage."""
    # Test physical gold mart
    res = get_table_schema.invoke({"table_name": "mart_expense_summary"})
    assert "[Table Schema: mart_expense_summary]" in res
    assert "Grain:" in res
    assert "Stats:" in res
    assert "Date Coverage:" in res

    # Test derived table PDT
    pdt_res = get_table_schema.invoke({"table_name": "pdt_category_mom_velocity"})
    assert "[Table Schema: pdt_category_mom_velocity]" in pdt_res
    assert "Stats:" in pdt_res
    assert "Date Coverage:" in pdt_res


def test_get_dimension_values_tool():
    """Verify get_dimension_values returns valid JSON with count and values list."""
    res = get_dimension_values.invoke({
        "table_name": "mart_expense_summary",
        "column_name": "category_name",
        "pattern": "transport",
    })
    import json
    parsed = json.loads(res)
    assert parsed["table"] == "mart_expense_summary"
    assert parsed["column"] == "category_name"
    assert parsed["count"] > 0
    assert isinstance(parsed["values"], list)


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


def test_intent_router_semantic_decomposition_schema():
    """Verify RouterDecision schema includes metric_search_phrase and entity_keywords."""
    from app.schemas.query import RouterDecision
    decision = RouterDecision(
        intent="simple_query",
        reasoning="User wants aggregated spend for food last month",
        metric_search_phrase="total food spending expenses",
        entity_keywords=["food"],
    )
    assert decision.intent == "simple_query"
    assert decision.metric_search_phrase == "total food spending expenses"
    assert decision.entity_keywords == ["food"]


def test_session_memory_turn_validation_safety():
    """Verify session memory saves clean (HumanMessage, AIMessage) dialogue pairs,
    preventing orphaned ToolMessages from violating Gemini turn-validation rules.
    """
    from langchain_core.messages import HumanMessage, AIMessage, ToolMessage

    test_session = "test-session-safety-check"
    agent_runner.clear_session(test_session)

    # Simulate saving 2 turns
    agent_runner._save_session_turn(
        session_id=test_session,
        user_query="How much did I spend on utilities last month?",
        assistant_answer="You spent $603,334.16 on Energy & utilities.",
        last_sql="SELECT year_month, net_spent FROM mart_expense_summary",
    )
    agent_runner._save_session_turn(
        session_id=test_session,
        user_query="How much did I spend on groceries last month?",
        assistant_answer="You spent $1,549,008.66 on Groceries.",
        last_sql="SELECT year_month, net_spent FROM mart_expense_summary",
    )

    history = agent_runner.sessions[test_session]
    assert len(history) == 4
    # Ensure all messages are alternating HumanMessage and AIMessage, and no ToolMessage exists
    assert isinstance(history[0], HumanMessage)
    assert isinstance(history[1], AIMessage)
    assert isinstance(history[2], HumanMessage)
    assert isinstance(history[3], AIMessage)
    assert not any(isinstance(m, ToolMessage) for m in history)
    # Ensure SQL is retained in the assistant turn for contextual continuity
    assert "Executed Lakehouse Query" in history[1].content
    assert "mart_expense_summary" in history[1].content


def test_get_session_endpoint():
    """Verify GET /api/v1/agent/session/{session_id} retrieves active dialogue history."""
    test_session = "test-session-endpoint-inspection"
    agent_runner.clear_session(test_session)

    # Empty session inspection
    res = client.get(f"/api/v1/agent/session/{test_session}")
    assert res.status_code == 200
    data = res.json()
    assert data["session_id"] == test_session
    assert data["turn_count"] == 0
    assert data["message_count"] == 0
    assert data["dialogue"] == []

    # Populate a turn
    agent_runner._save_session_turn(
        session_id=test_session,
        user_query="Hello test",
        assistant_answer="Response test",
    )

    res = client.get(f"/api/v1/agent/session/{test_session}")
    assert res.status_code == 200
    data = res.json()
    assert data["session_id"] == test_session
    assert data["turn_count"] == 1
    assert data["message_count"] == 2
    assert len(data["dialogue"]) == 2
    assert data["dialogue"][0]["type"] == "HumanMessage"
    assert data["dialogue"][0]["content"] == "Hello test"
    assert data["dialogue"][1]["type"] == "AIMessage"
    assert data["dialogue"][1]["content"] == "Response test"

    agent_runner.clear_session(test_session)
