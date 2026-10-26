import pytest
from fastapi.testclient import TestClient
from app.main import app
from app.agent.runner import (
    agent_runner,
    search_semantic_catalog,
    describe_explore,
    get_field_values,
    run_semantic_query,
    get_table_schema,
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
    """Verify the semantic tools and the raw SQL fallback execute and return string outputs."""
    import json

    # 1. Catalog search returns view.field names with the explores they live in
    cat_res = search_semantic_catalog.invoke({"query": "total spend net of refunds"})
    assert "total_spend_cop" in cat_res
    assert "explores:" in cat_res

    # 2. Explore description lists visible fields only
    desc = describe_explore.invoke({"explore": "fct_transactions"})
    assert "fct_transactions.total_spend_cop" in desc
    assert "dim_categories.category_name" in desc
    assert "fct_transactions.category_name " not in desc  # hidden stale ledger copy

    # 3. Field values come through the compiler
    values = json.loads(get_field_values.invoke({
        "explore": "fct_transactions",
        "field": "dim_categories.category_name",
        "pattern": "transport",
    }))
    assert "Public transport" in values["values"]

    # 4. Semantic query: the compiler writes the SQL
    res = json.loads(run_semantic_query.invoke({
        "explore": "mart_monthly_pnl",
        "fields": ["mart_monthly_pnl.year_month", "mart_monthly_pnl.total_spend_cop"],
        "limit": 3,
    }))
    assert res["row_count"] == 3
    assert "mart_monthly_pnl.total_spend_cop" in res["rows"][0]

    # 5. Raw SQL fallback still works, read-only
    sql_res = run_sql_query.invoke({"sql": "SELECT 1 AS test_col, 'abc' AS label"})
    assert "test_col" in sql_res
    assert "abc" in sql_res


def test_semantic_tool_errors_are_returned_to_the_model():
    """Unknown or hidden fields come back as a readable error with suggestions, not an exception."""
    out = run_semantic_query.invoke({
        "explore": "fct_transactions",
        "fields": ["fct_transactions.total_spend_usd"],
    })
    assert out.startswith("Error:")
    assert "fct_transactions.total_spend_cop" in out


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
    """Verify IntentRouter maps each intent to the expected model, step budget, and tools."""
    from app.agent.router import intent_router

    semantic_tools = ["search_semantic_catalog", "describe_explore", "get_field_values", "run_semantic_query"]
    fallback_tools = ["get_table_schema", "run_sql_query"]

    m_oos, steps_oos, tools_oos = intent_router.get_route_config("out_of_scope")
    assert m_oos == settings.GEMINI_ROUTER_MODEL
    assert steps_oos == 0
    assert tools_oos == []

    m_cap, steps_cap, tools_cap = intent_router.get_route_config("capabilities")
    assert m_cap == settings.GEMINI_ROUTER_MODEL
    assert steps_cap == 2
    assert tools_cap == ["search_semantic_catalog", "describe_explore"]

    m_sim, steps_sim, tools_sim = intent_router.get_route_config("simple_query")
    assert m_sim == settings.GEMINI_SIMPLE_MODEL
    assert steps_sim == 5
    assert tools_sim == semantic_tools + fallback_tools

    m_cmp, steps_cmp, tools_cmp = intent_router.get_route_config("complex_query")
    assert m_cmp == settings.GEMINI_MODEL
    assert steps_cmp == 8
    assert tools_cmp == semantic_tools + fallback_tools

    original = settings.ALLOW_RAW_SQL_FALLBACK
    try:
        settings.ALLOW_RAW_SQL_FALLBACK = False
        _, _, tools_locked = intent_router.get_route_config("complex_query")
        assert tools_locked == semantic_tools
    finally:
        settings.ALLOW_RAW_SQL_FALLBACK = original


def test_search_semantic_catalog_skips_tables_and_hidden_fields():
    """Verify the semantic search returns explores and fields, never raw tables or hidden fields."""
    res = search_semantic_catalog.invoke({"query": "category name of a transaction"})
    assert "[Table Schema:" not in res
    assert "fct_transactions.category_name\n" not in res


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


def test_session_memory_keeps_semantic_query_for_follow_ups():
    """The assistant turn carries the semantic query so a follow-up can reuse and adjust it."""
    test_session = "test-session-semantic-follow-up"
    agent_runner.clear_session(test_session)
    agent_runner._save_session_turn(
        session_id=test_session,
        user_query="How much did I spend last month?",
        assistant_answer="You spent 12,417,134 COP in September.",
        last_sql="SELECT ...",
        last_semantic_query={"explore": "mart_monthly_pnl", "fields": ["mart_monthly_pnl.total_spend_cop"]},
    )
    content = agent_runner.sessions[test_session][1].content
    assert "Semantic query used:" in content
    assert "mart_monthly_pnl.total_spend_cop" in content
    agent_runner.clear_session(test_session)


class _FakeLLM:
    """Stands in for Gemini: first asks for a semantic query, then answers from its result."""

    def __init__(self, query_args):
        self.query_args = query_args
        self.calls = 0

    def bind_tools(self, tools, **kwargs):
        return self

    def invoke(self, messages):
        from langchain_core.messages import AIMessage, ToolMessage

        self.calls += 1
        if self.calls == 1:
            return AIMessage(
                content="",
                tool_calls=[{"name": "run_semantic_query", "args": self.query_args, "id": "call-1"}],
            )
        tool_result = next(m for m in reversed(messages) if isinstance(m, ToolMessage))
        return AIMessage(content=f"Answer based on: {tool_result.content[:80]}")


def test_agent_loop_with_fake_llm_is_traceable(monkeypatch):
    """Runs the whole agent loop without Gemini: the response carries the semantic query,
    the fields used, the compiled SQL and the partial-period note."""
    from app.agent.router import intent_router
    from app.schemas.query import RouterDecision

    fake = _FakeLLM({
        "explore": "fct_transactions",
        "fields": ["fct_transactions.total_spend_cop"],
        "filters": {"fct_transactions.transaction_month": "2026-10"},
    })
    monkeypatch.setattr(agent_runner, "api_key", "test-key")
    monkeypatch.setattr(agent_runner, "llm_with_tools", fake)
    monkeypatch.setattr(agent_runner, "_get_or_create_llm", lambda model, tools: (fake, fake))
    monkeypatch.setattr(
        intent_router,
        "classify",
        lambda query, has_session_context=False: RouterDecision(
            intent="simple_query", reasoning="test", metric_search_phrase="total spend"
        ),
    )

    res = agent_runner.run_query(QueryRequest(query="How much did I spend in October 2026?"))

    assert res.governed is True
    assert res.semantic_query["explore"] == "fct_transactions"
    assert res.fields_used[0]["name"] == "fct_transactions.total_spend_cop"
    assert "net of refunds" in res.fields_used[0]["description"]
    assert "tier2_refund_offset" in res.sql
    assert any("partly covered" in n for n in res.notes)
    assert res.data and "fct_transactions.total_spend_cop" in res.data[0]


def test_raw_sql_fallback_is_marked_ungoverned():
    """An answer from the raw SQL fallback is flagged as not governed."""
    from app.agent.runner import AgentRunner, _AnswerTrace

    trace = _AnswerTrace()
    out = AgentRunner._run_tool(
        "run_sql_query", {"sql": "SELECT COUNT(*) AS n FROM master_records"},
        ["run_sql_query"], "complex_query", trace,
    )
    assert '"row_count": 1' in out
    assert trace.governed is False
    assert trace.notes == [_AnswerTrace.RAW_SQL_NOTE]

    denied = AgentRunner._run_tool("run_sql_query", {"sql": "SELECT 1"}, ["run_semantic_query"], "simple_query", trace)
    assert "not permitted" in denied
