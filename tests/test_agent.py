import pytest
from fastapi.testclient import TestClient
from app.main import app
from app.agent.runner import (
    agent_runner,
    search_semantic_catalog,
    get_explores,
    get_dimensions,
    get_measures,
    get_field_values,
    query,
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

    # 2. Looker-style metadata tools list explores, dimensions and measures (visible fields only)
    assert "fct_transactions:" in get_explores.invoke({})
    dims = get_dimensions.invoke({"explore": "fct_transactions"})
    measures = get_measures.invoke({"explore": "fct_transactions"})
    assert "dim_categories.category_name" in dims
    assert "fct_transactions.total_spend_cop" not in dims
    assert "fct_transactions.total_spend_cop" in measures
    assert "fct_transactions.category_name " not in dims  # hidden stale ledger copy

    # 3. Field values come through the compiler
    values = json.loads(get_field_values.invoke({
        "explore": "fct_transactions",
        "field": "dim_categories.category_name",
        "search": "transport",
    }))
    assert "Public transport" in values["values"]
    assert values["truncated"] is False

    # 4. Semantic query: the compiler writes the SQL
    res = json.loads(query.invoke({
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
    out = query.invoke({
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

    semantic_tools = [
        "search_semantic_catalog", "get_explores", "get_dimensions", "get_measures", "get_field_values", "query",
    ]
    fallback_tools = ["get_table_schema", "run_sql_query"]

    m_oos, steps_oos, tools_oos = intent_router.get_route_config("out_of_scope")
    assert m_oos == settings.GEMINI_ROUTER_MODEL
    assert steps_oos == 0
    assert tools_oos == []

    m_cap, steps_cap, tools_cap = intent_router.get_route_config("capabilities")
    assert m_cap == settings.GEMINI_ROUTER_MODEL
    assert steps_cap == 2
    assert tools_cap == ["get_explores", "get_measures", "get_dimensions"]

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


def test_intent_router_splits_question_in_lookml_terms():
    """RouterDecision splits a question into measures, dimensions, filter values and a time period."""
    from app.schemas.query import RouterDecision
    decision = RouterDecision(
        intent="simple_query",
        reasoning="User wants restaurant spend last month",
        measures=["total spend"],
        filter_values=["restaurants"],
        time_period="last month",
    )
    assert decision.measures == ["total spend"]
    assert decision.dimensions == []
    assert decision.filter_values == ["restaurants"]
    assert decision.time_period == "last month"


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
                tool_calls=[{"name": "query", "args": self.query_args, "id": "call-1"}],
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
        "filters": [{"field": "fct_transactions.transaction_month", "value": "2026-10"}],
    })
    monkeypatch.setattr(agent_runner, "api_key", "test-key")
    monkeypatch.setattr(agent_runner, "llm_with_tools", fake)
    monkeypatch.setattr(agent_runner, "_get_or_create_llm", lambda model, tools: (fake, fake))
    monkeypatch.setattr(
        intent_router,
        "classify",
        lambda query, has_session_context=False: RouterDecision(
            intent="simple_query", reasoning="test", measures=["total spend"], time_period="2026-10"
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

    denied = AgentRunner._run_tool("run_sql_query", {"sql": "SELECT 1"}, ["query"], "simple_query", trace)
    assert "not permitted" in denied


def test_token_usage_is_tracked_per_request():
    """Every Gemini call's usage_metadata is summed into the request's token_usage."""
    from langchain_core.messages import AIMessage
    from langchain_core.outputs import ChatGeneration, LLMResult
    from app.agent.usage import UsageCallback, start_tracking, summarize

    calls = start_tracking()
    for model, (i, o) in [("router", (300, 40)), ("main", (2500, 120))]:
        msg = AIMessage(content="", usage_metadata={"input_tokens": i, "output_tokens": o, "total_tokens": i + o})
        UsageCallback(model).on_llm_end(LLMResult(generations=[[ChatGeneration(message=msg)]]))

    usage = summarize(calls)
    assert usage["llm_calls"] == 2
    assert usage["input_tokens"] == 2800
    assert usage["output_tokens"] == 160
    assert [c["model"] for c in usage["calls"]] == ["router", "main"]


def test_token_usage_records_each_calls_input():
    """Each call in token_usage carries the messages it received and the names of the tools offered."""
    import uuid
    from langchain_core.language_models.fake_chat_models import FakeMessagesListChatModel
    from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, ToolMessage
    from langchain_core.outputs import ChatGeneration, LLMResult
    from app.agent.usage import UsageCallback, start_tracking, summarize

    usage_metadata = {"input_tokens": 10, "output_tokens": 2, "total_tokens": 12}
    calls = start_tracking()
    model = FakeMessagesListChatModel(
        responses=[AIMessage(content="ok", usage_metadata=usage_metadata)], callbacks=[UsageCallback("fake")]
    )
    model.invoke([SystemMessage(content="You are an analyst."), HumanMessage(content="How much?")])

    # A later turn of the tool loop, with the tools offered the way the Gemini integration passes them
    callback, run_id = UsageCallback("main"), uuid.uuid4()
    turn = [
        HumanMessage(content="How much?"),
        AIMessage(content="", tool_calls=[{"name": "query", "args": {"explore": "e"}, "id": "c1"}]),
        ToolMessage(content='{"rows": []}', tool_call_id="c1", name="query"),
    ]
    callback.on_chat_model_start(
        {}, [turn], run_id=run_id,
        invocation_params={"tools": [{"function_declarations": [{"name": "query"}, {"name": "get_explores"}]}]},
    )
    callback.on_llm_end(
        LLMResult(generations=[[ChatGeneration(message=AIMessage(content="done", usage_metadata=usage_metadata))]]),
        run_id=run_id,
    )

    first, second = summarize(calls)["calls"]
    assert first["input"]["messages"] == [
        {"role": "system", "ref": "agent"},
        {"role": "user", "content": "How much?"},
    ]
    assert first["input"]["tools"] == []
    assert second["input"]["tools"] == ["query", "get_explores"]
    assistant, tool_msg = second["input"]["messages"][1:]
    assert assistant["tool_calls"] == [{"name": "query", "args": {"explore": "e"}}]
    assert tool_msg == {"role": "tool", "content": '{"rows": []}', "tool_call_id": "c1", "name": "query"}
    assert not callback._inputs  # nothing left behind once the call ended

    # The router answers in a JSON schema instead of calling tools; its name is recorded
    router, run_id = UsageCallback("router"), uuid.uuid4()
    router.on_chat_model_start(
        {}, [[HumanMessage(content="How much?")]], run_id=run_id,
        invocation_params={"response_json_schema": {"title": "RouterDecision", "type": "object"}},
    )
    router.on_llm_end(
        LLMResult(generations=[[ChatGeneration(message=AIMessage(content="{}", usage_metadata=usage_metadata))]]),
        run_id=run_id,
    )
    assert calls[-1]["input"]["output_schema"] == "RouterDecision"


def test_prompts_name_no_dataset():
    """The fixed prompt text names no dataset values or fields: what the data covers comes from
    agent_instructions.yaml (data_description), the only place for dataset-specific wording."""
    import json as _json
    from app.agent.prompts import AGENT_SYSTEM_PROMPT, DEFAULT_PERSONA, RAW_SQL_DISABLED, RAW_SQL_ENABLED
    from app.agent.router import ROUTER_SYSTEM_PROMPT, data_description
    from app.agent.runner import tool_declarations
    from app.schemas.query import RouterDecision

    fixed_text = "\n".join([
        ROUTER_SYSTEM_PROMPT,
        _json.dumps(RouterDecision.model_json_schema()),
        AGENT_SYSTEM_PROMPT, DEFAULT_PERSONA, RAW_SQL_ENABLED, RAW_SQL_DISABLED,
        _json.dumps(tool_declarations(list(TOOL_REGISTRY))),
    ]).lower()
    for word in ["groceries", "rent", "uber", "eurotrip", "bancolombia", "restaurant", "merchant", "category group",
                 "fct_transactions", "dim_categories", "tier 1", "tier 2", "fx", "budget", "cop", "finance", "lakehouse"]:
        assert f" {word}" not in fixed_text and f"'{word}" not in fixed_text and f'"{word}' not in fixed_text, word

    from app.semantic.service import semantic_layer
    assert "{data}" in ROUTER_SYSTEM_PROMPT
    assert data_description() == semantic_layer.instructions.data_description.strip()


def test_data_description_falls_back_to_explore_labels(monkeypatch):
    """Without a data_description in the instructions, the router describes the data with the explore labels,
    without their layer suffix such as "(Gold)"."""
    from app.agent.router import data_description
    from app.semantic.instructions import AgentInstructions
    from app.semantic.service import SemanticLayer, semantic_layer

    monkeypatch.setattr(SemanticLayer, "instructions", property(lambda self: AgentInstructions()))
    text = data_description()
    labels = [e.label for e in semantic_layer.model.explores.values() if e.label]
    assert text.endswith(".")
    for label in labels:
        assert label.split(" (")[0] in text
    assert "(Gold)" not in text and "(Silver)" not in text and "(PDT)" not in text


def test_tool_descriptions_are_sent_once():
    """Gemini gets each tool's description on the function only, not again on its parameters object,
    while parameter descriptions (the query filter's field and value) are kept."""
    from langchain_google_genai import ChatGoogleGenerativeAI
    from langchain_google_genai._function_utils import convert_to_genai_function_declarations
    from app.agent.runner import tool_declarations
    from app.agent.usage import _tool_names

    llm = ChatGoogleGenerativeAI(model="gemini-3.5-flash-lite", google_api_key="test-key")
    bound = llm.bind_tools(tool_declarations(list(TOOL_REGISTRY)))
    params = llm._get_invocation_params(**bound.kwargs)
    # The integration turns the bound tools into Gemini function declarations when it builds the request
    sent = convert_to_genai_function_declarations(params["tools"])
    declarations = [d for tool in sent for d in tool.function_declarations]

    assert [d.name for d in declarations] == list(TOOL_REGISTRY)
    assert _tool_names(params) == list(TOOL_REGISTRY)
    for d in declarations:
        assert d.description
        assert not (d.parameters and d.parameters.description), f"{d.name} repeats its description on its parameters"
    query_decl = next(d for d in declarations if d.name == "query")
    filter_item = query_decl.parameters.properties["filters"].items
    assert "view.field" in filter_item.properties["field"].description
    assert filter_item.properties["value"].description


def test_retrieval_trace_carries_similarity_scores():
    """The retrieval step's trace shows each item's similarity and how each explore's score was built,
    while the context sent to the model stays free of scores."""
    from types import SimpleNamespace
    from app.agent.runner import _format_retrieval
    from app.agent.tools.rag_tool import retrieve

    r = retrieve("How much did I spend on cabs last month?", measure_phrase="total spend", filter_values=["cabs"])
    out = r.scored()
    assert [e["name"] for e in out["explores"]] == r.explores
    for explore in out["explores"]:
        parts = explore["parts"]
        assert set(parts) == {"own", "best_measure", "best_dimension", "filter_bonus"}
        assert abs(sum(parts.values()) - explore["score"]) < 0.01
    assert out["explores"] == sorted(out["explores"], key=lambda e: e["score"], reverse=True)
    assert all(0 < m["score"] <= 1 for m in out["measures"])
    assert [m["name"] for m in out["measures"]] == [i.id for i in r.measures]
    for g in out["glossary"]:
        assert g["match"] in ("keyword", "vector")
        assert (g["score"] is None) == (g["match"] == "keyword")
    assert all(g["score"] is not None for g in out["golden_queries"])

    prompt = _format_retrieval(r, SimpleNamespace(time_period="last month"))
    assert "score" not in prompt.lower()


def test_token_usage_stores_each_system_prompt_once():
    """Calls that repeat a system prompt point to one stored copy; a different prompt gets its own key."""
    from app.agent.usage import summarize

    def call(system, output_schema=None, extra=()):
        call_input = {"messages": [{"role": "system", "content": system}, {"role": "user", "content": "Q"}, *extra], "tools": []}
        if output_schema:
            call_input["output_schema"] = output_schema
        return {"model": "m", "input_tokens": 1, "output_tokens": 1, "total_tokens": 2, "input": call_input}

    tool_turn = ({"role": "assistant", "content": "", "tool_calls": [{"name": "query", "args": {}}]},)
    usage = summarize([
        call("Route it.", output_schema="RouterDecision"),
        call("Answer it."),
        call("Answer it.", extra=tool_turn),
        call("Decline politely."),
    ])
    assert usage["system_prompts"] == {"router": "Route it.", "agent": "Answer it.", "agent_2": "Decline politely."}
    refs = [c["input"]["messages"][0] for c in usage["calls"]]
    assert refs == [{"role": "system", "ref": r} for r in ["router", "agent", "agent", "agent_2"]]
    assert usage["calls"][2]["input"]["messages"][2]["tool_calls"][0]["name"] == "query"  # the loop's growth stays visible
    assert usage["calls"][0]["input"]["output_schema"] == "RouterDecision"
    assert summarize([])["calls"] == [] and "system_prompts" not in summarize([])


def test_field_values_flag_a_cut_off_list():
    """A value list longer than the limit says so, so the model searches instead of concluding."""
    from app.semantic.service import semantic_layer

    res = semantic_layer.field_values("fct_transactions", "dim_categories.category_name", limit=30)
    assert res["truncated"] is True
    assert res["total"] > 30
    assert len(res["values"]) == 30
    assert "Restaurants & fast food" not in res["values"]  # past the cut-off alphabetically


def test_value_search_finds_the_field_holding_a_word():
    """Looker-style fuzzy search: a word is looked up in every searchable text field of the explore."""
    from app.semantic.service import semantic_layer

    matches = semantic_layer.search_values("fct_transactions", "restaurants")
    assert matches[0]["field"] == "dim_categories.category_name"  # the category list ranks before merchants
    assert "Restaurants & fast food" in matches[0]["values"]

    fields = [m["field"] for m in semantic_layer.search_values("fct_transactions", "uber")]
    assert "fct_transactions.merchant" in fields


def test_value_search_respects_lookml_suggestions_setting():
    """Fields with `suggestions: no`, primary keys and hidden fields are never searched."""
    from app.semantic.service import semantic_layer

    searchable = {f.id for f in semantic_layer.model.searchable_fields("fct_transactions")}
    assert "dim_categories.category_name" in searchable
    assert "fct_transactions.transaction_id" not in searchable  # primary key
    note = next(f for f in semantic_layer.model.explore_fields("fct_transactions") if f.name == "note")
    if not note.suggestable:  # wallet-lookml marks the free-text note with `suggestions: no`
        assert "fct_transactions.note" not in searchable


def test_retrieval_resolves_filter_values_and_picks_explores():
    """Two-level retrieval: explores first, fields within them, and filter values found in the data."""
    from app.agent.tools.rag_tool import retrieve

    r = retrieve("How much did the eurotrip cost?", measure_phrase="total spend", filter_values=["eurotrip"])
    assert 0 < len(r.explores) <= settings.RETRIEVAL_EXPLORES
    assert all(set(m.explores) & set(r.explores) for m in r.measures)
    assert r.dimensions == []  # no group-by asked, so no dimension search
    found = {v for m in r.filter_values[0]["matches"] for v in m["values"]}
    assert "Eurotrip26" in found


def test_glossary_maps_business_words_to_filters():
    """Each glossary entry with a field mapping resolves a synonym the data doesn't contain to its value."""
    from app.agent.tools.rag_tool import resolve_filter_value, retrieve_glossary
    from app.semantic.service import semantic_layer

    mapped = [e for e in semantic_layer.instructions.glossary if e.field and e.value and e.synonyms]
    if not mapped:
        pytest.skip("No glossary entry with a field mapping in the agent instructions")
    for entry in mapped:
        word = entry.synonyms[0]
        glossary = retrieve_glossary(f"How much did I spend on {word}?")
        assert any(g["term"] == entry.term for g in glossary)
        resolved = resolve_filter_value(word, glossary)
        assert resolved["source"] == "glossary"
        assert resolved["matches"]
        assert all(m["values"] == [entry.value] for m in resolved["matches"])


def test_instructions_are_indexed():
    """The glossary entries and golden queries in the agent instructions all reach the Chroma catalog."""
    from app.semantic.service import semantic_layer

    instructions = semantic_layer.instructions
    items = indexer.extract_from_instructions()
    assert len([i for i in items if i.item_type == "glossary"]) == len(instructions.glossary)
    assert len([i for i in items if i.item_type == "golden_query"]) == len(instructions.golden_queries)


def test_list_filters_compile_like_dict_filters():
    """The query tool takes filters as a field/value list; it must compile to the same SQL as the dict form."""
    from app.agent.runner import AgentRunner, _AnswerTrace, filters_to_dict
    from app.semantic.compiler import SemanticQuery
    from app.semantic.service import semantic_layer

    as_dict = {"dim_categories.category_name": "Groceries", "fct_transactions.transaction_month": "2026-07"}
    as_list = [{"field": f, "value": v} for f, v in as_dict.items()]
    base = {"explore": "fct_transactions", "fields": ["fct_transactions.total_spend_cop"]}
    assert filters_to_dict(as_list) == as_dict
    assert (
        semantic_layer.compile(SemanticQuery(**base, filters=filters_to_dict(as_list))).sql
        == semantic_layer.compile(SemanticQuery(**base, filters=as_dict)).sql
    )
    trace = _AnswerTrace()
    output = AgentRunner._run_tool("query", {**base, "filters": as_list}, ["query"], "simple_query", trace)
    assert not output.startswith("Error"), output
    assert trace.semantic_query["filters"] == as_dict
    bad = AgentRunner._run_tool("query", {**base, "filters": [{"value": "Groceries"}]}, ["query"], "simple_query", trace)
    assert bad.startswith("Error")


def test_golden_queries_in_instructions_compile():
    """Every golden query in the agent instructions compiles, and none repeats an evaluation question."""
    import json as _json
    from pathlib import Path
    from app.semantic.compiler import SemanticQuery
    from app.semantic.service import semantic_layer

    eval_questions = {
        q["question"].lower()
        for q in _json.loads((Path(__file__).parent / "golden" / "questions.json").read_text(encoding="utf-8"))["questions"]
    }
    for golden in semantic_layer.instructions.golden_queries:
        semantic_layer.compile(SemanticQuery(**golden.query))
        assert golden.question.lower() not in eval_questions


def test_default_filters_apply_unless_the_query_sets_the_field(monkeypatch):
    """Looker-style default filters from the agent instructions are added and reported in the notes."""
    from app.semantic.compiler import SemanticQuery
    from app.semantic.instructions import AgentInstructions, DefaultFilter
    from app.semantic.service import semantic_layer

    instructions = AgentInstructions(default_filters=[
        DefaultFilter(explore="fct_transactions", field="fct_transactions.record_state", value="reconciled"),
    ])
    monkeypatch.setattr(semantic_layer, "_instructions", instructions)
    q = SemanticQuery(explore="fct_transactions", fields=["fct_transactions.transaction_count"])
    res = semantic_layer.run(q)
    assert "Default filter applied: fct_transactions.record_state = reconciled" in res["notes"]
    assert "reconciled" in res["sql"]

    own = semantic_layer.run(q.model_copy(update={"filters": {"fct_transactions.record_state": "cleared"}}))
    assert not any(n.startswith("Default filter") for n in own["notes"])
