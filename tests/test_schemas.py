import pytest
from app.schemas.catalog import CatalogItem, DimensionDefinition, MetricDefinition
from app.schemas.query import QueryRequest, QueryResponse, AgentThoughtStep


def test_catalog_item_schema():
    item = CatalogItem(
        id="fct_transactions.total_spend_usd",
        item_type="metric",
        name="total_spend_usd",
        view="fct_transactions",
        description="Total spend in USD excluding transfers",
        sql_formula="SUM(ABS(amount)) FILTER (WHERE is_transfer = false)"
    )
    assert item.name == "total_spend_usd"
    assert item.item_type == "metric"
    assert "FILTER" in item.sql_formula


def test_query_request_and_response():
    req = QueryRequest(query="What was my spending last month?")
    assert req.query == "What was my spending last month?"
    assert req.explain is True

    thought = AgentThoughtStep(
        step_number=1,
        tool="search_data_catalog",
        tool_input={"query": req.query},
        tool_output="Matched total_spend_usd",
        thought="Checking metrics"
    )

    resp = QueryResponse(
        query=req.query,
        answer="You spent $2,450.00.",
        sql="SELECT sum(amount) FROM master_records",
        data=[{"sum(amount)": -2450.0}],
        thought_steps=[thought],
        execution_time_ms=12.5
    )
    assert resp.answer == "You spent $2,450.00."
    assert len(resp.thought_steps) == 1
    assert resp.execution_time_ms == 12.5
