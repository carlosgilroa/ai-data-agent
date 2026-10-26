"""Golden questions, live tier: send each question to the agent through Gemini and check its numbers.

These tests spend Gemini credits, so they never run by default. Run them on demand:

    pytest -m live                                   # all golden questions
    GOLDEN_IDS=spend_sep,aug_vs_sep pytest -m live   # only some of them

A question passes when the agent answered from governed LookML metrics and every number of the
reference result appears in the rows of the semantic queries it ran.
"""

import json
import os

import pytest

from app.agent.runner import agent_runner
from app.agent.tools.duckdb_tool import duckdb_tool
from app.core.config import settings
from app.schemas.query import QueryRequest
from app.semantic.service import semantic_layer
from tests.golden.helpers import load_golden, normalize, values_match

pytestmark = pytest.mark.live

GOLDEN = load_golden()
SELECTED = {i.strip() for i in os.getenv("GOLDEN_IDS", "").split(",") if i.strip()}
CASES = [c for c in GOLDEN["questions"] if not SELECTED or c["id"] in SELECTED]


@pytest.fixture(scope="module", autouse=True)
def live_setup():
    if not settings.GEMINI_API_KEY:
        pytest.skip("GEMINI_API_KEY is not set; live golden tests need Gemini.")
    original = settings.AGENT_TODAY
    settings.AGENT_TODAY = GOLDEN["today"]
    semantic_layer._model = None
    yield
    settings.AGENT_TODAY = original
    semantic_layer._model = None


def _numbers(rows):
    return [v for row in rows for v in (row.values() if isinstance(row, dict) else row) if isinstance(normalize(v), float)]


def _agent_numbers(response):
    """Numbers from every semantic query the agent ran, since a comparison can take several queries."""
    rows = list(response.data or [])
    for step in response.thought_steps:
        if step.tool == "query" and isinstance(step.tool_output, str):
            try:
                rows += json.loads(step.tool_output).get("rows", [])
            except (ValueError, AttributeError):
                continue
    return _numbers(rows)


@pytest.mark.parametrize("case", CASES, ids=lambda c: c["id"])
def test_agent_answers_golden_question(case):
    response = agent_runner.run_query(QueryRequest(query=case["question"]))
    expected = duckdb_tool.conn.sql(case["reference_sql"]).fetchall()
    got = _agent_numbers(response)

    assert response.governed is True, f"{case['id']}: answered without governed LookML metrics. Answer: {response.answer}"
    missing = [n for n in _numbers(expected) if not any(values_match(n, g) for g in got)]
    assert not missing, (
        f"{case['id']} ({case['question']}): expected numbers {missing} not in the agent's data.\n"
        f"Semantic query: {response.semantic_query}\nAnswer: {response.answer}"
    )
