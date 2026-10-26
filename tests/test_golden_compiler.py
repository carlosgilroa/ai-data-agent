"""Golden questions, free tier: the LookML compiler must return the same numbers as hand-written SQL.

No LLM is called. Each golden question carries the semantic query a correct answer uses and an
independent reference query over the raw tables; both run on the same DuckDB data.
"""

import pytest

from app.agent.tools.duckdb_tool import duckdb_tool
from app.core.config import settings
from app.semantic.compiler import SemanticQuery
from app.semantic.service import semantic_layer
from tests.golden.helpers import load_golden, rows_match

GOLDEN = load_golden()


@pytest.fixture(scope="module", autouse=True)
def fixed_today():
    """Pin "today" so relative filters like "last month" resolve the same way the references assume."""
    original = settings.AGENT_TODAY
    settings.AGENT_TODAY = GOLDEN["today"]
    semantic_layer._model = None
    yield
    settings.AGENT_TODAY = original
    semantic_layer._model = None


@pytest.mark.parametrize("case", GOLDEN["questions"], ids=lambda c: c["id"])
def test_compiler_matches_reference(case):
    query = SemanticQuery(**case["semantic_query"])
    result = semantic_layer.run(query, max_rows=5000)
    actual = [[row[f["name"]] for f in result["fields"]] for row in result["rows"]]
    expected = duckdb_tool.conn.sql(case["reference_sql"]).fetchall()

    assert expected, f"{case['id']}: the reference query returned no rows, so it can't validate anything"
    ordered = bool(query.sorts)
    diff = rows_match(actual, expected, ordered=ordered)
    assert not diff, f"{case['id']} ({case['question']}): {diff}\nCompiled SQL:\n{result['sql']}"


def test_golden_ids_are_unique():
    ids = [c["id"] for c in GOLDEN["questions"]]
    assert len(ids) == len(set(ids))
