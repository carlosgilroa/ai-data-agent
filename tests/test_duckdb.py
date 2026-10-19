import pytest
from app.agent.tools.duckdb_tool import duckdb_tool


def test_duckdb_views_registered():
    """Verify DuckDB registers available Parquet views."""
    tables = duckdb_tool.get_available_tables()
    assert "master_records" in tables or len(tables) > 0


def test_duckdb_query_execution():
    """Verify DuckDB can safely run a SELECT query over registered views."""
    result = duckdb_tool.run_query("SELECT 1 AS num, 'test' AS label")
    assert result["row_count"] == 1
    assert result["columns"] == ["num", "label"]
    assert result["data"][0]["num"] == 1


def test_duckdb_forbidden_operations():
    """Verify SQL safety validator blocks destructive commands."""
    with pytest.raises(ValueError, match="Only SELECT or WITH queries are permitted"):
        duckdb_tool.run_query("DROP TABLE master_records")

    with pytest.raises(ValueError, match="Only SELECT or WITH queries are permitted"):
        duckdb_tool.run_query("INSERT INTO master_records VALUES (1, 2)")

    with pytest.raises(ValueError, match="Forbidden SQL operation detected: 'drop'"):
        duckdb_tool.run_query("SELECT * FROM master_records WHERE id = (drop table test)")


def test_query_master_records_if_present():
    """Verify query over master_records filters transfers."""
    tables = duckdb_tool.get_available_tables()
    if "master_records" in tables:
        res = duckdb_tool.run_query(
            "SELECT count(*) as total_transfers FROM master_records WHERE is_transfer = true"
        )
        assert res["row_count"] == 1
        assert "total_transfers" in res["data"][0]
