"""The agent reloads the parquet files when the wallet pipeline rewrites them, without a restart."""

import os
import time

import duckdb
import pytest

from app.agent.tools.duckdb_tool import DuckDBTool
from app.core.config import settings


def _write_records(path, amounts):
    path.parent.mkdir(parents=True, exist_ok=True)
    values = ", ".join(f"({a})" for a in amounts)
    duckdb.sql(f"COPY (SELECT * FROM (VALUES {values}) t(amount)) TO '{path.as_posix()}' (FORMAT parquet)")


def _age(path, seconds):
    """Pretend the file was last written `seconds` ago."""
    past = time.time() - seconds
    os.utime(path, (past, past))


@pytest.fixture
def tool(tmp_path, monkeypatch):
    lookml = tmp_path / "lookml"
    lookml.mkdir()
    monkeypatch.setattr(settings, "WALLET_DATA_DIR", tmp_path / "data")
    monkeypatch.setattr(settings, "LOOKML_DIR", lookml)
    records = settings.silver_records_path
    _write_records(records, [10, 20])
    _age(records, 600)
    return DuckDBTool(), records


def _total(tool):
    return tool.run_query("SELECT SUM(amount) AS s FROM master_records")["data"][0]["s"]


def test_unchanged_files_are_not_reloaded(tool):
    t, _ = tool
    assert t.refresh_if_changed(stable_seconds=0) is False
    assert _total(t) == 30


def test_changed_file_is_reloaded_once_it_is_stable(tool):
    t, records = tool
    _write_records(records, [10, 20, 70])

    # Just written: it may still be half written, so wait
    assert t.refresh_if_changed(stable_seconds=60) is False
    assert _total(t) == 30

    _age(records, 120)
    assert t.refresh_if_changed(stable_seconds=60) is True
    assert _total(t) == 100
    assert t.refresh_if_changed(stable_seconds=60) is False


def test_broken_file_keeps_the_loaded_data(tool):
    t, records = tool
    records.write_bytes(b"not a parquet file")
    _age(records, 120)

    assert t.refresh_if_changed(stable_seconds=60) is False
    assert _total(t) == 30

    # Fixed on the next pipeline run: picked up then
    _write_records(records, [5])
    _age(records, 120)
    assert t.refresh_if_changed(stable_seconds=60) is True
    assert _total(t) == 5


def test_reloaded_data_stays_locked_down(tool):
    t, records = tool
    _write_records(records, [1])
    _age(records, 120)
    assert t.refresh_if_changed(stable_seconds=0) is True
    with pytest.raises(Exception):
        t.conn.execute("SELECT * FROM read_text('.env')").fetchall()
