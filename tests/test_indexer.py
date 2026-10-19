import pytest
from app.rag.indexer import indexer
from app.rag.store import vector_store


def test_indexer_execution():
    """Verify that indexer extracts both LookML and lakehouse catalog items."""
    res = indexer.run_indexing()
    assert res["total_indexed"] > 0
    assert res["metrics"] > 0
    assert res["dimensions"] > 0


def test_vector_store_search():
    """Verify semantic/keyword search finds expected metrics."""
    matches = vector_store.search("spend expenses living", top_k=3)
    assert len(matches) > 0
    names = [m.name for m in matches]
    assert any("spend" in n or "expense" in n or "tier" in n for n in names)
