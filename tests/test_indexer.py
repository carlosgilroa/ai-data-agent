from app.rag.indexer import indexer
from app.rag.store import vector_store
from app.schemas.catalog import CatalogItem


def test_indexer_execution():
    """Verify that indexer extracts both LookML and lakehouse catalog items on a clean slate."""
    # Seed a dummy stale item first to confirm run_indexing() wipes ChromaDB clean
    vector_store.add_items([
        CatalogItem(
            id="dim_categories.stale_orphan_test",
            item_type="dimension",
            name="stale_orphan_test",
            view="dim_categories",
            description="Orphaned item that must be wiped by run_indexing",
            sql_formula="stale_col",
            table_name="master_records",
        )
    ])

    res = indexer.run_indexing()
    assert res["total_indexed"] > 0
    assert res["metrics"] > 0
    assert res["dimensions"] > 0
    assert res["explores"] > 0
    assert res["joins"] > 0
    assert res["tables"] > 0

    all_ids = {item.id for item in res["items"]}
    # Confirm orphaned item and removed hardcoded fallback item are absent
    assert "dim_categories.stale_orphan_test" not in all_ids
    assert "dim_categories.category_group" not in all_ids

    # Confirm LookML filters__all and aggregation wrapper are extracted
    items_by_id = {item.id: item for item in res["items"]}
    spend_metric = items_by_id.get("fct_transactions.total_spend_usd")
    assert spend_metric is not None
    assert "SUM(" in (spend_metric.sql_formula or "")
    assert "is_transfer = no" in spend_metric.description

    # Confirm dimension_groups and CASE dimensions are extracted
    assert "fct_transactions.transaction" in items_by_id
    tx_type_dim = items_by_id.get("fct_transactions.transaction_type")
    assert tx_type_dim is not None
    assert "CASE WHEN" in (tx_type_dim.sql_formula or "")


def test_vector_store_search_and_similarity_cutoff():
    """Verify cosine similarity scores are attached and noise below min_similarity is filtered out."""
    matches = vector_store.search("total_spend_usd", top_k=5, min_similarity=0.28)
    assert len(matches) > 0
    for item in matches:
        assert item.similarity_score is not None
        assert item.distance is not None
        assert item.similarity_score >= 0.28
        assert abs((1.0 - item.distance) - item.similarity_score) < 1e-3

    # Unrelated query should be filtered out by the 0.28 cosine similarity threshold
    noise_matches = vector_store.search("weather forecast in bogota tomorrow", top_k=5, min_similarity=0.28)
    assert len(noise_matches) == 0
