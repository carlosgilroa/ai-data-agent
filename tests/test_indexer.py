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
    # Confirm orphaned item is wiped clean
    assert "dim_categories.stale_orphan_test" not in all_ids

    # Confirm LookML filters__all, aggregation wrapper, and exact physical table_name bindings
    items_by_id = {item.id: item for item in res["items"]}
    assert items_by_id["dim_categories.category_group"].table_name == "dim_categories"
    assert items_by_id["dim_accounts.account_id"].table_name == "dim_accounts"
    assert items_by_id["dim_labels.label_name"].table_name == "dim_labels"
    assert "view.pdt_category_mom_velocity" in items_by_id

    spend_metric = items_by_id.get("fct_transactions.total_spend_cop")
    assert spend_metric is not None
    assert "gross_spend_cop" in (spend_metric.sql_formula or "")
    assert "total_refunds_cop" in (spend_metric.sql_formula or "")
    assert "excluding transfers" in spend_metric.description

    # Confirm dimension_groups and CASE dimensions are extracted
    assert "fct_transactions.transaction" in items_by_id
    tx_type_dim = items_by_id.get("fct_transactions.transaction_type")
    assert tx_type_dim is not None
    assert "CASE WHEN" in (tx_type_dim.sql_formula or "")

    # Confirm table metadata has grain, primary key, and dynamic stats
    exp_table = items_by_id.get("table.mart_expense_summary")
    assert exp_table is not None
    assert "Grain:" in exp_table.description
    assert "Primary Key: expense_summary_pk" in exp_table.description
    assert "Stats:" in exp_table.description
    assert "Date Coverage:" in exp_table.description


def test_vector_store_search_and_similarity_cutoff():
    """Verify dynamic 80% relative cutoff and noise floor behavior."""
    matches = vector_store.search("total_spend_usd", top_k=5)
    assert len(matches) > 0
    top_score = matches[0].similarity_score
    rel_cutoff = top_score * 0.80
    for item in matches:
        assert item.similarity_score is not None
        assert item.distance is not None
        assert item.similarity_score >= rel_cutoff
        assert abs((1.0 - item.distance) - item.similarity_score) < 1e-3

    # Domain entity search ('groceries') succeeds under dynamic cutoff
    grocery_matches = vector_store.search("groceries", top_k=5)
    assert len(grocery_matches) > 0
    matched_ids = {m.id for m in grocery_matches}
    assert "mart_expense_summary.category_name" in matched_ids or "mart_expense_summary.labels" in matched_ids

    # Unrelated query should be filtered out by the 0.20 noise floor threshold
    noise_matches = vector_store.search("weather forecast in bogota tomorrow", top_k=5)
    assert len(noise_matches) == 0
