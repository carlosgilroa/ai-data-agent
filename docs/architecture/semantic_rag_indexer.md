# Semantic RAG Indexer Architecture

This document details the design, LookML AST parsing pipeline, cosine similarity vector store indexing, and retrieval mechanisms of the **Semantic RAG Indexer** (`app/rag/indexer.py` & `app/rag/store.py`).

---

## 1. Single Source of Truth: LookML (Zero Hardcoded Fallbacks)

To prevent metric drift and hallucinated aggregations, `ai-data-agent` establishes **LookML (`projects/wallet-lookml/`)** and physical DuckDB Parquet schemas (`projects/wallet/data/`) as the sole analytical sources of truth:
- **No Static Fallbacks**: All semantic definitions are parsed dynamically from `.view.lkml`, `.model.lkml`, and `.parquet` files.
- **Clean-Slate Re-Indexing**: Every call to `indexer.run_indexing()` invokes `vector_store.clear()`, deleting and recreating the ChromaDB collection so orphaned or stale definitions never persist.

```
+------------------------------------+     +-----------------------------------+
|      LookML Views & Models         |     |      DuckDB Physical Schema       |
|  (views/*.view.lkml, models/*.lkml)|     |  (master_records, fx_rates, marts)|
+------------------+-----------------+     +-----------------+-----------------+
                   |                                         |
                   | AST Parsing (lkml)                      | DESCRIBE Table
                   v                                         v
+----------------------------------------------------------------------------------+
|                            CatalogIndexer Pipeline                               |
|                                                                                  |
|  1. Wipe existing ChromaDB collection (vector_store.clear())                     |
|  2. Parse View Dimensions, Dimension Groups (timeframes) & CASE statements       |
|  3. Parse View Measures with aggregation wrappers (SUM/COUNT/AVG) & filters__all |
|  4. Parse Model Explores & Relational Joins (sql_on, relationship, type)         |
|  5. Resolve LookML views to physical DuckDB tables (master_records, marts, fx)   |
+------------------------------------------+---------------------------------------+
                                           |
                                           | Cosine Space Indexing (hnsw:space="cosine")
                                           v
+----------------------------------------------------------------------------------+
|                             ChromaDB Vector Store                                |
|  - Collection: semantic_catalog (Cosine Distance)                                |
|  - Similarity Score: similarity = 1.0 - distance                                 |
|  - Relevancy Cutoff: min_similarity >= 0.28                                      |
+------------------------------------------+---------------------------------------+
                                           |
                                           | Filtered Top-K Retrieval
                                           v
+----------------------------------------------------------------------------------+
|                      Agent Tool: search_data_catalog                             |
|  Returns high-confidence LookML formulas, filters, tables & similarity scores    |
+----------------------------------------------------------------------------------+
```

---

## 2. LookML AST Parsing (`app/rag/indexer.py`)

The indexer uses the Python `lkml` parser to extract complete semantic metadata from `projects/wallet-lookml`:

1. **Dimensions & `CASE` Expressions**:
   - Extracts standard `sql` mappings as well as LookML `case: { when: ... else: ... }` blocks (compiled into `CASE WHEN ... THEN ... ELSE ... END` SQL formulas).
2. **Dimension Groups (`dimension_groups`)**:
   - Extracts time-based dimension groups (e.g. `transaction` with timeframes `[raw, time, date, week, month, quarter, year]`).
3. **Measures, Aggregation Wrappers & `filters__all`**:
   - Wraps measure SQL expressions in their LookML aggregation type (`sum` $\rightarrow$ `SUM(...)`, `average` $\rightarrow$ `AVG(...)`, `count` $\rightarrow$ `COUNT(...)`).
   - Extracts `filters__all` conditions and appends both a human-readable `[Filters: ...]` tag to the description and a `FILTER (WHERE ...)` clause to `sql_formula`:
     ```text
     [1] total_spend_usd (metric in fct_transactions | Table: master_records | SQL: SUM(ABS(${amount_usd})) FILTER (WHERE is_transfer = no AND amount_usd = <0) | score: 0.634 (distance: 0.366))
         Description: Total cash outflow in USD (excluding transfers) [Filters: is_transfer = no, amount_usd = <0]
     ```
4. **Models, Explores & Relational Joins (`models/**/*.model.lkml`)**:
   - Extracts `explores` and `joins` including `sql_on`, `relationship` (`many_to_one`), and `type` (`left_outer`).
5. **Physical Lakehouse Table Resolution**:
   - Resolves logical LookML views (`fct_transactions`, `dim_categories`, `dim_accounts`, `dim_fx_rates`, `pdt_monthly_cashflow`) to their physical DuckDB Parquet tables (`master_records`, `mart_expense_summary`, `fx_rates`, `mart_monthly_pnl`).

---

## 3. Cosine Similarity Scoring & Relevancy Cutoff (`app/rag/store.py`)

To prevent low-relevance items from polluting the LLM context and triggering exploratory "rabbit hole" tool loops:

- **Cosine Distance Collection**: ChromaDB is initialized with `metadata={"hnsw:space": "cosine"}`.
- **Similarity Formula**:
  $$\text{similarity\_score} = \max(0.0, \text{round}(1.0 - \text{distance}, 4))$$
- **Similarity Threshold (`RAG_MIN_SIMILARITY = 0.28`)**:
  - Every `CatalogItem` returned by `vector_store.search()` includes both `distance` and `similarity_score`.
  - Any candidate result with `similarity_score < 0.28` is discarded before reaching the LLM.
  - Both the `search_data_catalog` tool output and `GET /api/v1/catalog/search` expose the exact similarity and distance scores for full transparency.
