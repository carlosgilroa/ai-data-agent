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
|  5. Strip @{schema_name}. macro & register LookML PDTs (pdt_category_mom_velocity)|
|  6. Index 7 Physical DuckDB Parquet Tables (master_records, dims, fx, marts)     |
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
5. **Direct `sql_table_name` Binding & Persistent Derived Tables (PDTs)**:
   - Strips LookML project macro prefixes (e.g. `` `@{schema_name}.dim_accounts` `` $\rightarrow$ `dim_accounts`) via `_strip_lookml_table_macro()`, mapping every LookML view 1:1 to its physical Parquet table (`master_records`, `dim_accounts`, `dim_categories`, `dim_labels`, `fx_rates`, `mart_monthly_pnl`, `mart_expense_summary`) without hardcoded view-resolution dicts.
   - Extracts LookML `derived_table` definitions (`pdt_category_mom_velocity`) as `view` catalog items and compiles them directly as DuckDB SQL views.
   - When `search_data_catalog` returns matched dimensions or metrics, it automatically appends the physical column schema (`[Table Schema: <table_name>]`) for each referenced table so the LLM never guesses column names.

6. **Dynamic Table Statistics & Truncation Observability**:
   - At indexing time, DuckDB dynamically inspects each Parquet table to capture total row count (`COUNT(*)`) and temporal bounds (`MIN/MAX` over `date` or `year_month`), combining them with the LookML view's declared grain and primary key:
     ```text
     [Table Schema: mart_expense_summary] Monthly expense summary segmented by accounting tier, category, merchant, and labels. Grain: 1 row per year_month + tier_group + category_name + merchant + labels. | Primary Key: expense_summary_pk | Stats: 550 rows | Date Coverage: 2026-01 to 2026-10 | Available columns: ...
     ```
   - In `run_sql_query`, the tool response explicitly returns `total_rows_returned` and `preview_rows`. If total rows exceed the preview limit, a notice informs the model to use aggregated SQL (`GROUP BY` and `SUM/COUNT`) rather than computing sums on truncated preview samples.

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
