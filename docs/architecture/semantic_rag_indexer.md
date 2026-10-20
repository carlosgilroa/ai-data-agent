# Semantic RAG Indexer Architecture

This document details the design, AST parsing pipeline, vector store indexing, and retrieval mechanisms of the **Semantic RAG Indexer** (`app/rag/indexer.py`).

---

## 1. Single Source of Truth: LookML

In financial data applications, metric drift and hallucinated aggregations (e.g., summing multi-currency values without FX conversion, or treating internal transfers as expenses) are critical failure modes.

To prevent this, `ai-data-agent` establishes **LookML (`projects/wallet-lookml/`)** as the sole analytical source of truth:
- **Views (`views/**/*.view.lkml`)**: Define verified metric formulas, SQL column mappings, and descriptive business metadata.
- **Models & Explores (`models/**/*.model.lkml`)**: Define relational joins, `sql_on` conditions, and entity relationships (e.g. `many_to_one`).

The agent **never** writes ad-hoc metric definitions. Instead, it queries the Semantic RAG Indexer to retrieve verified LookML formulas before formulating SQL.

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
|  1. Parse View Dimensions & Measures                                             |
|  2. Parse Model Explores & Relational Joins (sql_on, relationship, type)          |
|  3. Parse Physical Lakehouse Column Types & Constraints                          |
|  4. Transform into standardized CatalogItem objects                              |
+------------------------------------------+---------------------------------------+
                                           |
                                           | Text Embeddings (all-MiniLM-L6-v2)
                                           v
+----------------------------------------------------------------------------------+
|                             ChromaDB Vector Store                                |
|  - Collection: lookml_semantic_catalog                                           |
|  - Metadata filtering: item_type in [measure, dimension, explore, join, table]   |
+------------------------------------------+---------------------------------------+
                                           |
                                           | Top-K Vector Similarity Search
                                           v
+----------------------------------------------------------------------------------+
|                      Agent Tool: search_data_catalog                             |
|  Returns formatted business definitions, SQL snippets, and join paths to LLM     |
+----------------------------------------------------------------------------------+
```

---

## 2. AST Parsing with `lkml`

The indexer employs the native Python `lkml` parser to convert LookML files into abstract syntax trees without heuristic regex or error-prone string splitting.

### A. View Ingestion (`views/**/*.view.lkml`)
For every view file, the parser iterates over:
- **Dimensions**: Extracts `name`, `type`, `sql`, and `description`.
- **Measures**: Extracts metric aggregations (e.g., `sum`, `count`, `average`, `number`), calculation formulas, and descriptions.

Example parsed output:
```text
[LookML Measure] total_spend_cop
  View: expense_transactions
  Type: sum
  SQL: ${amount_cop}
  Description: Total expenses in Colombian Pesos excluding transfers and investments.
```

### B. Model & Explore Ingestion (`models/**/*.model.lkml`)
For every model file, the parser extracts:
- **Explores**: The primary root view and business scope.
- **Joins**:
  - `name`: Target joined view.
  - `sql_on`: Precise SQL equality/join condition (e.g., `${transactions.account_id} = ${accounts.id}`).
  - `relationship`: Cardinality (`many_to_one`, `one_to_one`, `one_to_many`).
  - `type`: Join style (`left_outer`, `inner`, `full_outer`).

Example parsed output:
```text
[LookML Join] fx_rates
  Explore: wallet_analytics
  Joined View: fx_rates
  Relationship: many_to_one
  Join Type: left_outer
  SQL ON: ${master_records.date} = ${fx_rates.date} AND ${master_records.currency} = ${fx_rates.currency}
  Description: Daily FX rates joined on transaction date and currency.
```

### C. Physical Lakehouse Schema Ingestion
The indexer queries DuckDB using pure-Python `.fetchall()` calls (`DESCRIBE <table_name>`) to capture exact physical column types without loading data into memory or relying on external heavy dependencies.

---

## 3. Vector Storage & Embeddings

The catalog indexer persists embeddings locally in ChromaDB:

- **Embeddings Model**: `sentence-transformers/all-MiniLM-L6-v2` via LangChain `HuggingFaceEmbeddings`.
  - Lightweight, fast CPU inference, runs completely offline with zero API cost.
- **Persistence Path**: `data/chroma/`.
- **Collection Name**: `lookml_semantic_catalog`.
- **Document Payloads**: Formatted searchable text strings containing the item name, type, description, and underlying SQL formula.
- **Metadata Attributes**:
  - `name`: Identifier (e.g., `total_spend_cop`).
  - `item_type`: Categorization (`measure`, `dimension`, `explore`, `join`, `table`).
  - `source_file`: File path of origin.
  - `view_name`: Parent LookML view.

---

## 4. Search & Retrieval Tool

The agent invokes the `search_data_catalog` tool during its ReAct loop:

```python
@tool
def search_data_catalog(query: str) -> str:
    """Search the LookML semantic catalog and lakehouse schema for business metric definitions,
    calculation formulas, dimension names, and table structures.
    """
    return raw_search_catalog(query, top_k=5)
```

The tool performs similarity search on ChromaDB, returning top-$k$ semantic items formatted as concise Markdown snippets. This provides the LLM with exact column names, table names, and arithmetic definitions right before it writes DuckDB SQL queries.

---

## 5. Lifecycle & Re-Indexing

1. **Automatic Initialization**:
   - During FastAPI startup (`app/main.py`), the indexer automatically checks if the ChromaDB catalog exists. If empty, it indexes all LookML files and DuckDB tables automatically.
2. **On-Demand Re-Indexing**:
   - `POST /api/v1/catalog/index`: Triggers full re-parsing of LookML views, models, and lakehouse schemas.
   - Clears existing ChromaDB collection documents and rebuilds fresh embeddings.
   - Returns indexed item counts:
     ```json
     {
       "status": "success",
       "indexed_count": 54,
       "message": "Indexed 54 LookML metrics, explores, joins, and lakehouse schema items."
     }
     ```
