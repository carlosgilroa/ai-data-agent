# REST API Endpoints Reference

The **AI-Native Semantic Query Agent & RAG Microservice** exposes a RESTful interface over HTTP.

- **Base URL**: `http://localhost:8000`
- **API Prefix**: `/api/v1`
- **Interactive OpenAPI UI**: [http://localhost:8000/docs](http://localhost:8000/docs) (Swagger)
- **ReDoc Specification**: [http://localhost:8000/redoc](http://localhost:8000/redoc)

---

## Endpoint Summary

| Method | Endpoint | Description | Tag |
|:---|:---|:---|:---|
| `GET` | [`/api/v1/health`](#1-system-health-check) | Check lakehouse connectivity, Parquet tables, and vector store status | `Health` |
| `GET` | [`/api/v1/catalog`](#2-list-semantic-catalog) | List all indexed LookML measures, dimensions, and lakehouse tables | `Semantic Catalog` |
| `GET` | [`/api/v1/catalog/search`](#3-search-semantic-catalog) | Semantically search LookML metrics and dimensions using ChromaDB | `Semantic Catalog` |
| `POST` | [`/api/v1/catalog/index`](#4-trigger-catalog-re-indexing) | Trigger re-parsing of LookML files and Parquet schemas into vector store | `Semantic Catalog` |
| `POST` | [`/api/v1/agent/query`](#5-natural-language-agent-query) | Primary natural language financial inquiry endpoint with thought traces | `AI Data Agent` |
| `GET` | [`/api/v1/agent/session/{session_id}`](#6-inspect-conversational-session-memory) | Inspect active conversational dialogue turns and message counts | `AI Data Agent` |
| `DELETE` | [`/api/v1/agent/session/{session_id}`](#7-reset-conversational-session-memory) | Clear multi-turn conversational session memory | `AI Data Agent` |
| `POST` | [`/api/v1/agent/sql`](#8-direct-read-only-sql-execution) | Execute read-only DuckDB SQL queries with strict AST safety enforcement | `Data Lakehouse` |
| `POST` | [`/api/v1/semantic/query`](#9-semantic-query-no-llm) | Compile and run a LookML semantic query without calling Gemini | `Semantic Layer` |
| `GET` | [`/api/v1/semantic/explores`](#10-list-and-describe-explores) | List LookML explores | `Semantic Layer` |
| `GET` | [`/api/v1/semantic/explores/{explore}`](#10-list-and-describe-explores) | List an explore's visible dimensions and measures | `Semantic Layer` |

---

## 1. System Health Check

### `GET /api/v1/health`

Verifies the operational status of the service, DuckDB in-memory registration of Parquet files, vector store availability, and Gemini API configuration.

#### Response (`200 OK`)
```json
{
  "status": "healthy",
  "version": "0.1.0",
  "duckdb_connected": true,
  "tables_available": [
    "master_records",
    "dim_accounts",
    "dim_categories",
    "dim_labels",
    "fx_rates",
    "mart_monthly_pnl",
    "mart_expense_summary",
    "pdt_category_mom_velocity"
  ],
  "vector_store_ready": true,
  "gemini_configured": false
}
```

*Note: A valid `GEMINI_API_KEY` is required to run the autonomous agent. If `gemini_configured` is `false`, requests to `/api/v1/agent/query` will return HTTP 400 Bad Request.*

---

## 2. List Semantic Catalog

### `GET /api/v1/catalog`

Returns all verified LookML measures, dimensions, and physical lakehouse table schemas currently held in the semantic index.

#### Response (`200 OK`)
```json
{
  "total": 8,
  "items": [
    {
      "id": "fct_transactions.tier1_operational_spend_usd",
      "name": "tier1_operational_spend_usd",
      "item_type": "metric",
      "view": "fct_transactions",
      "description": "Two-Tier Accounting: Essential routine living burn rate (Tier 1 Needs, rent, groceries, utilities).",
      "sql_formula": "SUM(ABS(amount)) FILTER (WHERE tier_group LIKE '%Tier 1%')",
      "table_name": "master_records",
      "examples": [
        "What is my Tier 1 burn rate?",
        "How much do essential living expenses cost?"
      ]
    }
  ]
}
```

---

## 3. Search Semantic Catalog

### `GET /api/v1/catalog/search`

Semantically searches LookML definitions using ChromaDB vector similarity.

#### Query Parameters
| Parameter | Type | Required | Default | Description |
|:---|:---|:---|:---|:---|
| `q` | `string` | **Yes** | — | Natural language search term (e.g., `savings rate`, `operational burn rate`) |
| `top_k` | `integer`| No | `5` | Maximum number of semantic matches to return (1 to 20) |
| `min_similarity` | `float` | No | `0.28` | Minimum cosine similarity score threshold (0.0 to 1.0) |

#### Response (`200 OK`)
```json
{
  "query": "operational burn rate",
  "total_matches": 2,
  "results": [
    {
      "id": "fct_transactions.tier1_operational_spend_usd",
      "name": "tier1_operational_spend_usd",
      "item_type": "metric",
      "view": "fct_transactions",
      "description": "Two-Tier Accounting: Essential routine living burn rate (Tier 1) [Filters: is_transfer = no, amount_usd = <0, dim_categories.accounting_tier = 1]",
      "sql_formula": "SUM(ABS(${amount_usd})) FILTER (WHERE is_transfer = no AND amount_usd = <0 AND dim_categories.accounting_tier = 1)",
      "table_name": "master_records",
      "examples": [],
      "distance": 0.412,
      "similarity_score": 0.588
    }
  ]
}
```

---

## 4. Trigger Catalog Re-Indexing

### `POST /api/v1/catalog/index`

Wipes the ChromaDB collection clean (`vector_store.clear()`) and forces a live re-parse of LookML views/models in [`wallet-lookml`](file:///c:/projects/wallet-lookml) and DuckDB Parquet schemas in [`wallet/data`](file:///c:/projects/wallet/data).

#### Response (`200 OK`)
```json
{
  "status": "success",
  "total_indexed": 147,
  "metrics_count": 32,
  "dimensions_count": 99,
  "explores_count": 3,
  "joins_count": 5,
  "tables_count": 7
}
```

---

## 5. Natural Language Agent Query

### `POST /api/v1/agent/query`

The primary interface for natural language analytics:
1. The intent router classifies the question (`out_of_scope`, `capabilities`, `simple_query` with 5 steps, `complex_query` with 8 steps) and splits it into measures, dimensions, filter values and a time period.
2. Retrieval picks the best explores and their fields in Chroma, resolves the filter values to exact values in the data (or through the glossary), and adds relevant glossary entries and golden queries to the prompt.
3. The model calls `query` with an explore, fields and filters; the LookML compiler writes and runs the SQL. Errors come back with suggestions so the model can correct itself.
4. Only when no explore fits, and `ALLOW_RAW_SQL_FALLBACK=True`, the model may use read-only raw SQL; the answer then says so and `governed` is `false`.
5. The answer leads with the number in COP and a "Metric:" line, and the response carries everything behind it.

#### Request Body
```json
{
  "query": "What is my net cashflow and savings rate by month?",
  "explain": true
}
```

| Field | Type | Required | Default | Description |
|:---|:---|:---|:---|:---|
| `query` | `string` | **Yes** | — | Natural language inquiry about household finances or expenses |
| `session_id` | `string` | No | `null` | Optional session identifier for multi-turn conversational memory |
| `explain` | `boolean` | No | `true` | If `true`, returns raw table rows (`data`) and intermediate agent reasoning (`thought_steps`) |

#### Successful Response (`200 OK`)
Example with `AGENT_TODAY=2026-10-26` (SQL and tool outputs shortened):
```json
{
  "query": "How much did I spend last month?",
  "intent": "simple_query",
  "model_used": "gemini-3.5-flash-lite",
  "answer": "You spent 12,417,134 COP in September 2026.\nMetric: total spend, net of refunds, excluding transfers.",
  "sql": "SELECT COALESCE(SUM(CASE WHEN ... END), 0) - COALESCE(SUM(CASE WHEN ... END), 0) AS \"fct_transactions.total_spend_cop\" FROM master_records AS fct_transactions WHERE ...",
  "semantic_query": {
    "explore": "fct_transactions",
    "fields": ["fct_transactions.total_spend_cop"],
    "filters": {"fct_transactions.transaction_month": "last month"},
    "sorts": [],
    "limit": 500
  },
  "fields_used": [
    {"name": "fct_transactions.total_spend_cop", "kind": "measure", "type": "number", "description": "..."}
  ],
  "governed": true,
  "notes": [],
  "data": [{"fct_transactions.total_spend_cop": 12417133.52}],
  "thought_steps": [
    {
      "step_number": 1,
      "tool": "intent_router",
      "tool_input": {"query": "How much did I spend last month?", "router_model": "gemini-3.1-flash-lite"},
      "tool_output": {"intent": "simple_query", "max_iterations": 5, "measures": ["total spend"], "filter_values": [], "time_period": "last month"},
      "thought": "..."
    },
    {
      "step_number": 2,
      "tool": "query",
      "tool_input": {"explore": "fct_transactions", "fields": ["fct_transactions.total_spend_cop"], "filters": [{"field": "fct_transactions.transaction_month", "value": "last month"}]},
      "tool_output": "{\"fields\": [...], \"row_count\": 1, \"rows\": [{\"fct_transactions.total_spend_cop\": 12417133.52}]}",
      "thought": "..."
    }
  ],
  "execution_time_ms": 2140.5
}
```

| Field | Description |
|:---|:---|
| `semantic_query` | The explore, fields, filters, sorts and limit behind the answer (`null` for the raw SQL fallback) |
| `fields_used` | The LookML fields used, with their descriptions |
| `governed` | `true` when the numbers come from LookML metrics, `false` for the raw SQL fallback |
| `notes` | Caveats, e.g. "2026-10-01 to 2026-10-31 is only partly covered: the data runs through 2026-10-20." |
| `sql` | The compiled SQL (or the fallback SQL) |
| `data` | Rows of the last query behind the answer |

#### Error Response: Missing LLM Configuration (`400 Bad Request`)

If `GEMINI_API_KEY` is empty or unconfigured in `.env`, the endpoint rejects requests with an explicit error rather than executing brittle heuristics:

```json
{
  "detail": "GEMINI_API_KEY is not configured. An active Google Gemini API key is required to execute the autonomous tool-calling agent loop. Please configure GEMINI_API_KEY in your .env file."
}
```

---

## 6. Inspect Conversational Session Memory

### `GET /api/v1/agent/session/{session_id}`

Retrieves active conversational dialogue turns, turn counts, and message objects for a specific `session_id`.

#### Path Parameters
| Parameter | Type | Required | Description |
|:---|:---|:---|:---|
| `session_id` | `string` | **Yes** | Unique session identifier previously passed in `QueryRequest` |

#### Response (`200 OK`)
```json
{
  "session_id": "session-financial-review-1",
  "turn_count": 1,
  "message_count": 2,
  "dialogue": [
    {
      "type": "HumanMessage",
      "content": "What was my total spending in August 2026?"
    },
    {
      "type": "AIMessage",
      "content": "In August 2026, your total net operational spending was $1,549,008.66...\n\n```sql\n-- Executed Lakehouse Query\nSELECT ...\n```"
    }
  ]
}
```

---

## 7. Reset Conversational Session Memory

### `DELETE /api/v1/agent/session/{session_id}`

Resets and clears the in-memory conversation history and sliding message window for a specific `session_id`.

#### Path Parameters
| Parameter | Type | Required | Description |
|:---|:---|:---|:---|
| `session_id` | `string` | **Yes** | Unique session identifier previously passed in `QueryRequest` |

#### Response (`200 OK`)
```json
{
  "session_id": "session-financial-review-1",
  "cleared": true,
  "message": "Session 'session-financial-review-1' memory cleared successfully."
}
```

If the session ID was not active or already expired, `cleared` is `false`:
```json
{
  "session_id": "session-financial-review-1",
  "cleared": false,
  "message": "Session 'session-financial-review-1' was not active."
}
```

---

## 8. Direct Read-Only SQL Execution

### `POST /api/v1/agent/sql`

Allows direct SQL execution against registered Parquet views (`master_records`, `fx_rates`, `mart_monthly_pnl`, `mart_expense_summary`).

#### Safety Restrictions (AST Guards)
- Only `SELECT` or `WITH` queries are permitted.
- The query must not contain mutating keywords (`DROP`, `DELETE`, `INSERT`, `UPDATE`, `ALTER`, `TRUNCATE`, `ATTACH`, `DETACH`, `PRAGMA`, etc.).

#### Request Body
```json
{
  "sql": "SELECT category_name, SUM(amount) as total FROM master_records WHERE is_transfer = false GROUP BY 1 ORDER BY 2 ASC LIMIT 5"
}
```

#### Response (`200 OK`)
```json
{
  "sql": "SELECT category_name, SUM(amount) as total FROM master_records WHERE is_transfer = false GROUP BY 1 ORDER BY 2 ASC LIMIT 5",
  "row_count": 5,
  "columns": ["category_name", "total"],
  "data": [
    { "category_name": "Rent", "total": -3825644.26 },
    { "category_name": "Groceries", "total": -469905.26 }
  ],
  "execution_time_ms": 3.12
}
```

#### Response on Forbidden Operation (`400 Bad Request`)
```json
{
  "detail": "Forbidden SQL operation detected: 'drop'. Queries must be read-only."
}
```

---

## 9. Semantic Query (no LLM)

### `POST /api/v1/semantic/query`

Compiles a LookML semantic query to SQL and runs it, without calling Gemini. Useful to check a metric by hand.

#### Request Body
```json
{
  "explore": "fct_transactions",
  "fields": ["dim_categories.category_group", "fct_transactions.total_spend_cop"],
  "filters": {"fct_transactions.transaction_month": "2026-09", "dim_categories.category_group": "-Income"},
  "sorts": ["fct_transactions.total_spend_cop desc"],
  "limit": 10
}
```

Filter grammar: `Groceries,Rent`, `-Rent`, `%uber%`, `^,` (literal comma), `yes`/`no`, `>1000000`, `[100,500]`, `2026-09`, `2026-Q3`, `2026-01 through 2026-03`, `before 2026-06`, `last month`, `this year`, `last 3 months`.

#### Response (`200 OK`)
```json
{
  "explore": "fct_transactions",
  "fields": [{"name": "dim_categories.category_group", "kind": "dimension", "type": "string", "description": "..."}, "..."],
  "filters": {"fct_transactions.transaction_month": "2026-09", "dim_categories.category_group": "-Income"},
  "sql": "SELECT ...",
  "row_count": 8,
  "rows": [{"dim_categories.category_group": "Food & Drinks", "fct_transactions.total_spend_cop": 6709751.14}, "..."],
  "notes": []
}
```

An unknown field, a hidden field or a bad filter returns `400 Bad Request` with suggestions, e.g. `No field 'fct_transactions.total_spend' in explore 'fct_transactions'. Did you mean: fct_transactions.total_spend_cop, ...?`

---

## 10. List and Describe Explores

### `GET /api/v1/semantic/explores`
Returns `{"explores": [{"name", "label", "description"}, ...]}`.

### `GET /api/v1/semantic/explores/{explore}`
Returns `{"explore": "...", "fields": [{"name", "kind", "type", "description"}, ...]}` with the visible dimensions and measures. Unknown explores return `404`.

---

## Client Usage Examples

### Python (`httpx` or `requests`)
```python
import httpx

client = httpx.Client(base_url="http://localhost:8000/api/v1")

# Natural Language Query
res = client.post("/agent/query", json={
    "query": "What are my top 5 expense categories?",
    "explain": True
})
data = res.json()
print("Answer:", data["answer"])
print("SQL:", data["sql"])
```

### PowerShell / cURL
```powershell
curl -X POST "http://localhost:8000/api/v1/agent/query" `
  -H "Content-Type: application/json" `
  -d '{"query": "Show monthly savings rate", "explain": true}'
```
