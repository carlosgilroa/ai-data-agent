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
| `DELETE` | [`/api/v1/agent/session/{session_id}`](#6-reset-conversational-session-memory) | Clear multi-turn conversational session memory | `AI Data Agent` |
| `POST` | [`/api/v1/agent/sql`](#7-direct-read-only-sql-execution) | Execute read-only DuckDB SQL queries with strict AST safety enforcement | `Data Lakehouse` |

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

The primary interface for natural language analytics. Executes a **Two-Step Intent Router (`gemini-3.1-flash-lite`)** followed by an autonomous **ReAct tool-calling loop**:
1. Classifies user intent into `out_of_scope` (`0` steps), `capabilities` (`1` step on `gemini-3.1-flash-lite`), `simple_query` (`3` steps on `gemini-3.5-flash-lite`), or `complex_query` (`6` steps on `gemini-3.8-flash`).
2. Dynamically invokes `search_data_catalog` (filtered by cosine similarity $\ge 0.28$) to discover LookML measures, business rules, and table schemas.
3. Formulates LookML-aligned SQL and invokes `run_sql_query` against the local DuckDB lakehouse.
4. Automatically self-corrects SQL if a syntax error or missing column is reported by DuckDB.
5. Synthesizes an accurate, grounded financial answer with full thought-step traceability.

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
```json
{
  "query": "What is my net cashflow and savings rate by month?",
  "intent": "complex_query",
  "model_used": "gemini-3.8-flash",
  "answer": "Based on reconciled monthly P&L records (via LookML mart_monthly_pnl)...",
  "sql": "SELECT year_month, operating_income, operational_expenses, net_cashflow, savings_rate_pct FROM mart_monthly_pnl ORDER BY year_month ASC LIMIT 12;",
  "data": [
    {
      "year_month": "2026-09",
      "operating_income": 1351550.0,
      "operational_expenses": 5522875.06,
      "net_cashflow": -8926969.0,
      "savings_rate_pct": -308.6
    }
  ],
  "thought_steps": [
    {
      "step_number": 1,
      "tool": "intent_router",
      "tool_input": { "query": "What is my net cashflow and savings rate by month?", "router_model": "gemini-3.1-flash-lite" },
      "tool_output": { "intent": "complex_query", "target_model": "gemini-3.8-flash", "max_iterations": 6 },
      "thought": "Router (gemini-3.1-flash-lite) classified intent as 'complex_query' -> routed to 'gemini-3.8-flash' with max_iterations=6"
    },
    {
      "step_number": 2,
      "tool": "search_data_catalog",
      "tool_input": { "query": "net cashflow savings rate monthly" },
      "tool_output": "[1] net_cash_flow_usd (metric in fct_transactions | Table: master_records | SQL: ${total_income_usd} - ${total_spend_usd} | score: 0.550 (distance: 0.450))...",
      "thought": "Agent (gemini-3.8-flash) decided to invoke tool 'search_data_catalog'..."
    },
    {
      "step_number": 3,
      "tool": "run_sql_query",
      "tool_input": { "sql": "SELECT year_month, operating_income, operational_expenses, net_cashflow, savings_rate_pct FROM mart_monthly_pnl ORDER BY year_month ASC LIMIT 12;" },
      "tool_output": "{\"columns\": [\"year_month\", \"operating_income\", \"operational_expenses\", \"net_cashflow\", \"savings_rate_pct\"], \"row_count\": 1, \"rows\": [...]}",
      "thought": "Agent (gemini-3.8-flash) decided to invoke tool 'run_sql_query'..."
    }
  ],
  "execution_time_ms": 4820.12
}
```

#### Error Response: Missing LLM Configuration (`400 Bad Request`)

If `GEMINI_API_KEY` is empty or unconfigured in `.env`, the endpoint rejects requests with an explicit error rather than executing brittle heuristics:

```json
{
  "detail": "GEMINI_API_KEY is not configured. An active Google Gemini API key is required to execute the autonomous tool-calling agent loop. Please configure GEMINI_API_KEY in your .env file."
}
```

---

## 6. Reset Conversational Session Memory

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

## 7. Direct Read-Only SQL Execution

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
