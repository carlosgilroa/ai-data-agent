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
| `POST` | [`/api/v1/agent/sql`](#6-direct-read-only-sql-execution) | Execute read-only DuckDB SQL queries with strict AST safety enforcement | `Data Lakehouse` |

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
    "fx_rates",
    "mart_monthly_pnl",
    "mart_expense_summary"
  ],
  "vector_store_ready": true,
  "gemini_configured": false
}
```

*Note: If `gemini_configured` is `false`, the agent operates in intelligent semantic heuristic fallback mode.*

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
      "description": "Two-Tier Accounting: Essential routine living burn rate (Tier 1 Needs, rent, groceries, utilities).",
      "sql_formula": "SUM(ABS(amount)) FILTER (WHERE tier_group LIKE '%Tier 1%')",
      "table_name": "master_records",
      "examples": ["What is my Tier 1 burn rate?"]
    }
  ]
}
```

---

## 4. Trigger Catalog Re-Indexing

### `POST /api/v1/catalog/index`

Forces a live re-parse of LookML views in [`wallet-lookml`](file:///c:/projects/wallet-lookml) and DuckDB Parquet schemas in [`wallet/data`](file:///c:/projects/wallet/data), updating ChromaDB embeddings.

#### Response (`200 OK`)
```json
{
  "status": "success",
  "total_indexed": 8,
  "metrics_count": 6,
  "dimensions_count": 2,
  "tables_count": 4
}
```

---

## 5. Natural Language Agent Query

### `POST /api/v1/agent/query`

The primary interface for natural language analytics. The orchestrator retrieves relevant LookML context, formulates SQL, executes it against DuckDB in < 10 ms, and synthesizes an answer.

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
| `explain` | `boolean` | No | `false` | If `true`, returns raw table rows (`data`) and intermediate agent reasoning (`thought_steps`) |

#### Response (`200 OK`)
```json
{
  "query": "What is my net cashflow and savings rate by month?",
  "answer": "Based on reconciled monthly P&L records (via LookML mart_monthly_pnl):\n- Latest Month (2026-09): Net Cash Flow was -8,926,969.00 with a savings rate of -308.6%.\n- Total operating income was 1,351,550.00 against operational expenses of 5,522,875.06.",
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
      "tool": "search_data_catalog",
      "tool_input": { "query": "What is my net cashflow and savings rate by month?" },
      "tool_output": "Matched mart_monthly_pnl and savings_rate_percentage",
      "thought": "Searching semantic catalog for metrics and dimensions..."
    },
    {
      "step_number": 2,
      "tool": "run_sql_query",
      "tool_input": { "sql": "SELECT year_month, operating_income, operational_expenses, net_cashflow, savings_rate_pct FROM mart_monthly_pnl ORDER BY year_month ASC LIMIT 12;" },
      "tool_output": "Returned 1 rows. Columns: ['year_month', 'operating_income', 'operational_expenses', 'net_cashflow', 'savings_rate_pct']",
      "thought": "Executing LookML-aligned SQL against local DuckDB Lakehouse..."
    }
  ],
  "execution_time_ms": 6.44
}
```

---

## 6. Direct Read-Only SQL Execution

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
