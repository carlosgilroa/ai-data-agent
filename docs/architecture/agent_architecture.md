# AI Data Agent: System Architecture & Two-Step ReAct Orchestration

This document details the architectural design, two-step intent routing, reasoning loop, session memory management, and safety mechanisms of the **AI Data Agent** (`app/agent/`).

---

## 1. Architectural Overview

The AI Data Agent is an autonomous, multi-tier tool-calling reasoning engine built with **LangChain** and **Google Gemini** (`gemini-3.1-flash-lite`, `gemini-3.5-flash-lite`, and `gemini-3.8-flash`). It bridges natural language financial and data engineering queries to local columnar lakehouse tables (`projects/wallet/data/`) using verified LookML semantic models (`projects/wallet-lookml/`).

```
                              +---------------------------+
                              |   Client (REST / UI)      |
                              +-------------+-------------+
                                            |
                                            | POST /api/v1/agent/query
                                            v
+-----------------------------------------------------------------------------------------+
|                                    AgentRunner                                          |
|                                                                                         |
|  +-----------------------------------------------------------------------------------+  |
|  | Step 1: Two-Step IntentRouter (gemini-3.1-flash-lite)                             |  |
|  | Classifies query into: out_of_scope | capabilities | simple_query | complex_query |  |
|  +----------------------------------------+------------------------------------------+  |
|                                           |                                             |
|        +----------------------+-----------+-----------+----------------------+          |
|        | out_of_scope         | capabilities          | simple_query         | complex  |
|        v (0 tool steps)       v (max 1 step)          v (max 3 steps)        v (max 6)  |
|  +----------------+   +-------------------+   +--------------------+   +--------------+ |
|  | 3.1-flash-lite |   | 3.1-flash-lite    |   | 3.5-flash-lite     |   | 3.8-flash    | |
|  | (No tools)     |   | (Catalog tool)    |   | (Catalog + DuckDB) |   | (Full ReAct) | |
|  +----------------+   +---------+---------+   +---------+----------+   +------+-------+ |
|                                 |                       |                     |         |
|                                 +-----------+-----------+---------------------+         |
|                                             |                                           |
|                                             v                                           |
|                      +---------------------------+       +----------------------------+ |
|                      |   search_data_catalog     |       |       run_sql_query        | |
|                      | (Cosine similarity >=0.28)|       |  (Read-Only AST Lakehouse) | |
|                      +-------------+-------------+       +--------------+-------------+ |
+------------------------------------|------------------------------------|---------------+
                                     |                                    |
                                     v                                    v
                       +---------------------------+       +----------------------------+
                       |   ChromaDB Vector Store   |       |  Parquet Lakehouse Tables  |
                       |  (LookML Semantics & AST) |       |  (master_records, fx_rates)|
                       +---------------------------+       +----------------------------+
```

---

## 2. Two-Step Intent Router (`app/agent/router.py`)

To optimize latency and token cost while preventing unnecessary exploratory tool loops, every incoming request first passes through [`IntentRouter`](file:///c:/projects/ai-data-agent/app/agent/router.py).

### Step 1: Classification (`gemini-3.1-flash-lite`)
`gemini-3.1-flash-lite` inspects the user prompt and outputs a structured `RouterDecision` (`intent` and `reasoning`) with zero tools bound.

### Step 2: Routed Model & Dynamic Iteration Budget
Based on the classified `intent`, `AgentRunner` selects the target Gemini model, allowed tools, and `max_iterations` ceiling:

| Intent Category | Target Execution Model | Allowed Tools | `max_iterations` | Purpose |
|:---|:---|:---|:---:|:---|
| `out_of_scope` | `gemini-3.1-flash-lite` | *None* | `0` | Politely declines non-financial/unrelated trivia questions without touching ChromaDB or DuckDB. |
| `capabilities` | `gemini-3.1-flash-lite` | `search_data_catalog` | `1` | Grounds "what can you do?" responses in live indexed LookML tables/metrics. |
| `simple_query` | `gemini-3.5-flash-lite` | `search_data_catalog`, `run_sql_query` | `3` | Fast, cost-efficient execution for single-metric lookups, counts, and single-table aggregations. |
| `complex_query` | `gemini-3.8-flash` | `search_data_catalog`, `run_sql_query` | `6` | Deep multi-hop reasoning, period-over-period comparisons, Tier 1 vs Tier 2 reconciliations, and joins. |

---

## 3. Minimalist Prompt Design

Unlike legacy agents that overload the system prompt with hardcoded table schemas, database columns, business rules, or tool instructions, the AI Data Agent enforces a **pure minimalist prompt** ([`app/agent/prompts.py`](file:///c:/projects/ai-data-agent/app/agent/prompts.py)).

### Principles
1. **Dynamic Tool Schema via API**: Tool descriptions, parameters, and types are declared natively as LangChain `@tool` definitions and bound directly to Gemini via `llm.bind_tools(...)`.
2. **No Hardcoded Schemas or Rules**: Table schemas, LookML view definitions, accounting logic (e.g. transfers vs expenses, multi-currency conversion), and join conditions reside strictly in LookML and DuckDB.
3. **Role & Grounding Directives Only**:
   - Instructs the agent on analytical persona (Senior Financial Analyst & Analytics Engineer).
   - Enforces grounding: never invent metrics, dimensions, or tables.
   - Enforces tool usage: search catalog for metric definitions before writing SQL; execute SQL to verify data before answering.

---

## 4. Autonomous ReAct Reasoning Loop

The core orchestration resides in [`AgentRunner.run_query()`](file:///c:/projects/ai-data-agent/app/agent/runner.py):

1. **Router Classification (`Step 1`)**: Records an `intent_router` thought step detailing the chosen intent, target model, and step budget.
2. **Out-of-Scope Short Path**: If `intent == "out_of_scope"`, invokes `gemini-3.1-flash-lite` without tools to return a polite scope refusal.
3. **ReAct Iteration Step (up to `max_iterations`)**:
   - Passes message history to the routed `routed_llm_with_tools.invoke(messages)`.
   - Executes `search_data_catalog` (cosine similarity filtered $\ge 0.28$) or `run_sql_query` (DuckDB execution).
   - Captures output into an `AgentThoughtStep` for full observability.
   - **Self-Correction on Error**: If DuckDB returns an error, the raw error is returned to Gemini in the `ToolMessage` for automatic query refinement.
4. **Synthesis**: When Gemini outputs a response without tool calls (or when `max_iterations` is reached, using `tool_choice="none"`), it synthesizes a grounded analytical answer.

---

## 5. Multi-Turn Session Memory

Conversational memory enables natural back-and-forth financial exploration:

- **Keyed by Session ID**: Requests provide an optional `session_id` in `QueryRequest`.
- **In-Memory Store**: Stored in `agent_runner.sessions: Dict[str, List[BaseMessage]]`.
- **15-Message Sliding Window**: Retains the initial `SystemMessage` plus the latest 14 messages:
  ```python
  if len(messages) > 15:
      messages = [messages[0]] + messages[-14:]
  ```
- **Lifecycle Management**:
  - `DELETE /api/v1/agent/session/{session_id}` explicitly purges session state.

---

## 6. SQL Safety & Execution Guardrails

1. **Read-Only Enforced**:
   - AST token inspection rejects any query containing mutating keywords (`INSERT`, `UPDATE`, `DELETE`, `DROP`, `ALTER`, `CREATE`, `REPLACE`, `TRUNCATE`, `GRANT`, `REVOKE`, `VACUUM`).
2. **Table & Derived View Whitelist**:
   - Queries can access the 7 verified lakehouse Parquet tables (`master_records`, `dim_accounts`, `dim_categories`, `dim_labels`, `fx_rates`, `mart_monthly_pnl`, `mart_expense_summary`) plus LookML Persistent Derived Tables compiled in DuckDB (`pdt_category_mom_velocity`).
3. **Pure Python DuckDB Execution**:
   - Native `.fetchall()` bindings ensure a fast, pandas-free memory footprint.
