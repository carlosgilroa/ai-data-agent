# AI Data Agent: System Architecture & ReAct Orchestration

This document details the architectural design, reasoning loop, session memory management, and safety mechanisms of the **AI Data Agent** (`app/agent/`).

---

## 1. Architectural Overview

The AI Data Agent is an autonomous, tool-calling reasoning engine built with **LangChain** and **Google Gemini** (`gemini-3.8-flash`). It bridges natural language financial and data engineering queries to local columnar lakehouse tables (`projects/wallet/data/`) using verified LookML semantic models (`projects/wallet-lookml/`).

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
|  +--------------------+     +------------------------+     +-------------------------+  |
|  | Minimalist System  | --> |  Multi-Turn Session    | --> | Gemini ReAct Loop       |  |
|  | Prompt (Zero-Shot) |     |  Memory (Sliding Win)  |     | (gemini-3.8-flash)      |  |
|  +--------------------+     +------------------------+     +------------+------------+  |
|                                                                         |               |
|                                     Tool Calling Execution Loop         |               |
|                                    +------------------------------------+               |
|                                    |                                    |               |
|                                    v                                    v               |
|                      +---------------------------+       +----------------------------+ |
|                      |   search_data_catalog     |       |       run_sql_query        | |
|                      +-------------+-------------+       +--------------+-------------+ |
+------------------------------------|------------------------------------|---------------+
                                     |                                    |
                                     v                                    v
                       +---------------------------+       +----------------------------+
                       |   ChromaDB Vector Store   |       |       DuckDBTool           |
                       |  (LookML Semantics & AST) |       |  (Read-Only AST Lakehouse) |
                       +---------------------------+       +--------------+-------------+
                                                                          |
                                                                          v
                                                           +----------------------------+
                                                           |  Parquet Lakehouse Tables  |
                                                           |  (master_records, fx_rates)|
                                                           +----------------------------+
```

---

## 2. Minimalist Prompt Design

Unlike legacy agents that overload the system prompt with hardcoded table schemas, database columns, business rules, or tool instructions, the AI Data Agent enforces a **pure minimalist prompt** ([`app/agent/prompts.py`](file:///c:/projects/ai-data-agent/app/agent/prompts.py)).

### Principles
1. **Dynamic Tool Schema via API**: Tool descriptions, parameters, and types are declared natively as LangChain `@tool` definitions and bound directly to Gemini via `llm.bind_tools([search_data_catalog, run_sql_query])`. Gemini receives exact JSON Schema definitions natively through the API.
2. **No Hardcoded Schemas or Rules**: Table schemas, LookML view definitions, accounting logic (e.g. transfers vs expenses, multi-currency conversion), and join conditions reside strictly in LookML and DuckDB. The agent dynamically retrieves them on-demand via `search_data_catalog`.
3. **Role & Grounding Directives Only**:
   - Instructs the agent on analytical persona (Senior Financial Analyst & Analytics Engineer).
   - Enforces grounding: never invent metrics, dimensions, or tables.
   - Enforces tool usage: search catalog for metric definitions before writing SQL; execute SQL to verify data before answering.

---

## 3. Autonomous ReAct Reasoning Loop

The core orchestration resides in [`AgentRunner.run_query()`](file:///c:/projects/ai-data-agent/app/agent/runner.py):

1. **Session Context Ingestion**: Prepares conversation history from `self.sessions[session_id]` if active.
2. **ReAct Iteration Step (up to 8 iterations)**:
   - Passes message history to `self.llm_with_tools.invoke(messages)`.
   - If Gemini returns tool calls (`ai_msg.tool_calls`):
     - Executes `search_data_catalog` (catalog semantic lookup) or `run_sql_query` (DuckDB execution).
     - Captures output into an `AgentThoughtStep` for full observability.
     - Appends `ToolMessage` with the exact execution output back to the conversation.
   - **Self-Correction on Error**: If DuckDB returns an error (e.g., column mismatch, missing date cast), the raw error is returned to Gemini in the `ToolMessage`. Gemini reads the error, refines its SQL, and retries automatically.
3. **Synthesis**: When Gemini outputs a response without tool calls (or max iterations are reached), it synthesizes a final analytical answer grounded in the retrieved lakehouse data.

---

## 4. Multi-Turn Session Memory

Conversational memory enables natural back-and-forth financial exploration (e.g. *"What were my food expenses in March 2026?"* followed by *"Compare that to February"*).

- **Keyed by Session ID**: Requests provide an optional `session_id` in `QueryRequest`.
- **In-Memory Store**: Stored in `agent_runner.sessions: Dict[str, List[BaseMessage]]`.
- **15-Message Sliding Window**: To prevent LLM context exhaustion while maintaining conversation coherence, history retains the initial `SystemMessage` plus the latest 14 messages:
  ```python
  if len(messages) > 15:
      messages = [messages[0]] + messages[-14:]
  ```
- **Lifecycle Management**:
  - `DELETE /api/v1/agent/session/{session_id}` explicitly purges session state.
  - Returns `{"session_id": "...", "cleared": true}`.

---

## 5. SQL Safety & Execution Guardrails

To prevent data corruption, resource starvation, or prompt injection attacks via SQL:

1. **Read-Only Enforced**:
   - AST token inspection rejects any query containing mutating keywords: `INSERT`, `UPDATE`, `DELETE`, `DROP`, `ALTER`, `CREATE`, `REPLACE`, `TRUNCATE`, `GRANT`, `REVOKE`, `VACUUM`.
2. **Table Whitelist**:
   - Queries can only access verified lakehouse tables registered as views:
     - `master_records` (Silver transaction grain)
     - `fx_rates` (Silver daily currency exchange rates)
     - `mart_monthly_pnl` (Gold aggregated P&L)
     - `mart_expense_summary` (Gold category summary)
3. **Execution Guardrails**:
   - Automatic execution timeout and row limit guards.
   - Pure Python DuckDB bindings (`.fetchall()`) ensure safe, pandas-free memory footprint.

---

## 6. Observability & Explainability

When `explain: true` is passed in `QueryRequest`, the agent exposes its complete internal reasoning trajectory:

```json
{
  "query": "What was my net cashflow in January 2026?",
  "answer": "In January 2026, your net cashflow was $1,250,000 COP...",
  "sql": "SELECT net_cashflow_cop FROM mart_monthly_pnl WHERE month = '2026-01'",
  "thought_steps": [
    {
      "step_number": 1,
      "tool": "search_data_catalog",
      "tool_input": {"query": "net cashflow"},
      "tool_output": "[LookML Measure] net_cashflow_cop ...",
      "thought": "Agent decided to invoke tool 'search_data_catalog'..."
    },
    {
      "step_number": 2,
      "tool": "run_sql_query",
      "tool_input": {"sql": "SELECT ..."},
      "tool_output": "{\"columns\": [\"net_cashflow_cop\"], ...}",
      "thought": "Agent decided to invoke tool 'run_sql_query'..."
    }
  ],
  "execution_time_ms": 1420.5
}
```
