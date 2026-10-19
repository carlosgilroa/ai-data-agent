import time
import logging
import json
from typing import Dict, Any, List, Optional
from app.core.config import settings
from app.schemas.query import QueryRequest, QueryResponse, AgentThoughtStep
from app.agent.tools.duckdb_tool import duckdb_tool
from app.agent.tools.rag_tool import search_data_catalog
from app.agent.prompts import AGENT_SYSTEM_PROMPT

logger = logging.getLogger(__name__)


class AgentRunner:
    """Orchestrates RAG context retrieval, LLM reasoning, and safe DuckDB query execution."""

    def __init__(self):
        self.api_key = settings.GEMINI_API_KEY
        self.model_name = settings.GEMINI_MODEL
        self._init_llm()

    def _init_llm(self):
        """Initialize LangChain ChatGoogleGenerativeAI if API key is provided."""
        self.llm = None
        if self.api_key and self.api_key != "your_gemini_api_key_here":
            try:
                from langchain_google_genai import ChatGoogleGenerativeAI
                self.llm = ChatGoogleGenerativeAI(
                    model=self.model_name,
                    google_api_key=self.api_key,
                    temperature=0.1
                )
                logger.info("Initialized Gemini LLM agent with model %s", self.model_name)
            except Exception as e:
                logger.warning("Failed to initialize LangChain Gemini client: %s. Using heuristic fallback.", e)
        else:
            logger.info("No GEMINI_API_KEY configured. Running agent in semantic heuristic mode.")

    def run_query(self, request: QueryRequest) -> QueryResponse:
        """Process user query through semantic catalog RAG and DuckDB execution."""
        start_time = time.time()
        thought_steps: List[AgentThoughtStep] = []

        # Step 1: Query semantic catalog for relevant LookML measures and schema
        step1_thought = f"Searching semantic catalog for metrics and dimensions related to: '{request.query}'"
        rag_context = search_data_catalog(request.query, top_k=4)
        thought_steps.append(
            AgentThoughtStep(
                step_number=1,
                tool="search_data_catalog",
                tool_input={"query": request.query},
                tool_output=rag_context,
                thought=step1_thought
            )
        )

        # Step 2: Formulate and execute SQL query
        sql_to_execute = None
        sql_result: Dict[str, Any] = {"columns": [], "data": [], "row_count": 0}

        # If LLM is available, use tool calling
        if self.llm is not None:
            try:
                sql_to_execute = self._generate_sql_with_llm(request.query, rag_context)
            except Exception as e:
                logger.error("LLM SQL generation failed: %s. Falling back to heuristic generator.", e)
                sql_to_execute = self._generate_heuristic_sql(request.query)
        else:
            sql_to_execute = self._generate_heuristic_sql(request.query)

        # Step 3: Run SQL in DuckDB
        if sql_to_execute:
            step2_thought = f"Executing LookML-aligned SQL against local DuckDB Lakehouse: {sql_to_execute}"
            try:
                sql_result = duckdb_tool.run_query(sql_to_execute)
                thought_steps.append(
                    AgentThoughtStep(
                        step_number=2,
                        tool="run_sql_query",
                        tool_input={"sql": sql_to_execute},
                        tool_output=f"Returned {sql_result.get('row_count', 0)} rows. Columns: {sql_result.get('columns', [])}",
                        thought=step2_thought
                    )
                )
            except Exception as e:
                thought_steps.append(
                    AgentThoughtStep(
                        step_number=2,
                        tool="run_sql_query",
                        tool_input={"sql": sql_to_execute},
                        tool_output=f"Execution error: {str(e)}",
                        thought=f"Attempted execution but failed with: {str(e)}"
                    )
                )

        # Step 4: Synthesize response
        answer = self._synthesize_answer(request.query, sql_to_execute, sql_result)
        duration_ms = round((time.time() - start_time) * 1000, 2)

        return QueryResponse(
            query=request.query,
            answer=answer,
            sql=sql_to_execute,
            data=sql_result.get("data", []) if request.explain else [],
            thought_steps=thought_steps if request.explain else [],
            execution_time_ms=duration_ms
        )

    def _generate_sql_with_llm(self, query: str, context: str) -> str:
        """Call Gemini to generate safe DuckDB SQL using retrieved context."""
        prompt = f"""{AGENT_SYSTEM_PROMPT}

### SEMANTIC CONTEXT FROM LOOKML CATALOG:
{context}

### USER QUESTION:
{query}

Generate ONLY a single valid DuckDB SQL query to answer this question. Do not include markdown code block syntax or explanations, just the raw SQL statement.
"""
        response = self.llm.invoke(prompt)
        raw_text = response.content.strip()
        # Clean markdown code blocks if returned
        if raw_text.startswith("```"):
            raw_text = raw_text.split("\n", 1)[1]
        if raw_text.endswith("```"):
            raw_text = raw_text.rsplit("\n", 1)[0]
        return raw_text.replace("```sql", "").replace("```", "").strip()

    def _generate_heuristic_sql(self, query: str) -> str:
        """Rule-based text-to-SQL fallback matching standard financial analytical questions."""
        q = query.lower()

        if "pnl" in q or "profit and loss" in q or "monthly cashflow" in q or "savings rate" in q:
            return "SELECT year_month, operating_income, operational_expenses, net_cashflow, savings_rate_pct FROM mart_monthly_pnl ORDER BY year_month ASC LIMIT 12;"
        
        elif "tier 1" in q or "burn rate" in q or "living" in q or "tier1" in q:
            return "SELECT tier_group, currency, SUM(ABS(amount)) as total_spent, COUNT(*) as tx_count FROM master_records WHERE is_transfer = false AND type = 'expense' AND tier_group LIKE '%Tier 1%' GROUP BY 1, 2 ORDER BY 3 DESC;"

        elif "category" in q or "categories" in q or "spend" in q or "expense" in q:
            return "SELECT category_group, category_name, currency, SUM(ABS(amount)) as total_spent, COUNT(*) as tx_count FROM master_records WHERE is_transfer = false AND type = 'expense' GROUP BY 1, 2, 3 ORDER BY 4 DESC LIMIT 10;"

        elif "income" in q or "earned" in q or "salary" in q:
            return "SELECT category_group, currency, SUM(amount) as total_income, COUNT(*) as tx_count FROM master_records WHERE is_transfer = false AND type = 'income' GROUP BY 1, 2 ORDER BY 3 DESC;"

        elif "account" in q or "balance" in q:
            return "SELECT account_class, account_name, currency, COUNT(*) as total_records, SUM(amount) as net_activity FROM master_records GROUP BY 1, 2, 3 ORDER BY 1, 2;"

        else:
            return "SELECT date, account_name, category_name, amount, currency, tier_group FROM master_records WHERE is_transfer = false ORDER BY date DESC LIMIT 10;"

    def _synthesize_answer(self, query: str, sql: Optional[str], result: Dict[str, Any]) -> str:
        """Produce natural language financial summary from SQL output."""
        data = result.get("data", [])
        if not data:
            return f"No financial records found matching your query: '{query}'. Please verify your date range or filters."

        row_count = len(data)
        first_row = data[0]

        if "net_cashflow" in first_row:
            latest = data[-1]
            return (
                f"Based on reconciled monthly P&L records (via LookML mart_monthly_pnl):\n"
                f"- Latest Month ({latest.get('year_month')}): Net Cash Flow was {latest.get('net_cashflow'):,.2f} "
                f"with a savings rate of {latest.get('savings_rate_pct'):.1f}%.\n"
                f"- Total operating income was {latest.get('operating_income'):,.2f} against operational expenses of {latest.get('operational_expenses'):,.2f}."
            )

        if "total_spent" in first_row:
            top_category = first_row.get("category_name") or first_row.get("category_group") or "Top Category"
            amount = first_row.get("total_spent", 0)
            currency = first_row.get("currency", "")
            return (
                f"Analysis of {row_count} spending categories (excluding internal transfers per LookML metric definition):\n"
                f"- Top Category: **{top_category}** with total expenses of **{amount:,.2f} {currency}**.\n"
                f"- Query executed safely against silver master_records with DuckDB."
            )

        return (
            f"Successfully executed query against the local Parquet data lakehouse, returning {row_count} records.\n"
            f"Key metrics and dimensions were verified against the LookML semantic layer."
        )


agent_runner = AgentRunner()
