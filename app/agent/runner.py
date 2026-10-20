import time
import logging
import json
from typing import Dict, Any, List, Optional
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage, ToolMessage
from langchain_core.tools import tool
from app.core.config import settings
from app.schemas.query import QueryRequest, QueryResponse, AgentThoughtStep
from app.agent.tools.duckdb_tool import duckdb_tool
from app.agent.tools.rag_tool import search_data_catalog as raw_search_catalog
from app.agent.prompts import AGENT_SYSTEM_PROMPT

logger = logging.getLogger(__name__)


# -------------------------------------------------------------------------
# First-Class LangChain Tools for Autonomous Agent
# -------------------------------------------------------------------------

@tool
def search_data_catalog(query: str) -> str:
    """Search the LookML semantic catalog and lakehouse schema for business metric definitions,
    calculation formulas, dimension names, and table structures. Call this whenever you need to find
    how financial metrics (like total spend, net cashflow, savings rate, burn rate) are defined or what
    tables and columns exist.
    """
    return raw_search_catalog(query, top_k=5)


@tool
def run_sql_query(sql: str) -> str:
    """Execute a safe read-only SQL query against local DuckDB lakehouse tables
    (master_records, fx_rates, mart_monthly_pnl, mart_expense_summary).
    Returns columns, row count, and sample data rows.
    """
    try:
        res = duckdb_tool.run_query(sql)
        row_count = res.get("row_count", 0)
        columns = res.get("columns", [])
        data = res.get("data", [])
        if row_count == 0:
            return f"Query executed successfully ({res.get('execution_time_ms', 0)}ms) but returned 0 rows. Columns: {columns}."
        return json.dumps({
            "columns": columns,
            "row_count": row_count,
            "execution_time_ms": res.get("execution_time_ms", 0),
            "rows": data[:15]
        }, default=str)
    except Exception as e:
        return f"DuckDB Execution Error: {str(e)}"


# -------------------------------------------------------------------------
# Autonomous ReAct Agent Runner
# -------------------------------------------------------------------------

class AgentRunner:
    """Autonomous tool-calling ReAct agent orchestrating LookML catalog RAG,
    multi-step LLM reasoning, conversational session memory, and safe DuckDB analytical execution.
    """

    def __init__(self):
        self.api_key = settings.GEMINI_API_KEY
        self.model_name = settings.GEMINI_MODEL
        self.llm = None
        self.llm_with_tools = None
        self.init_error = None
        self.sessions: Dict[str, List[Any]] = {}
        self._init_llm()

    def _init_llm(self):
        """Initialize LangChain ChatGoogleGenerativeAI and bind tools if API key is provided."""
        self.api_key = settings.GEMINI_API_KEY
        self.model_name = settings.GEMINI_MODEL
        self.init_error = None

        if self.api_key and self.api_key.strip() not in ("", "your_gemini_api_key_here"):
            try:
                from langchain_google_genai import ChatGoogleGenerativeAI
                self.llm = ChatGoogleGenerativeAI(
                    model=self.model_name,
                    google_api_key=self.api_key,
                    temperature=0.1
                )
                self.llm_with_tools = self.llm.bind_tools([search_data_catalog, run_sql_query])
                logger.info("Initialized Gemini autonomous ReAct agent with model %s and bound tools", self.model_name)
            except Exception as e:
                self.init_error = str(e)
                self.llm = None
                self.llm_with_tools = None
                logger.warning("Failed to initialize LangChain Gemini client: %s", e)
        else:
            self.llm = None
            self.llm_with_tools = None
            logger.info("No GEMINI_API_KEY configured. Autonomous tool-calling agent requires a valid key.")

    def run_query(self, request: QueryRequest) -> QueryResponse:
        """Execute autonomous ReAct loop:
        1. Check API key presence.
        2. LLM autonomously invokes search_data_catalog and/or run_sql_query.
        3. If DuckDB returns SQL error, LLM self-corrects.
        4. LLM synthesizes grounded final financial answer.
        """
        start_time = time.time()

        # Re-check API key in case settings were updated dynamically
        if self.llm_with_tools is None:
            self._init_llm()

        if not self.api_key or self.api_key.strip() in ("", "your_gemini_api_key_here"):
            raise RuntimeError(
                "GEMINI_API_KEY is not configured. An active Google Gemini API key is required "
                "to execute the autonomous tool-calling agent loop. Please configure GEMINI_API_KEY in your .env file."
            )

        if self.llm_with_tools is None:
            raise RuntimeError(
                f"Gemini autonomous agent failed to initialize: {self.init_error or 'LLM client is not ready'}. "
                "Please verify your GEMINI_API_KEY and network connectivity."
            )

        thought_steps: List[AgentThoughtStep] = []
        last_sql_executed: Optional[str] = None
        last_sql_data: List[Dict[str, Any]] = []

        session_id = request.session_id.strip() if (request.session_id and request.session_id.strip()) else None

        # Multi-turn session memory retrieval
        if session_id and session_id in self.sessions:
            messages: List[Any] = self.sessions[session_id]
            messages.append(HumanMessage(content=request.query))
        else:
            messages = [
                SystemMessage(content=AGENT_SYSTEM_PROMPT),
                HumanMessage(content=request.query)
            ]

        step_counter = 1
        max_iterations = 8

        for _ in range(max_iterations):
            ai_msg = self.llm_with_tools.invoke(messages)
            messages.append(ai_msg)

            # If no tool calls were requested, the model has completed its final answer
            if not ai_msg.tool_calls:
                if isinstance(ai_msg.content, str):
                    answer = ai_msg.content
                elif isinstance(ai_msg.content, list):
                    text_parts = [
                        item.get("text", "") if isinstance(item, dict) else str(item)
                        for item in ai_msg.content
                    ]
                    answer = "".join(text_parts).strip()
                else:
                    answer = str(ai_msg.content).strip()

                # Update session history with sliding window
                if session_id:
                    if len(messages) > 15:
                        messages = [messages[0]] + messages[-14:]
                    self.sessions[session_id] = messages

                duration_ms = round((time.time() - start_time) * 1000, 2)
                return QueryResponse(
                    query=request.query,
                    answer=answer,
                    sql=last_sql_executed,
                    data=last_sql_data if request.explain else [],
                    thought_steps=thought_steps if request.explain else [],
                    execution_time_ms=duration_ms
                )

            # Process each autonomous tool invocation
            for tc in ai_msg.tool_calls:
                tool_name = tc.get("name")
                tool_args = tc.get("args") or {}
                tool_id = tc.get("id")

                thought_text = f"Agent decided to invoke tool '{tool_name}' with parameters: {tool_args}"

                if tool_name == "search_data_catalog":
                    tool_output = search_data_catalog.invoke(tool_args)
                    thought_steps.append(
                        AgentThoughtStep(
                            step_number=step_counter,
                            tool="search_data_catalog",
                            tool_input=tool_args,
                            tool_output=tool_output,
                            thought=thought_text
                        )
                    )
                elif tool_name == "run_sql_query":
                    sql_arg = tool_args.get("sql", "")
                    last_sql_executed = sql_arg
                    try:
                        sql_res = duckdb_tool.run_query(sql_arg)
                        last_sql_data = sql_res.get("data", [])
                        tool_output = json.dumps({
                            "columns": sql_res.get("columns", []),
                            "row_count": sql_res.get("row_count", 0),
                            "execution_time_ms": sql_res.get("execution_time_ms", 0),
                            "rows": last_sql_data[:15]
                        }, default=str)
                    except Exception as e:
                        tool_output = f"DuckDB Execution Error: {str(e)}"

                    thought_steps.append(
                        AgentThoughtStep(
                            step_number=step_counter,
                            tool="run_sql_query",
                            tool_input=tool_args,
                            tool_output=tool_output,
                            thought=thought_text
                        )
                    )
                else:
                    tool_output = f"Error: Unknown tool '{tool_name}'."
                    thought_steps.append(
                        AgentThoughtStep(
                            step_number=step_counter,
                            tool=tool_name or "unknown",
                            tool_input=tool_args,
                            tool_output=tool_output,
                            thought=f"Attempted unknown tool '{tool_name}'"
                        )
                    )

                step_counter += 1
                messages.append(ToolMessage(content=str(tool_output), tool_call_id=tool_id))

        # If max iterations reached, invoke LLM once more to force synthesis from gathered context
        try:
            messages.append(HumanMessage(content="Please synthesize your final analytical answer now based on the query results retrieved above. Do not call any more tools."))
            final_msg = self.llm.invoke(messages)
            if isinstance(final_msg.content, str):
                final_answer = final_msg.content.strip()
            elif isinstance(final_msg.content, list):
                final_answer = "".join([part.get("text", "") if isinstance(part, dict) else str(part) for part in final_msg.content]).strip()
            else:
                final_answer = str(final_msg.content).strip()
        except Exception as e:
            final_answer = f"Agent reached reasoning iteration limit. Last SQL executed: {last_sql_executed}"

        # Update session history with sliding window
        if session_id:
            messages.append(AIMessage(content=final_answer))
            if len(messages) > 15:
                messages = [messages[0]] + messages[-14:]
            self.sessions[session_id] = messages

        duration_ms = round((time.time() - start_time) * 1000, 2)
        return QueryResponse(
            query=request.query,
            answer=final_answer,
            sql=last_sql_executed,
            data=last_sql_data if request.explain else [],
            thought_steps=thought_steps if request.explain else [],
            execution_time_ms=duration_ms
        )

    def clear_session(self, session_id: str) -> bool:
        """Clear conversation history for a given session ID."""
        if session_id in self.sessions:
            del self.sessions[session_id]
            return True
        return False


agent_runner = AgentRunner()
