import time
import logging
import json
from typing import Dict, Any, List, Optional, Tuple
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage, ToolMessage
from langchain_core.tools import tool
from app.core.config import settings
from app.schemas.query import QueryRequest, QueryResponse, AgentThoughtStep
from app.agent.tools.duckdb_tool import duckdb_tool
from app.agent.tools.rag_tool import search_data_catalog as raw_search_catalog
from app.agent.prompts import AGENT_SYSTEM_PROMPT
from app.agent.router import intent_router

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
    return raw_search_catalog(
        query,
        top_k=settings.RAG_TOP_K,
        min_similarity=settings.RAG_MIN_SIMILARITY,
    )


@tool
def run_sql_query(sql: str) -> str:
    """Execute a safe read-only SQL query against local DuckDB lakehouse tables and LookML PDTs
    (master_records, dim_accounts, dim_categories, dim_labels, fx_rates, mart_monthly_pnl,
    mart_expense_summary, pdt_category_mom_velocity).
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


TOOL_REGISTRY = {
    "search_data_catalog": search_data_catalog,
    "run_sql_query": run_sql_query,
}


# -------------------------------------------------------------------------
# Autonomous ReAct Agent Runner with Two-Step Intent Routing
# -------------------------------------------------------------------------

class AgentRunner:
    """Autonomous tool-calling ReAct agent orchestrating two-step intent routing,
    LookML catalog RAG, multi-step LLM reasoning, conversational session memory,
    and safe DuckDB analytical execution.
    """

    def __init__(self):
        self.api_key = settings.GEMINI_API_KEY
        self.model_name = settings.GEMINI_MODEL
        self.llm = None
        self.llm_with_tools = None
        self.init_error = None
        self.sessions: Dict[str, List[Any]] = {}
        self._model_cache: Dict[str, Any] = {}
        self._init_llm()

    def _init_llm(self):
        """Initialize default LangChain ChatGoogleGenerativeAI and bind tools if API key is provided."""
        self.api_key = settings.GEMINI_API_KEY
        self.model_name = settings.GEMINI_MODEL
        self.init_error = None
        self._model_cache.clear()

        if self.api_key and self.api_key.strip() not in ("", "your_gemini_api_key_here"):
            try:
                self.llm, self.llm_with_tools = self._get_or_create_llm(
                    self.model_name, ["search_data_catalog", "run_sql_query"]
                )
                logger.info("Initialized Gemini autonomous ReAct agent with default model %s", self.model_name)
            except Exception as e:
                self.init_error = str(e)
                self.llm = None
                self.llm_with_tools = None
                logger.warning("Failed to initialize LangChain Gemini client: %s", e)
        else:
            self.llm = None
            self.llm_with_tools = None
            logger.info("No GEMINI_API_KEY configured. Autonomous tool-calling agent requires a valid key.")

    def _get_or_create_llm(self, model_name: str, allowed_tool_names: List[str]) -> Tuple[Any, Any]:
        """Retrieve or create a cached ChatGoogleGenerativeAI instance and bind requested tools."""
        from langchain_google_genai import ChatGoogleGenerativeAI

        if model_name not in self._model_cache:
            llm_kwargs: Dict[str, Any] = {
                "model": model_name,
                "google_api_key": self.api_key,
            }
            if "3.5-flash-lite" not in model_name:
                llm_kwargs["temperature"] = 0.1
            self._model_cache[model_name] = ChatGoogleGenerativeAI(**llm_kwargs)
        base_llm = self._model_cache[model_name]
        tools_to_bind = [TOOL_REGISTRY[name] for name in allowed_tool_names if name in TOOL_REGISTRY]
        bound_llm = base_llm.bind_tools(tools_to_bind) if tools_to_bind else base_llm
        return base_llm, bound_llm

    @staticmethod
    def _extract_text(content: Any) -> str:
        """Normalize LangChain AIMessage content into a clean string."""
        if isinstance(content, str):
            return content.strip()
        if isinstance(content, list):
            parts = [
                item.get("text", "") if isinstance(item, dict) else str(item)
                for item in content
            ]
            return "".join(parts).strip()
        return str(content).strip()

    def run_query(self, request: QueryRequest) -> QueryResponse:
        """Execute two-step routed ReAct loop:
        1. Check API key presence.
        2. Classify user intent via IntentRouter (gemini-3.1-flash-lite).
        3. Route to target model & iteration budget (0, 1, 3, or 6 steps).
        4. Synthesize grounded final answer.
        """
        start_time = time.time()

        # Re-check API key in case settings were updated dynamically
        if self.llm_with_tools is None and settings.GEMINI_API_KEY == self.api_key:
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
        has_session_context = bool(session_id and session_id in self.sessions)

        # -----------------------------------------------------------------
        # Step 1: Two-Step Intent Router Classification (gemini-3.1-flash-lite)
        # -----------------------------------------------------------------
        decision = intent_router.classify(request.query, has_session_context=has_session_context)
        target_model, max_iterations, allowed_tool_names = intent_router.get_route_config(decision.intent)

        step_counter = 1
        thought_steps.append(
            AgentThoughtStep(
                step_number=step_counter,
                tool="intent_router",
                tool_input={
                    "query": request.query,
                    "router_model": settings.GEMINI_ROUTER_MODEL,
                },
                tool_output={
                    "intent": decision.intent,
                    "target_model": target_model,
                    "max_iterations": max_iterations,
                    "allowed_tools": allowed_tool_names,
                    "reasoning": decision.reasoning,
                },
                thought=(
                    f"Router ({settings.GEMINI_ROUTER_MODEL}) classified intent as '{decision.intent}' "
                    f"({decision.reasoning}) -> routed to '{target_model}' with max_iterations={max_iterations}"
                ),
            )
        )
        step_counter += 1

        base_llm, routed_llm_with_tools = self._get_or_create_llm(target_model, allowed_tool_names)

        # Multi-turn session memory retrieval
        if session_id and session_id in self.sessions:
            messages: List[Any] = self.sessions[session_id]
            messages.append(HumanMessage(content=request.query))
        else:
            messages = [
                SystemMessage(content=AGENT_SYSTEM_PROMPT),
                HumanMessage(content=request.query)
            ]

        # -----------------------------------------------------------------
        # Step 2A: Out-of-Scope Handling (0 tool steps, separate LLM call)
        # -----------------------------------------------------------------
        if decision.intent == "out_of_scope" or max_iterations == 0:
            oos_messages = [
                SystemMessage(
                    content=(
                        "You are a specialized personal finance and data lakehouse AI assistant. "
                        "Politely decline the user's question because it is outside the scope of "
                        "personal finance, accounting, and lakehouse analytics, and briefly remind "
                        "them what financial questions you can help answer."
                    )
                ),
                HumanMessage(content=request.query),
            ]
            ai_msg = base_llm.invoke(oos_messages)
            answer = self._extract_text(ai_msg.content)

            if session_id:
                messages.append(AIMessage(content=answer))
                if len(messages) > 15:
                    messages = [messages[0]] + messages[-14:]
                self.sessions[session_id] = messages

            duration_ms = round((time.time() - start_time) * 1000, 2)
            return QueryResponse(
                query=request.query,
                intent=decision.intent,
                model_used=target_model,
                answer=answer,
                sql=None,
                data=[],
                thought_steps=thought_steps if request.explain else [],
                execution_time_ms=duration_ms,
            )

        # -----------------------------------------------------------------
        # Step 2B: Tool-Calling ReAct Loop (capabilities=1, simple=3, complex=6)
        # -----------------------------------------------------------------
        for _ in range(max_iterations):
            ai_msg = routed_llm_with_tools.invoke(messages)
            messages.append(ai_msg)

            # If no tool calls were requested, the model has completed its final answer
            if not ai_msg.tool_calls:
                answer = self._extract_text(ai_msg.content)

                # Update session history with sliding window
                if session_id:
                    if len(messages) > 15:
                        messages = [messages[0]] + messages[-14:]
                    self.sessions[session_id] = messages

                duration_ms = round((time.time() - start_time) * 1000, 2)
                return QueryResponse(
                    query=request.query,
                    intent=decision.intent,
                    model_used=target_model,
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

                thought_text = f"Agent ({target_model}) decided to invoke tool '{tool_name}' with parameters: {tool_args}"

                if tool_name == "search_data_catalog" and "search_data_catalog" in allowed_tool_names:
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
                elif tool_name == "run_sql_query" and "run_sql_query" in allowed_tool_names:
                    sql_arg = tool_args.get("sql", "")
                    try:
                        sql_res = duckdb_tool.run_query(sql_arg)
                        last_sql_executed = sql_arg
                        last_sql_data = sql_res.get("data", [])
                        total_rows = sql_res.get("row_count", 0)
                        preview_limit = 15
                        preview_rows = min(total_rows, preview_limit)

                        payload = {
                            "columns": sql_res.get("columns", []),
                            "total_rows_returned": total_rows,
                            "preview_rows": preview_rows,
                            "execution_time_ms": sql_res.get("execution_time_ms", 0),
                            "rows": last_sql_data[:preview_limit],
                        }
                        if total_rows > preview_limit:
                            payload["notice"] = (
                                f"Query returned {total_rows} total rows. Showing first {preview_limit} preview rows. "
                                "For complete category totals or summaries, execute an aggregated SQL query (GROUP BY and SUM/COUNT)."
                            )
                        tool_output = json.dumps(payload, default=str)
                    except Exception as e:
                        if last_sql_executed is None:
                            last_sql_executed = sql_arg
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
                    tool_output = f"Error: Tool '{tool_name}' is not permitted for intent '{decision.intent}'."
                    thought_steps.append(
                        AgentThoughtStep(
                            step_number=step_counter,
                            tool=tool_name or "unknown",
                            tool_input=tool_args,
                            tool_output=tool_output,
                            thought=f"Attempted disallowed or unknown tool '{tool_name}'"
                        )
                    )

                step_counter += 1
                messages.append(ToolMessage(content=str(tool_output), tool_call_id=tool_id))

        # If max iterations reached, invoke LLM with tool_choice='none' (preserving tool declarations for Gemini history) to force synthesis
        try:
            messages.append(
                HumanMessage(
                    content="Please synthesize your final analytical answer now based on the context and results retrieved above. Do not call any more tools."
                )
            )
            tools_to_bind = [TOOL_REGISTRY[name] for name in allowed_tool_names if name in TOOL_REGISTRY]
            synthesis_llm = (
                base_llm.bind_tools(tools_to_bind, tool_choice="none")
                if tools_to_bind
                else base_llm
            )
            final_msg = synthesis_llm.invoke(messages)
            final_answer = self._extract_text(final_msg.content)
            if not final_answer:
                final_msg_fallback = routed_llm_with_tools.invoke(messages)
                final_answer = self._extract_text(final_msg_fallback.content)
        except Exception:
            final_answer = f"Agent reached reasoning iteration limit ({max_iterations}). Last SQL executed: {last_sql_executed}"

        # Update session history with sliding window
        if session_id:
            messages.append(AIMessage(content=final_answer))
            if len(messages) > 15:
                messages = [messages[0]] + messages[-14:]
            self.sessions[session_id] = messages

        duration_ms = round((time.time() - start_time) * 1000, 2)
        return QueryResponse(
            query=request.query,
            intent=decision.intent,
            model_used=target_model,
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
