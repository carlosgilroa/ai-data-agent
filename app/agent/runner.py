import time
import logging
import json
from typing import Dict, Any, List, Optional, Tuple
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage, ToolMessage
from langchain_core.tools import tool
from langchain_core.utils.function_calling import convert_to_openai_tool
from pydantic import BaseModel, Field as PydanticField
from app.core.config import settings
from app.schemas.query import QueryRequest, QueryResponse, AgentThoughtStep
from app.agent.tools.duckdb_tool import duckdb_tool
from app.agent.tools.rag_tool import (
    get_table_schema as raw_get_table_schema,
    Retrieval,
    retrieve,
    search_semantic_catalog as raw_search_semantic_catalog,
)
from app.agent.prompts import AGENT_SYSTEM_PROMPT, DEFAULT_PERSONA, RAW_SQL_DISABLED, RAW_SQL_ENABLED
from app.agent.router import data_description, intent_router
from app.agent.usage import UsageCallback, start_tracking, summarize
from app.semantic.compiler import SemanticQuery, SemanticQueryError
from app.semantic.service import semantic_layer

logger = logging.getLogger(__name__)

TOOL_PREVIEW_ROWS = 50


# -------------------------------------------------------------------------
# LangChain tools: the model picks LookML fields, the compiler writes the SQL
# -------------------------------------------------------------------------

@tool
def search_semantic_catalog(query: str) -> str:
    """Search the descriptions of LookML measures and dimensions for a phrase in plain words.
    Returns view.field names and the explores they are in.
    """
    return raw_search_semantic_catalog(query)


@tool
def get_explores() -> str:
    """List the LookML explores (the queryable data models) with what each one answers."""
    return semantic_layer.explore_overview()


@tool
def get_dimensions(explore: str) -> str:
    """List the visible dimensions of an explore (fields to group by or filter on), with descriptions."""
    try:
        return semantic_layer.describe_fields(explore, "dimension")
    except SemanticQueryError as e:
        return f"Error: {e}"


@tool
def get_measures(explore: str) -> str:
    """List the visible measures of an explore (the metrics it can calculate), with descriptions."""
    try:
        return semantic_layer.describe_fields(explore, "measure")
    except SemanticQueryError as e:
        return f"Error: {e}"


@tool
def get_field_values(explore: str, field: Optional[str] = None, search: Optional[str] = None) -> str:
    """Look up filter values, like Looker's filter suggestions.
    With `field` (view.field): up to 100 distinct values of that dimension, only those containing `search` if given.
    With only `search`: searches every searchable text dimension of the explore and returns which ones contain it.
    "truncated": true means more values exist than were returned; search for the value instead of listing.
    """
    try:
        if field:
            return json.dumps(semantic_layer.field_values(explore, field, search=search), default=str)
        if search:
            matches = semantic_layer.search_values(explore, search)
            if not matches:
                return f"No text field of {explore} contains '{search}'. Try another word or a broader term."
            return json.dumps({"search": search, "matches": matches[:5]}, default=str)
        return "Error: pass a field to list its values, or a search term to find which field contains it."
    except SemanticQueryError as e:
        return f"Error: {e}"


# Filters go to the model as a list of field/value objects, not a {field: value} map: Gemini's function
# calling garbled free-form object keys (dots became underscores, keys were invented).
class QueryFilter(BaseModel):
    """One filter: a field and the filter expression to apply to it."""

    field: str = PydanticField(..., description="Exact field name as view.field, as listed in the context or a tool result")
    value: str = PydanticField(..., description="Looker filter expression, e.g. <value>, -<value>, %<text>%, >1000, last month")


def filters_to_dict(filters: Any) -> Dict[str, str]:
    """The compiler's {view.field: expression} map from the tool's list of field/value filters.
    A dict passes through unchanged, so older callers and session examples keep working.
    """
    if not filters:
        return {}
    if isinstance(filters, dict):
        return {str(k): str(v) for k, v in filters.items()}
    result: Dict[str, str] = {}
    for item in filters:
        if isinstance(item, QueryFilter):
            item = item.model_dump()
        if not isinstance(item, dict) or not item.get("field") or item.get("value") is None:
            raise ValueError(f'Each filter must be {{"field": "view.field", "value": "expression"}}, got {item!r}')
        result[str(item["field"])] = str(item["value"])
    return result


def filters_to_list(filters: Dict[str, str]) -> List[Dict[str, str]]:
    return [{"field": f, "value": v} for f, v in filters.items()]


def _tool_args(semantic_query: Dict[str, Any]) -> Dict[str, Any]:
    """A semantic query written the way the `query` tool takes it (filters as a field/value list)."""
    args = dict(semantic_query)
    if isinstance(args.get("filters"), dict):
        args["filters"] = filters_to_list(args["filters"])
    return args


@tool
def query(
    explore: str,
    fields: List[str],
    filters: Optional[List[QueryFilter]] = None,
    sorts: Optional[List[str]] = None,
    limit: int = 500,
) -> str:
    """Run a query against the LookML semantic layer. The compiler writes the SQL from the official
    LookML definitions, so metrics are always computed the governed way.
    explore: explore name, as listed.
    fields: dimensions and measures as view.field, e.g. ['<view>.<dimension>', '<view>.<measure>'].
    filters: list of {field, value}, e.g. [{'field': '<view>.<time field>_month', 'value': 'last month'}].
    sorts: e.g. ['<view>.<measure> desc']. limit: maximum rows.
    """
    try:
        result = semantic_layer.run(
            SemanticQuery(explore=explore, fields=fields, filters=filters_to_dict(filters), sorts=sorts or [], limit=limit)
        )
    except (SemanticQueryError, ValueError) as e:
        return f"Error: {e}"
    return _semantic_tool_output(result)


@tool
def get_table_schema(table_name: str) -> str:
    """Fallback only: column names and types of a raw table, for read-only SQL when no explore fits."""
    return raw_get_table_schema(table_name)


@tool
def run_sql_query(sql: str) -> str:
    """Fallback only: run read-only SQL on the raw tables when no explore can answer the question.
    Results are not governed metrics.
    """
    try:
        res = duckdb_tool.run_query(sql)
        return json.dumps({
            "columns": res.get("columns", []),
            "row_count": res.get("row_count", 0),
            "rows": res.get("data", [])[:TOOL_PREVIEW_ROWS],
        }, default=str)
    except Exception as e:
        return f"DuckDB Execution Error: {str(e)}"


def _semantic_tool_output(result: Dict[str, Any]) -> str:
    payload: Dict[str, Any] = {
        "fields": [f"{f['name']}: {f['description']}" for f in result["fields"]],
        "row_count": result["row_count"],
        "rows": result["rows"][:TOOL_PREVIEW_ROWS],
    }
    if result["row_count"] > TOOL_PREVIEW_ROWS:
        payload["notice"] = f"Showing the first {TOOL_PREVIEW_ROWS} of {result['row_count']} rows."
    if result["notes"]:
        payload["notes"] = result["notes"]
    return json.dumps(payload, default=str)


def _format_retrieval(retrieval: Retrieval, decision: Any, previous_query: Optional[Dict[str, Any]] = None) -> str:
    """The per-question context, in LookML terms: explores, fields, resolved filter values, time,
    glossary and golden queries. Only what retrieval found is shown, so the prompt stays small.
    For a follow-up, the previous query's explore and filters are listed as already resolved.
    """
    model = semantic_layer.model
    lines = ["### Context for this question"]
    lines.append("Explores that fit best:")
    for name in retrieval.explores:
        explore = model.explores.get(name)
        if explore is None:
            continue
        groups: Dict[str, List[str]] = {}
        for f in model.explore_fields(name):
            if f.is_time and f.timeframe != "raw":
                groups.setdefault(f"{f.view}.{f.group}", []).append(f.timeframe)
        lines.append(f"- {name}: {explore.label}. {explore.description}")
        if groups:
            lines.append("  Time fields: " + "; ".join(f"{g}_{{{', '.join(tfs)}}}" for g, tfs in groups.items()))
    if retrieval.measures:
        lines.append("Measures:")
        lines += [f"- {i.view}.{i.name} (in {', '.join(i.explores)}): {i.description}" for i in retrieval.measures]
    if retrieval.dimensions:
        lines.append("Dimensions:")
        lines += [f"- {i.view}.{i.name} (in {', '.join(i.explores)}): {i.description}" for i in retrieval.dimensions]
    if retrieval.filter_values:
        lines.append("Filter values found:")
        for value in retrieval.filter_values:
            if not value["matches"]:
                lines.append(f'- "{value["value"]}": not found in any searchable field of these explores.')
                continue
            found = "; ".join(
                f"{m['field']} = {', '.join(repr(v) for v in m['values'])}" for m in value["matches"]
            )
            via = " (from the glossary)" if value["source"] == "glossary" else ""
            lines.append(f'- "{value["value"]}"{via}: {found}')
    if previous_query and previous_query.get("explore"):
        previous_filters = filters_to_dict(previous_query.get("filters"))
        lines.append(
            f"Previous question's query (explore {previous_query['explore']}); its filter values are already "
            "resolved, keep them unless this question changes them:"
        )
        lines += [f"- {f} = {v!r}" for f, v in previous_filters.items()] or ["- no filters"]
    if decision.time_period:
        lines.append(f'Time period asked: "{decision.time_period}" (use it as the filter on the explore\'s time field).')
    if retrieval.glossary:
        lines.append("Glossary:")
        for entry in retrieval.glossary:
            text = f'- "{entry["term"]}": {entry["meaning"]}'
            if entry.get("field") and entry.get("value"):
                text += f' (filter {entry["field"]} = "{entry["value"]}")'
            lines.append(text)
    if retrieval.golden_queries:
        lines.append("Verified example queries:")
        for golden in retrieval.golden_queries:
            lines.append(f'- "{golden["question"]}" -> query({json.dumps(_tool_args(golden["query"]))})')
    return "\n".join(lines)


TOOL_REGISTRY = {
    "search_semantic_catalog": search_semantic_catalog,
    "get_explores": get_explores,
    "get_dimensions": get_dimensions,
    "get_measures": get_measures,
    "get_field_values": get_field_values,
    "query": query,
    "get_table_schema": get_table_schema,
    "run_sql_query": run_sql_query,
}

# What Gemini is sent for each tool. Bound as OpenAI-style function dicts rather than the tools themselves:
# converting a tool directly repeats its whole docstring as the description of its parameters object
# (about 2,000 characters of duplicates per call). Parameter descriptions, such as QueryFilter's, are kept.
TOOL_DECLARATIONS = {name: convert_to_openai_tool(t) for name, t in TOOL_REGISTRY.items()}


def tool_declarations(names: List[str]) -> List[Dict[str, Any]]:
    return [TOOL_DECLARATIONS[name] for name in names if name in TOOL_DECLARATIONS]


# -------------------------------------------------------------------------
# Autonomous ReAct Agent Runner with Two-Step Intent Routing
# -------------------------------------------------------------------------

class _AnswerTrace:
    """What the final answer is based on: the last successful semantic query, or the raw SQL fallback."""

    RAW_SQL_NOTE = "Answered with raw SQL, not with a governed LookML metric."

    def __init__(self):
        self.sql: Optional[str] = None
        self.semantic_query: Optional[Dict[str, Any]] = None
        self.fields: List[Dict[str, Any]] = []
        self.notes: List[str] = []
        self.data: List[Dict[str, Any]] = []
        self.governed: Optional[bool] = None
        self.llm_calls: List[Dict[str, Any]] = []

    def record_semantic(self, query: SemanticQuery, result: Dict[str, Any]) -> None:
        self.sql = result["sql"]
        self.semantic_query = query.model_dump()
        self.fields = result["fields"]
        self.notes = result["notes"]
        self.data = result["rows"]
        self.governed = True

    def record_sql(self, sql: str, rows: List[Dict[str, Any]]) -> None:
        self.sql = sql
        self.semantic_query = None
        self.fields = []
        self.notes = [self.RAW_SQL_NOTE]
        self.data = rows
        self.governed = False


class AgentRunner:
    """Tool-calling agent: intent routing, catalog retrieval, LookML semantic queries compiled to SQL,
    conversational session memory and a read-only raw SQL fallback.
    """

    def __init__(self):
        self.api_key = settings.GEMINI_API_KEY
        self.model_name = settings.GEMINI_MODEL
        self.llm = None
        self.llm_with_tools = None
        self.init_error = None
        self.sessions: Dict[str, List[Any]] = {}
        # Each session's last question and its semantic query, for the router to resolve follow-ups
        self.last_turns: Dict[str, Dict[str, Any]] = {}
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
                    self.model_name, list(TOOL_REGISTRY.keys())
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
                "callbacks": [UsageCallback(model_name)],
            }
            if "3.5-flash-lite" not in model_name:
                llm_kwargs["temperature"] = 0.1
            self._model_cache[model_name] = ChatGoogleGenerativeAI(**llm_kwargs)
        base_llm = self._model_cache[model_name]
        tools_to_bind = tool_declarations(allowed_tool_names)
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

    def _save_session_turn(
        self,
        session_id: Optional[str],
        user_query: str,
        assistant_answer: str,
        last_sql: Optional[str] = None,
        last_semantic_query: Optional[Dict[str, Any]] = None,
    ) -> None:
        """Persist clean conversational dialogue turns (HumanMessage, AIMessage) to session memory.

        Includes the semantic query (or the raw SQL) in the assistant turn for continuity,
        without preserving raw intermediate ToolMessages that cause Gemini turn-validation
        errors (400 INVALID_ARGUMENT).
        """
        if not session_id:
            return

        assistant_content = assistant_answer
        if last_semantic_query:
            # Lets a follow-up ("and in August?") reuse the previous query and change one part
            assistant_content += f"\n\nSemantic query used: {json.dumps(_tool_args(last_semantic_query))}"
        elif last_sql and "```sql" not in assistant_content:
            assistant_content += f"\n\n```sql\n-- Executed Lakehouse Query\n{last_sql}\n```"

        history = self.sessions.get(session_id, [])
        history.append(HumanMessage(content=user_query))
        history.append(AIMessage(content=assistant_content))

        # Sliding window: keep the last 10 conversation turns (20 messages)
        max_dialogue_messages = 20
        if len(history) > max_dialogue_messages:
            history = history[-max_dialogue_messages:]

        self.sessions[session_id] = history
        self.last_turns[session_id] = {"question": user_query, "semantic_query": last_semantic_query}

    def _system_prompt(
        self, decision, question: str, previous_query: Optional[Dict[str, Any]] = None
    ) -> Tuple[str, Retrieval]:
        """Base prompt plus the per-question context: dates, data coverage, what retrieval found and,
        for a follow-up, the previous query's explore and filters."""
        instructions = semantic_layer.instructions
        prompt = AGENT_SYSTEM_PROMPT.format(
            persona=instructions.persona.strip() or DEFAULT_PERSONA,
            raw_sql_rule=RAW_SQL_ENABLED if settings.ALLOW_RAW_SQL_FALLBACK else RAW_SQL_DISABLED,
        )
        first, last = duckdb_tool.get_data_coverage()
        context = f"\n### Runtime context\n- Today is {settings.today().isoformat()}."
        if first and last:
            context += f"\n- The data covers {first.isoformat()} to {last.isoformat()}."

        retrieval = retrieve(
            question,
            measure_phrase=" ".join(decision.measures),
            dimension_phrase=" ".join(decision.dimensions),
            filter_values=decision.filter_values,
        )
        context += "\n\n" + _format_retrieval(retrieval, decision, previous_query)
        return prompt + context, retrieval

    def run_query(self, request: QueryRequest) -> QueryResponse:
        """Route the question, let the model pick LookML fields, compile and run them, and synthesize an answer:
        1. Check the API key and reload the semantic model if the LookML changed.
        2. Classify intent and pick model, step budget and tools (IntentRouter).
        3. Retrieve explores, fields, resolved filter values and agent instructions into the prompt.
        4. Tool-calling loop: query (or the raw SQL fallback) until the model answers.
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

        semantic_layer.ensure_fresh()

        thought_steps: List[AgentThoughtStep] = []
        trace = _AnswerTrace()
        trace.llm_calls = start_tracking()

        session_id = request.session_id.strip() if (request.session_id and request.session_id.strip()) else None
        has_session_context = bool(session_id and session_id in self.sessions)

        # -----------------------------------------------------------------
        # Step 1: Intent routing
        # -----------------------------------------------------------------
        previous = self.last_turns.get(session_id) if has_session_context else None
        decision = intent_router.classify(request.query, has_session_context=has_session_context, previous=previous)
        # Retrieval searches the follow-up rewritten as a full question, so it finds what the history implies
        question = (decision.standalone_question or "").strip() or request.query
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
                    "standalone_question": decision.standalone_question,
                    "measures": decision.measures,
                    "dimensions": decision.dimensions,
                    "filter_values": decision.filter_values,
                    "time_period": decision.time_period,
                },
                thought=(
                    f"Router ({settings.GEMINI_ROUTER_MODEL}) classified intent as '{decision.intent}' "
                    f"({decision.reasoning}) -> routed to '{target_model}' with max_iterations={max_iterations}."
                ),
            )
        )
        step_counter += 1

        base_llm, routed_llm_with_tools = self._get_or_create_llm(target_model, allowed_tool_names)

        # -----------------------------------------------------------------
        # Step 2A: Out-of-Scope Handling (0 tool steps, separate LLM call)
        # -----------------------------------------------------------------
        if decision.intent == "out_of_scope" or max_iterations == 0:
            oos_messages = [
                SystemMessage(
                    content=(
                        f"{semantic_layer.instructions.persona.strip() or DEFAULT_PERSONA}\n\n"
                        "The user's question is outside the data you can answer about. Politely decline, and "
                        f"say briefly what kinds of questions you can answer. The data you cover: {data_description()}"
                    )
                ),
                HumanMessage(content=request.query),
            ]
            ai_msg = base_llm.invoke(oos_messages)
            answer = self._extract_text(ai_msg.content)
            self._save_session_turn(session_id, request.query, answer)
            return self._response(request, decision.intent, target_model, answer, trace, thought_steps, start_time)

        previous_query = previous.get("semantic_query") if previous else None
        system_prompt, retrieval = self._system_prompt(decision, question, previous_query)
        thought_steps.append(
            AgentThoughtStep(
                step_number=step_counter,
                tool="retrieval",
                tool_input={
                    "question": question,
                    "measures": decision.measures,
                    "dimensions": decision.dimensions,
                    "filter_values": decision.filter_values,
                },
                tool_output=retrieval.scored(),
                thought=(
                    "Chroma retrieved explores, fields, glossary and golden queries (cosine similarity scores; an "
                    "explore's score is its own similarity + its best measure + half its best dimension + 0.5 per "
                    "filter value it contains); filter values were looked up in the data."
                ),
            )
        )
        step_counter += 1
        messages: List[Any] = [SystemMessage(content=system_prompt)]
        if session_id and session_id in self.sessions:
            messages.extend(self.sessions[session_id])
        messages.append(HumanMessage(content=request.query))

        # -----------------------------------------------------------------
        # Step 2B: Tool-calling loop
        # -----------------------------------------------------------------
        for _ in range(max_iterations):
            ai_msg = routed_llm_with_tools.invoke(messages)
            messages.append(ai_msg)

            if not ai_msg.tool_calls:
                answer = self._extract_text(ai_msg.content)
                self._save_session_turn(session_id, request.query, answer, trace.sql, trace.semantic_query)
                return self._response(request, decision.intent, target_model, answer, trace, thought_steps, start_time)

            for tc in ai_msg.tool_calls:
                tool_name = tc.get("name")
                tool_args = tc.get("args") or {}
                tool_output = self._run_tool(tool_name, tool_args, allowed_tool_names, decision.intent, trace)
                thought_steps.append(
                    AgentThoughtStep(
                        step_number=step_counter,
                        tool=tool_name or "unknown",
                        tool_input=tool_args,
                        tool_output=tool_output,
                        thought=f"Agent ({target_model}) invoked '{tool_name}' with parameters: {tool_args}",
                    )
                )
                step_counter += 1
                messages.append(ToolMessage(content=str(tool_output), tool_call_id=tc.get("id")))

        # Step budget exhausted: force a final answer from what was retrieved
        try:
            messages.append(
                HumanMessage(
                    content=(
                        "Please synthesize your final answer now based strictly on the tool results above. "
                        "Do not call any more tools. If no query returned the numbers needed, say clearly that "
                        "the answer was not found. Never fabricate numbers."
                    )
                )
            )
            tools_to_bind = tool_declarations(allowed_tool_names)
            synthesis_llm = (
                base_llm.bind_tools(tools_to_bind, tool_choice="none")
                if tools_to_bind
                else base_llm
            )
            final_answer = self._extract_text(synthesis_llm.invoke(messages).content)
        except Exception:
            final_answer = ""

        if not final_answer:
            final_answer = (
                f"I couldn't finish the analysis within {max_iterations} steps. "
                "Please try asking a narrower question."
            )

        self._save_session_turn(session_id, request.query, final_answer, trace.sql, trace.semantic_query)
        return self._response(request, decision.intent, target_model, final_answer, trace, thought_steps, start_time)

    @staticmethod
    def _run_tool(name: str, args: Dict[str, Any], allowed: List[str], intent: str, trace: "_AnswerTrace") -> str:
        """Execute one tool call and record what an answer would be based on."""
        if name not in allowed or name not in TOOL_REGISTRY:
            return f"Error: Tool '{name}' is not permitted for intent '{intent}'."

        if name == "query":
            try:
                semantic_query = SemanticQuery(
                    explore=args.get("explore", ""),
                    fields=args.get("fields") or [],
                    filters=filters_to_dict(args.get("filters")),
                    sorts=args.get("sorts") or [],
                    limit=args.get("limit") or 500,
                )
                result = semantic_layer.run(semantic_query)
            except (SemanticQueryError, ValueError) as e:
                return f"Error: {e}"
            trace.record_semantic(semantic_query, result)
            return _semantic_tool_output(result)

        if name == "run_sql_query":
            sql = args.get("sql", "")
            output = run_sql_query.invoke(args)
            if not output.startswith("DuckDB Execution Error"):
                trace.record_sql(sql, json.loads(output).get("rows", []))
            return output

        return TOOL_REGISTRY[name].invoke(args)

    @staticmethod
    def _response(request, intent, model, answer, trace: "_AnswerTrace", steps, start_time) -> QueryResponse:
        return QueryResponse(
            query=request.query,
            intent=intent,
            model_used=model,
            answer=answer,
            sql=trace.sql,
            semantic_query=trace.semantic_query,
            fields_used=trace.fields,
            governed=trace.governed,
            notes=trace.notes,
            data=trace.data if request.explain else [],
            thought_steps=steps if request.explain else [],
            token_usage=summarize(trace.llm_calls),
            execution_time_ms=round((time.time() - start_time) * 1000, 2),
        )

    def clear_session(self, session_id: str) -> bool:
        """Clear conversation history for a given session ID."""
        self.last_turns.pop(session_id, None)
        if session_id in self.sessions:
            del self.sessions[session_id]
            return True
        return False


agent_runner = AgentRunner()
