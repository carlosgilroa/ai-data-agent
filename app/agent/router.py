import json
import logging
from typing import Tuple, List, Optional
from langchain_core.messages import SystemMessage, HumanMessage
from app.agent.usage import UsageCallback
from app.core.config import settings
from app.schemas.query import IntentType, RouterDecision

logger = logging.getLogger(__name__)

# Named after Looker's agent tools (get_explores, get_dimensions, get_measures, query)
SEMANTIC_TOOLS = [
    "search_semantic_catalog",
    "get_explores",
    "get_dimensions",
    "get_measures",
    "get_field_values",
    "query",
]
RAW_SQL_TOOLS = ["get_table_schema", "run_sql_query"]

# Nothing here names the dataset: the domain comes from the explores in the semantic layer (domain_overview)
ROUTER_SYSTEM_PROMPT = """You route questions for a data agent that answers only from a LookML semantic layer.
The data it can answer about (one line per explore):
{domain}

1. Classify the user's message into one intent:
   - "out_of_scope": unrelated to the data above.
   - "capabilities": asks what the agent can do, how it works, or what data or tools it has.
   - "simple_query": one or a few measures, optionally filtered, for one period or broken down by one dimension.
   - "complex_query": needs several steps or queries: comparisons between periods or groups, ratios across explores, diagnosing a change, or open-ended exploration.

2. For simple_query and complex_query, split the question into LookML terms (leave a list empty when the question has none):
   - "measures": what to calculate, in the user's words.
   - "dimensions": what to group or break the result down by, in the user's words. Empty for a single total.
   - "filter_values": specific values to filter on, exactly as the user wrote them. Never dates.
   - "time_period": the period as a filter phrase, e.g. "last month", "this year", "2026-09", "2026-08, 2026-09".
   Example: "What was <measure> for <value> last month?" -> measures ["<measure>"], dimensions [], filter_values ["<value>"], time_period "last month".
   Example: "Top <dimension> by <measure> this year" -> measures ["<measure>"], dimensions ["<dimension>"], filter_values [], time_period "this year".
"""


def domain_overview() -> str:
    """What the agent can answer about, from the explores' LookML labels (so no dataset is hardcoded)."""
    from app.semantic.service import semantic_layer

    return "\n".join(f"- {e.label or e.name}" for e in semantic_layer.model.explores.values())


class IntentRouter:
    """Two-step intent router powered by gemini-3.1-flash-lite."""

    def __init__(self):
        self.api_key: str = settings.GEMINI_API_KEY
        self.router_model: str = settings.GEMINI_ROUTER_MODEL
        self.llm = None
        self.structured_llm = None

    def _ensure_llm(self):
        """Lazily initialize or refresh the gemini-3.1-flash-lite router client."""
        current_key = settings.GEMINI_API_KEY
        current_model = settings.GEMINI_ROUTER_MODEL
        if (
            self.structured_llm is not None
            and self.api_key == current_key
            and self.router_model == current_model
        ):
            return

        self.api_key = current_key
        self.router_model = current_model

        if self.api_key and self.api_key.strip() not in ("", "your_gemini_api_key_here"):
            from langchain_google_genai import ChatGoogleGenerativeAI

            self.llm = ChatGoogleGenerativeAI(
                model=self.router_model,
                google_api_key=self.api_key,
                temperature=0.0,
                callbacks=[UsageCallback(self.router_model)],
            )
            self.structured_llm = self.llm.with_structured_output(RouterDecision)
            logger.info("Initialized IntentRouter with model %s", self.router_model)
        else:
            self.llm = None
            self.structured_llm = None

    def classify(self, query: str, has_session_context: bool = False) -> RouterDecision:
        """Classify user query into an IntentType and extract semantic search parameters using gemini-3.1-flash-lite."""
        self._ensure_llm()
        if self.structured_llm is None:
            raise RuntimeError(
                "GEMINI_API_KEY is not configured. An active Google Gemini API key is required "
                "to execute the intent router."
            )

        context_hint = (
            "\n(Note: This message is a follow-up within an active conversation session.)"
            if has_session_context
            else ""
        )
        messages = [
            SystemMessage(content=ROUTER_SYSTEM_PROMPT.format(domain=domain_overview()) + context_hint),
            HumanMessage(content=query),
        ]

        try:
            decision = self.structured_llm.invoke(messages)
            if isinstance(decision, RouterDecision):
                return decision
            if isinstance(decision, dict):
                return RouterDecision(**decision)
        except Exception as e:
            logger.warning("Structured router classification fallback triggered: %s", e)

        # Fallback: direct JSON prompt if structured output fails
        if self.llm is not None:
            raw = self.llm.invoke(
                messages
                + [
                    HumanMessage(
                        content=(
                            'Respond ONLY with valid JSON: {"intent": "out_of_scope"|"capabilities"|"simple_query"|"complex_query", '
                            '"reasoning": "...", "measures": ["..."], "dimensions": ["..."], "filter_values": ["..."], "time_period": "..."}'
                        )
                    )
                ]
            )
            content = raw.content if isinstance(raw.content, str) else str(raw.content)
            content = content.strip().removeprefix("```json").removeprefix("```").removesuffix("```").strip()
            try:
                parsed = json.loads(content)
                return RouterDecision(
                    intent=parsed.get("intent", "simple_query"),
                    reasoning=parsed.get("reasoning", "Parsed via JSON fallback."),
                    measures=parsed.get("measures") or [],
                    dimensions=parsed.get("dimensions") or [],
                    filter_values=parsed.get("filter_values") or [],
                    time_period=parsed.get("time_period"),
                )
            except Exception:
                pass

        return RouterDecision(
            intent="simple_query",
            reasoning="Defaulted to simple_query due to unstructured router output.",
        )

    @staticmethod
    def get_route_config(intent: IntentType) -> Tuple[str, int, List[str]]:
        """Map classified intent to (target_model_name, max_iterations, allowed_tool_names)."""
        analytical_tools = list(SEMANTIC_TOOLS)
        if settings.ALLOW_RAW_SQL_FALLBACK:
            analytical_tools += RAW_SQL_TOOLS
        if intent == "out_of_scope":
            return (settings.GEMINI_ROUTER_MODEL, 0, [])
        if intent == "capabilities":
            return (settings.GEMINI_ROUTER_MODEL, 2, ["get_explores", "get_measures", "get_dimensions"])
        if intent == "simple_query":
            return (settings.GEMINI_SIMPLE_MODEL, 5, analytical_tools)
        # complex_query
        return (settings.GEMINI_MODEL, 8, analytical_tools)


intent_router = IntentRouter()
