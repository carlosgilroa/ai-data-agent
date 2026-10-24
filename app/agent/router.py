import json
import logging
from typing import Tuple, List, Optional
from langchain_core.messages import SystemMessage, HumanMessage
from app.core.config import settings
from app.schemas.query import IntentType, RouterDecision

logger = logging.getLogger(__name__)

ROUTER_SYSTEM_PROMPT = """You are an Intent Classification Router for a personal finance and lakehouse analytics AI agent.
Your sole responsibility is to classify the user's message into one of four intent categories:

1. "out_of_scope":
   - Questions completely unrelated to personal finance, expenses, income, cashflow, budgets, currencies, accounting, or the data lakehouse (e.g., general trivia, weather, sports, coding tasks unrelated to our financial data, recipes).
2. "capabilities":
   - Questions asking what you can do, how you work, what tools or tables you have access to, or general greetings asking for help with the system.
3. "simple_query":
   - Direct single-metric lookups, distinct counts, metadata lookups, or single-table aggregations (e.g., "How many spend categories do we have?", "What was my total spend in March 2026?", "List my accounts").
4. "complex_query":
   - Multi-hop analytical questions, period-over-period comparisons, Tier 1 vs Tier 2 accounting reconciliations, multi-currency FX analysis, budget variance diagnostics, or queries requiring multiple SQL queries/joins.
"""


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
            )
            self.structured_llm = self.llm.with_structured_output(RouterDecision)
            logger.info("Initialized IntentRouter with model %s", self.router_model)
        else:
            self.llm = None
            self.structured_llm = None

    def classify(self, query: str, has_session_context: bool = False) -> RouterDecision:
        """Classify user query into an IntentType using gemini-3.1-flash-lite."""
        self._ensure_llm()
        if self.structured_llm is None:
            raise RuntimeError(
                "GEMINI_API_KEY is not configured. An active Google Gemini API key is required "
                "to execute the intent router."
            )

        context_hint = (
            "\n(Note: This message is a follow-up within an active financial data conversation session.)"
            if has_session_context
            else ""
        )
        messages = [
            SystemMessage(content=ROUTER_SYSTEM_PROMPT + context_hint),
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
                        content='Respond ONLY with valid JSON: {"intent": "out_of_scope"|"capabilities"|"simple_query"|"complex_query", "reasoning": "..."}'
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
        if intent == "out_of_scope":
            return (settings.GEMINI_ROUTER_MODEL, 0, [])
        if intent == "capabilities":
            return (settings.GEMINI_ROUTER_MODEL, 1, ["search_data_catalog"])
        if intent == "simple_query":
            return (settings.GEMINI_SIMPLE_MODEL, 3, ["search_data_catalog", "run_sql_query"])
        # complex_query
        return (settings.GEMINI_MODEL, 6, ["search_data_catalog", "run_sql_query"])


intent_router = IntentRouter()
