"""Per-request token accounting for every Gemini call (router, tool loop and final synthesis).

Each ChatGoogleGenerativeAI instance gets a `UsageCallback` naming its model. The callback adds the
call's `usage_metadata` to the tracker of the request being served, held in a context variable so
concurrent requests don't mix.
"""

from contextvars import ContextVar
from typing import Any, Dict, List, Optional

from langchain_core.callbacks import BaseCallbackHandler

_current: ContextVar[Optional[List[Dict[str, Any]]]] = ContextVar("token_usage_calls", default=None)


def start_tracking() -> List[Dict[str, Any]]:
    """Begin a fresh record of LLM calls for the current request and return it."""
    calls: List[Dict[str, Any]] = []
    _current.set(calls)
    return calls


def summarize(calls: List[Dict[str, Any]]) -> Dict[str, Any]:
    return {
        "llm_calls": len(calls),
        "input_tokens": sum(c["input_tokens"] for c in calls),
        "output_tokens": sum(c["output_tokens"] for c in calls),
        "total_tokens": sum(c["total_tokens"] for c in calls),
        "calls": calls,
    }


class UsageCallback(BaseCallbackHandler):
    """Records the token usage of each call made by one model."""

    def __init__(self, model: str):
        self.model = model

    def on_llm_end(self, response, **kwargs: Any) -> None:
        calls = _current.get()
        if calls is None:
            return
        for generations in response.generations:
            for gen in generations:
                usage = getattr(getattr(gen, "message", None), "usage_metadata", None) or {}
                if not usage:
                    continue
                calls.append(
                    {
                        "model": self.model,
                        "input_tokens": int(usage.get("input_tokens", 0)),
                        "output_tokens": int(usage.get("output_tokens", 0)),
                        "total_tokens": int(usage.get("total_tokens", 0)),
                    }
                )
