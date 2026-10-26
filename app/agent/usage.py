"""Per-request token accounting for every Gemini call (router, tool loop and final synthesis).

Each ChatGoogleGenerativeAI instance gets a `UsageCallback` naming its model. The callback adds the
call's `usage_metadata` to the tracker of the request being served, held in a context variable so
concurrent requests don't mix. It also records the call's input (the messages and the names of the
tools offered), so a response shows exactly what each model call was given.
"""

from contextvars import ContextVar
from typing import Any, Dict, List, Optional
from uuid import UUID

from langchain_core.callbacks import BaseCallbackHandler
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, SystemMessage, ToolMessage

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


def _text(content: Any) -> str:
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        return "".join(part.get("text", "") if isinstance(part, dict) else str(part) for part in content)
    return str(content)


def _message_record(message: BaseMessage) -> Dict[str, Any]:
    """One input message as {role, content}, plus the tool calls it makes or the tool call it answers."""
    roles = {SystemMessage: "system", HumanMessage: "user", AIMessage: "assistant", ToolMessage: "tool"}
    role = next((r for cls, r in roles.items() if isinstance(message, cls)), message.type)
    record: Dict[str, Any] = {"role": role, "content": _text(message.content)}
    if isinstance(message, AIMessage) and message.tool_calls:
        record["tool_calls"] = [{"name": tc.get("name"), "args": tc.get("args")} for tc in message.tool_calls]
    if isinstance(message, ToolMessage):
        record["tool_call_id"] = message.tool_call_id
        if message.name:
            record["name"] = message.name
    return record


def _tool_names(invocation_params: Dict[str, Any]) -> List[str]:
    """Names of the tools offered to the model, in whichever shape the integration passes them."""
    names: List[str] = []

    def collect(tool: Any) -> None:
        get = tool.get if isinstance(tool, dict) else (lambda key, default=None: getattr(tool, key, default))
        for declaration in get("function_declarations") or get("functionDeclarations") or []:
            collect(declaration)
        function = get("function")
        name = get("name") or (function.get("name") if isinstance(function, dict) else None)
        if name:
            names.append(str(name))

    for tool in invocation_params.get("tools") or invocation_params.get("functions") or []:
        collect(tool)
    return names


class UsageCallback(BaseCallbackHandler):
    """Records the input and the token usage of each call made by one model."""

    def __init__(self, model: str):
        self.model = model
        # Input of each call in flight, by LangChain run id, until its on_llm_end
        self._inputs: Dict[UUID, Dict[str, Any]] = {}

    def on_chat_model_start(
        self, serialized: Dict[str, Any], messages: List[List[BaseMessage]], *, run_id: UUID, **kwargs: Any
    ) -> None:
        if _current.get() is None:
            return
        params = kwargs.get("invocation_params") or {}
        record: Dict[str, Any] = {
            "messages": [_message_record(m) for batch in messages for m in batch],
            "tools": _tool_names(params),
        }
        # Structured output (the router) has no tools, but answers in the shape of a JSON schema
        schema = params.get("response_json_schema") or params.get("response_schema")
        if isinstance(schema, dict):
            record["output_schema"] = schema.get("title") or "JSON schema"
        self._inputs[run_id] = record

    def on_llm_error(self, error: BaseException, *, run_id: UUID, **kwargs: Any) -> None:
        self._inputs.pop(run_id, None)

    def on_llm_end(self, response, *, run_id: Optional[UUID] = None, **kwargs: Any) -> None:
        call_input = self._inputs.pop(run_id, None) if run_id is not None else None
        calls = _current.get()
        if calls is None:
            return
        for generations in response.generations:
            for gen in generations:
                usage = getattr(getattr(gen, "message", None), "usage_metadata", None) or {}
                if not usage:
                    continue
                call: Dict[str, Any] = {
                    "model": self.model,
                    "input_tokens": int(usage.get("input_tokens", 0)),
                    "output_tokens": int(usage.get("output_tokens", 0)),
                    "total_tokens": int(usage.get("total_tokens", 0)),
                }
                if call_input is not None:
                    call["input"] = call_input
                calls.append(call)
