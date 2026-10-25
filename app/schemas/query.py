from typing import Any, Dict, List, Optional, Literal
from pydantic import BaseModel, Field


IntentType = Literal["out_of_scope", "capabilities", "simple_query", "complex_query"]


class RouterDecision(BaseModel):
    """Structured output returned by the Intent Router."""
    intent: IntentType = Field(..., description="Classified query intent category")
    reasoning: str = Field(..., description="Concise explanation for the chosen intent")
    metric_search_phrase: Optional[str] = Field(
        default=None,
        description="A natural semantic phrase combining the metric/calculation intent with domain context (e.g. 'total food spending expenses', 'monthly savings rate', 'net cashflow after refunds') suitable for dense vector catalog search.",
    )
    entity_keywords: List[str] = Field(
        default_factory=list,
        description="Extracted domain entities, topics, category names, or account names (e.g. ['food'], ['transportation', 'uber'])",
    )


class AgentThoughtStep(BaseModel):
    """Represents a single reasoning step or tool execution by the AI agent."""
    step_number: int = Field(..., description="Order of step execution")
    tool: str = Field(..., description="Tool invoked (e.g. search_data_catalog, run_sql_query)")
    tool_input: Dict[str, Any] = Field(default_factory=dict, description="Input arguments passed to tool")
    tool_output: Any = Field(None, description="Output returned by tool")
    thought: Optional[str] = Field(None, description="Internal reasoning leading to this action")


class QueryRequest(BaseModel):
    """User request to the AI Data Agent."""
    query: str = Field(..., description="Natural language question about personal finances or lakehouse metrics", json_schema_extra={"example": "What was my total spending in August 2026 by category group?"})
    session_id: Optional[str] = Field(None, description="Optional session or conversation ID")
    explain: bool = Field(default=True, description="Whether to include full agent reasoning steps and generated SQL")


class QueryResponse(BaseModel):
    """Response returned by the AI Data Agent."""
    query: str = Field(..., description="Original user prompt")
    intent: Optional[str] = Field(None, description="Classified user intent from the router")
    model_used: Optional[str] = Field(None, description="Gemini model selected to execute the query")
    answer: str = Field(..., description="Natural language synthesized answer with financial insights")
    sql: Optional[str] = Field(None, description="The SQL query generated and executed against DuckDB")
    data: Optional[List[Dict[str, Any]]] = Field(default_factory=list, description="Raw rows returned by the SQL execution")
    thought_steps: List[AgentThoughtStep] = Field(default_factory=list, description="Traceability log of agent tool calls")
    execution_time_ms: float = Field(default=0.0, description="End-to-end execution time in milliseconds")


class DirectSQLRequest(BaseModel):
    """Request to directly execute a validated SQL query against the local DuckDB lakehouse."""
    sql: str = Field(..., description="DuckDB SQL statement (read-only queries only)", json_schema_extra={"example": "SELECT category_group, sum(amount) as total FROM master_records WHERE is_transfer = false GROUP BY 1 ORDER BY 2 DESC"})


class DirectSQLResponse(BaseModel):
    """Response from direct SQL execution."""
    sql: str
    row_count: int
    columns: List[str]
    data: List[Dict[str, Any]]
    execution_time_ms: float


class HealthResponse(BaseModel):
    """Application health status."""
    status: str = "ok"
    version: str
    duckdb_connected: bool
    tables_available: List[str]
    vector_store_ready: bool
    gemini_configured: bool

