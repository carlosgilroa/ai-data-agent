from typing import List, Optional, Literal
from pydantic import BaseModel, Field


class DimensionDefinition(BaseModel):
    """Pydantic model representing a LookML dimension or lakehouse attribute."""
    name: str = Field(..., description="Technical identifier of the dimension (e.g. category_group)")
    view: str = Field(..., description="LookML view name (e.g. dim_categories)")
    type: str = Field(default="string", description="Data type: string, number, date, boolean")
    sql_formula: str = Field(..., description="SQL expression or column mapping (e.g. ${TABLE}.category_group)")
    description: str = Field(..., description="Business meaning and context of the dimension")
    tags: List[str] = Field(default_factory=list, description="Categorization tags")


class MetricDefinition(BaseModel):
    """Pydantic model representing a LookML measure or business KPI."""
    name: str = Field(..., description="Technical identifier of the measure (e.g. net_cashflow_eur)")
    view: str = Field(..., description="LookML view name (e.g. fct_transactions)")
    type: str = Field(default="sum", description="Aggregate type: sum, count, average, number")
    sql_formula: str = Field(..., description="SQL aggregate expression (e.g. sum(amount_eur))")
    description: str = Field(..., description="Exact business logic and definition of this metric")
    filter_condition: Optional[str] = Field(None, description="SQL filter applied (e.g. is_transfer = false)")
    tags: List[str] = Field(default_factory=list, description="Categorization tags")


class CatalogItem(BaseModel):
    """Unified document representation for ChromaDB vector indexing."""
    id: str = Field(..., description="Unique document ID (e.g. fct_transactions.net_cashflow_eur)")
    item_type: Literal["dimension", "metric", "table", "view", "explore", "join", "glossary", "golden_query"] = Field(..., description="Catalog entry type")
    name: str = Field(..., description="Name of the item")
    view: str = Field(..., description="Parent view or table name")
    description: str = Field(..., description="Detailed semantic description used for embeddings")
    sql_formula: Optional[str] = Field(None, description="Associated SQL code/formula")
    table_name: Optional[str] = Field(None, description="Physical parquet or BigQuery table")
    examples: List[str] = Field(default_factory=list, description="Sample user questions this item answers")
    explores: List[str] = Field(default_factory=list, description="LookML explores in which this field can be queried")
    distance: Optional[float] = Field(None, description="Raw cosine distance from ChromaDB (0.0 = identical)")
    similarity_score: Optional[float] = Field(None, description="Cosine similarity score (1.0 - distance)")


class CatalogListResponse(BaseModel):
    """API response listing indexed catalog items."""
    total: int
    items: List[CatalogItem]


class CatalogSearchResponse(BaseModel):
    """API response for semantic vector search over data catalog."""
    query: str
    total_matches: int
    results: List[CatalogItem]
