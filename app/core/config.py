import os
from datetime import date
from pathlib import Path
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Application settings and environment configuration."""

    # Project metadata
    PROJECT_NAME: str = "AI-Native Semantic Query Agent & RAG Microservice"
    VERSION: str = "0.1.0"
    API_V1_STR: str = "/api/v1"
    DEBUG: bool = True

    # Server configuration
    HOST: str = "0.0.0.0"
    PORT: int = 8000

    # Gemini LLM configuration & Multi-Tier Model Routing
    GEMINI_API_KEY: str = ""
    GEMINI_ROUTER_MODEL: str = "gemini-3.1-flash-lite"
    GEMINI_SIMPLE_MODEL: str = "gemini-3.5-flash-lite"
    GEMINI_MODEL: str = "gemini-3.8-flash"

    # Semantic RAG Thresholds (0.20 noise floor + 80% relative cutoff to top match)
    RAG_TOP_K: int = 5
    RAG_MIN_SIMILARITY: float = 0.20
    RAG_RELATIVE_CUTOFF_RATIO: float = 0.80

    # Semantic layer agent
    # Fixed "today" (YYYY-MM-DD) for tests and demos; empty = the real date
    AGENT_TODAY: str = ""
    # Let the agent fall back to read-only raw SQL when no explore can answer a question
    ALLOW_RAW_SQL_FALLBACK: bool = True
    AGENT_INSTRUCTIONS_PATH: str = "agent_instructions.yaml"  # glossary, default filters, golden queries
    QUERY_LOG_PATH: str = "logs/agent_queries.jsonl"  # full trace of each agent answer; empty disables it
    # Number of explores/fields the catalog search returns to the model
    # Looker-style retrieval per question: explores first, then fields within them, then instructions
    RETRIEVAL_EXPLORES: int = 3
    RETRIEVAL_MEASURES: int = 5
    RETRIEVAL_DIMENSIONS: int = 5
    RETRIEVAL_GLOSSARY: int = 3
    RETRIEVAL_GOLDEN_QUERIES: int = 2

    # Data paths (pointing to local lakehouse and semantic layer)
    WALLET_DATA_DIR: Path = Path("C:/projects/wallet/data")
    LOOKML_DIR: Path = Path("C:/projects/wallet-lookml")
    CHROMA_PERSIST_DIR: str = "data/chroma"

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore"
    )

    @property
    def silver_records_path(self) -> Path:
        return self.WALLET_DATA_DIR / "silver" / "master_records.parquet"

    @property
    def silver_accounts_path(self) -> Path:
        return self.WALLET_DATA_DIR / "silver" / "dim_accounts.parquet"

    @property
    def silver_categories_path(self) -> Path:
        return self.WALLET_DATA_DIR / "silver" / "dim_categories.parquet"

    @property
    def silver_labels_path(self) -> Path:
        return self.WALLET_DATA_DIR / "silver" / "dim_labels.parquet"

    @property
    def silver_fx_path(self) -> Path:
        return self.WALLET_DATA_DIR / "silver" / "fx_rates.parquet"

    @property
    def gold_pnl_path(self) -> Path:
        return self.WALLET_DATA_DIR / "gold" / "mart_monthly_pnl.parquet"

    @property
    def gold_expense_summary_path(self) -> Path:
        return self.WALLET_DATA_DIR / "gold" / "mart_expense_summary.parquet"


    def today(self) -> date:
        return date.fromisoformat(self.AGENT_TODAY) if self.AGENT_TODAY else date.today()


settings = Settings()
