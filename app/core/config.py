import os
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

    # Semantic RAG Thresholds
    RAG_TOP_K: int = 5
    RAG_MIN_SIMILARITY: float = 0.28

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
    def silver_fx_path(self) -> Path:
        return self.WALLET_DATA_DIR / "silver" / "fx_rates.parquet"

    @property
    def gold_pnl_path(self) -> Path:
        return self.WALLET_DATA_DIR / "gold" / "mart_monthly_pnl.parquet"

    @property
    def gold_expense_summary_path(self) -> Path:
        return self.WALLET_DATA_DIR / "gold" / "mart_expense_summary.parquet"


settings = Settings()
