import time
import logging
import math
from decimal import Decimal
from typing import Dict, Any, List, Optional
import duckdb
from app.core.config import settings

logger = logging.getLogger(__name__)

FORBIDDEN_KEYWORDS = [
    "drop", "delete", "insert", "update", "alter", "create", 
    "truncate", "grant", "revoke", "copy", "attach", "detach", "pragma", "call"
]


class DuckDBTool:
    """Safe, read-only SQL execution engine over local Parquet Lakehouse files."""

    def __init__(self):
        self.conn = duckdb.connect(":memory:")
        self._register_views()

    def _register_views(self):
        """Register friendly table names pointing directly to parquet data files."""
        tables = [
            ("master_records", settings.silver_records_path),
            ("fx_rates", settings.silver_fx_path),
            ("mart_monthly_pnl", settings.gold_pnl_path),
            ("mart_expense_summary", settings.gold_expense_summary_path),
        ]

        registered = []
        for name, path in tables:
            if path.exists():
                try:
                    posix_path = path.as_posix()
                    self.conn.execute(f"CREATE OR REPLACE VIEW {name} AS SELECT * FROM read_parquet('{posix_path}')")
                    registered.append(name)
                except Exception as e:
                    logger.error("Failed to register view %s for %s: %s", name, path, e)
            else:
                logger.warning("Parquet path %s does not exist; view %s skipped.", path, name)

        logger.info("DuckDB initialized with views: %s", registered)

    def validate_sql(self, sql: str) -> None:
        """Ensure the SQL query is strictly read-only and safe."""
        import re
        cleaned = sql.strip().lower()
        if not (cleaned.startswith("select") or cleaned.startswith("with")):
            raise ValueError("Only SELECT or WITH queries are permitted.")

        # Check forbidden keywords with word boundary
        for kw in FORBIDDEN_KEYWORDS:
            if re.search(rf"\b{kw}\b", cleaned):
                raise ValueError(f"Forbidden SQL operation detected: '{kw}'. Queries must be read-only.")

    def run_query(self, sql: str, max_rows: int = 100) -> Dict[str, Any]:
        """Execute SQL query safely and return results as structured dictionary."""
        start_time = time.time()
        self.validate_sql(sql)

        try:
            rel = self.conn.sql(sql)
            if rel is None:
                return {
                    "columns": [],
                    "row_count": 0,
                    "data": [],
                    "execution_time_ms": (time.time() - start_time) * 1000
                }

            # Limit result size for agent safety and fetch rows natively without pandas
            rel_limited = rel.limit(max_rows)
            columns = list(rel_limited.columns)
            rows = rel_limited.fetchall()
            duration_ms = (time.time() - start_time) * 1000

            # Pure Python JSON-safe record conversion
            records = []
            for row in rows:
                record = {}
                for col_name, val in zip(columns, row):
                    if val is None:
                        record[col_name] = None
                    elif isinstance(val, (bool, str, int)):
                        record[col_name] = val
                    elif isinstance(val, float):
                        record[col_name] = None if (math.isnan(val) or math.isinf(val)) else val
                    elif isinstance(val, Decimal):
                        record[col_name] = float(val) if val.is_finite() else None
                    elif hasattr(val, "isoformat"):  # date, datetime, time
                        record[col_name] = val.isoformat()
                    else:
                        record[col_name] = str(val)
                records.append(record)

            return {
                "columns": columns,
                "row_count": len(records),
                "data": records,
                "execution_time_ms": round(duration_ms, 2)
            }
        except Exception as e:
            logger.error("DuckDB query failed: %s | SQL: %s", e, sql)
            raise RuntimeError(f"DuckDB Execution Error: {str(e)}")

    def get_available_tables(self) -> List[str]:
        """Return list of active views in DuckDB."""
        try:
            res = self.conn.execute("SHOW TABLES").fetchall()
            return [row[0] for row in res]
        except Exception:
            return []


# Global singleton instance
duckdb_tool = DuckDBTool()
