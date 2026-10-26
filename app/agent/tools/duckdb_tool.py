import time
import logging
import math
from decimal import Decimal
from typing import Dict, Any, List, Optional, Tuple
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
        self._lock_down()

    def _register_views(self):
        """Load the parquet data files into in-memory tables and register LookML derived tables as views.

        The data is loaded into memory (not left as read_parquet views) so that file access
        can be switched off afterwards; restart the app to pick up new data.
        """
        import re

        tables = [
            ("master_records", settings.silver_records_path),
            ("dim_accounts", settings.silver_accounts_path),
            ("dim_categories", settings.silver_categories_path),
            ("dim_labels", settings.silver_labels_path),
            ("fx_rates", settings.silver_fx_path),
            ("mart_monthly_pnl", settings.gold_pnl_path),
            ("mart_expense_summary", settings.gold_expense_summary_path),
        ]

        registered = []
        for name, path in tables:
            if path.exists():
                try:
                    posix_path = path.as_posix()
                    self.conn.execute(f"CREATE OR REPLACE TABLE {name} AS SELECT * FROM read_parquet('{posix_path}')")
                    registered.append(name)
                except Exception as e:
                    logger.error("Failed to register view %s for %s: %s", name, path, e)
            else:
                logger.warning("Parquet path %s does not exist; view %s skipped.", path, name)

        # Dynamically register any LookML derived_table SQL views so PDTs are queryable in DuckDB
        try:
            import lkml

            views_dir = settings.LOOKML_DIR / "views"
            if views_dir.exists():
                for lkml_path in views_dir.glob("**/*.view.lkml"):
                    with open(lkml_path, "r", encoding="utf-8") as f:
                        parsed = lkml.load(f)
                    for view in parsed.get("views", []):
                        v_name = view.get("name")
                        raw_dt_sql = (view.get("derived_table") or {}).get("sql")
                        if v_name and raw_dt_sql:
                            compiled_sql = re.sub(
                                r"`?@\{schema_name\}\.([a-zA-Z0-9_]+)`?", r"\1", raw_dt_sql
                            ).strip().rstrip(";")
                            self.conn.execute(f"CREATE OR REPLACE VIEW {v_name} AS {compiled_sql}")
                            registered.append(v_name)
        except Exception as e:
            logger.warning("Could not register LookML derived table views in DuckDB: %s", e)

        logger.info("DuckDB initialized with views: %s", registered)

    def _lock_down(self):
        """Disable file system, network and extension access, then freeze the configuration.

        Without this, a read-only SELECT could still read any file on the machine
        (e.g. read_text('.env')), because DuckDB table functions access the file system.
        """
        self.conn.execute("SET enable_external_access = false")
        self.conn.execute("SET lock_configuration = true")

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

    def get_table_columns(self, table: str) -> List[str]:
        """Return list of column names for a given table or view."""
        try:
            res = self.conn.execute(f"DESCRIBE {table}").fetchall()
            return [row[0] for row in res]
        except Exception:
            return []

    def get_dimension_values(
        self,
        table: str,
        column: str,
        pattern: Optional[str] = None,
        limit: int = 20,
    ) -> List[str]:
        """Retrieve distinct or sample values for a categorical column safely.
        Validates table and column names to prevent SQL injection.
        """
        active_tables = self.get_available_tables()
        clean_tbl = table.strip()
        matched_tbl = next((t for t in active_tables if t.lower() == clean_tbl.lower()), None)
        if not matched_tbl:
            raise ValueError(
                f"Table '{table}' does not exist in lakehouse database. Available tables: {active_tables}"
            )

        active_columns = self.get_table_columns(matched_tbl)
        clean_col = column.strip()
        matched_col = next((c for c in active_columns if c.lower() == clean_col.lower()), None)
        if not matched_col:
            raise ValueError(
                f"Column '{column}' does not exist in table '{matched_tbl}'. Available columns: {active_columns}"
            )

        safe_limit = max(1, min(limit, 100))

        if pattern and pattern.strip():
            query = (
                f"SELECT DISTINCT {matched_col} FROM {matched_tbl} "
                f"WHERE CAST({matched_col} AS VARCHAR) ILIKE ? AND {matched_col} IS NOT NULL "
                f"ORDER BY 1 LIMIT ?"
            )
            res = self.conn.execute(query, [f"%{pattern.strip()}%", safe_limit]).fetchall()
        else:
            query = (
                f"SELECT DISTINCT {matched_col} FROM {matched_tbl} "
                f"WHERE {matched_col} IS NOT NULL ORDER BY 1 LIMIT ?"
            )
            res = self.conn.execute(query, [safe_limit]).fetchall()

        return [str(row[0]) for row in res]

    def get_temporal_bounds(self) -> Tuple[Optional[str], Optional[str]]:
        """Return (min_period, max_period) across lakehouse gold/silver marts."""
        for tbl in ["mart_monthly_pnl", "mart_expense_summary"]:
            try:
                res = self.conn.execute(
                    f"SELECT MIN(year_month), MAX(year_month) FROM {tbl}"
                ).fetchone()
                if res and res[0] and res[1]:
                    return (str(res[0]), str(res[1]))
            except Exception:
                continue

        try:
            res = self.conn.execute(
                "SELECT MIN(date), MAX(date) FROM master_records"
            ).fetchone()
            if res and res[0] and res[1]:
                return (str(res[0]), str(res[1]))
        except Exception:
            pass

        return (None, None)


# Global singleton instance
duckdb_tool = DuckDBTool()
