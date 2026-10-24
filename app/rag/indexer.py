import logging
from pathlib import Path
from typing import List, Dict, Any
import duckdb
from app.core.config import settings
from app.schemas.catalog import CatalogItem
from app.rag.store import vector_store

logger = logging.getLogger(__name__)


class SemanticCatalogIndexer:
    """Extracts semantic definitions from LookML views, explores, joins, and data lake tables,
    indexing them into ChromaDB as the single analytical source of truth.
    """

    def __init__(self):
        self.lookml_dir = Path(settings.LOOKML_DIR)
        self.data_dir = Path(settings.WALLET_DATA_DIR)
        self._view_meta: Dict[str, Dict[str, Any]] = {}

    @staticmethod
    def _extract_measure_filters(measure: Dict[str, Any]) -> List[str]:
        """Extract filter key-value conditions from lkml 'filters__all' or 'filters' structures."""
        filter_strs: List[str] = []
        raw_filters_all = measure.get("filters__all", [])
        for group in raw_filters_all:
            if isinstance(group, list):
                for item in group:
                    if isinstance(item, dict):
                        for k, v in item.items():
                            filter_strs.append(f"{k} = {v}")
            elif isinstance(group, dict):
                for k, v in group.items():
                    filter_strs.append(f"{k} = {v}")

        if not filter_strs:
            raw_filters = measure.get("filters", [])
            if isinstance(raw_filters, list):
                for item in raw_filters:
                    if isinstance(item, dict):
                        for k, v in item.items():
                            filter_strs.append(f"{k} = {v}")
            elif isinstance(raw_filters, dict):
                for k, v in raw_filters.items():
                    filter_strs.append(f"{k} = {v}")

        return filter_strs

    @staticmethod
    def _format_measure_sql(measure: Dict[str, Any], filter_strs: List[str]) -> str:
        """Format measure SQL formula with its LookML aggregation type and filter clause."""
        m_type = (measure.get("type") or "number").lower()
        raw_sql = (measure.get("sql") or "").strip()

        agg_map = {
            "sum": "SUM",
            "count": "COUNT",
            "count_distinct": "COUNT(DISTINCT",
            "average": "AVG",
            "min": "MIN",
            "max": "MAX",
        }

        if not raw_sql:
            formula = "COUNT(*)" if m_type == "count" else f"{m_type.upper()}(*)"
        elif m_type == "count_distinct":
            formula = f"COUNT(DISTINCT {raw_sql})"
        elif m_type in agg_map:
            agg_fn = agg_map[m_type]
            if raw_sql.upper().startswith(f"{agg_fn}("):
                formula = raw_sql
            else:
                formula = f"{agg_fn}({raw_sql})"
        else:
            formula = raw_sql

        if filter_strs:
            formula = f"{formula} FILTER (WHERE {' AND '.join(filter_strs)})"
        return formula

    @staticmethod
    def _format_dimension_sql(dim: Dict[str, Any]) -> str:
        """Extract SQL expression or CASE WHEN statement from a LookML dimension."""
        if dim.get("sql"):
            return str(dim.get("sql")).strip()

        case_block = dim.get("case")
        if isinstance(case_block, dict):
            when_parts = []
            for w in case_block.get("whens", []):
                if isinstance(w, dict) and w.get("sql") and w.get("label"):
                    when_parts.append(f"WHEN {w['sql']} THEN '{w['label']}'")
            else_label = case_block.get("else")
            else_str = f" ELSE '{else_label}'" if else_label is not None else ""
            if when_parts:
                return f"CASE {' '.join(when_parts)}{else_str} END"

        return f"${{TABLE}}.{dim.get('name', '')}"

    @staticmethod
    def _strip_lookml_table_macro(raw_sql_table: str, view_name: str) -> str:
        """Strip LookML `@{schema_name}.` macro prefix and backticks from sql_table_name."""
        if not raw_sql_table:
            return view_name
        cleaned = raw_sql_table.strip().strip("`").strip()
        if cleaned.startswith("@{schema_name}."):
            cleaned = cleaned.removeprefix("@{schema_name}.")
        return cleaned.strip("`").strip() or view_name

    def extract_from_lookml(self) -> List[CatalogItem]:
        """Parse .view.lkml and .model.lkml files to extract measures, dimensions, dimension_groups, explores, and joins."""
        items: List[CatalogItem] = []
        import lkml

        # 1. Parse LookML Views (Dimensions, Dimension Groups & Measures)
        views_dir = self.lookml_dir / "views"
        if views_dir.exists():
            for lkml_path in views_dir.glob("**/*.view.lkml"):
                try:
                    with open(lkml_path, "r", encoding="utf-8") as f:
                        parsed = lkml.load(f)

                    views = parsed.get("views", [])
                    for view in views:
                        view_name = view.get("name", lkml_path.stem.replace(".view", ""))
                        table_name = self._strip_lookml_table_macro(
                            view.get("sql_table_name", ""), view_name
                        )

                        # Capture LookML grain description and primary key for lakehouse table metadata
                        view_desc = view.get("description", "")
                        pk_dim = next(
                            (
                                d.get("name")
                                for d in view.get("dimensions", [])
                                if str(d.get("primary_key", "")).lower() in ("yes", "true")
                            ),
                            None,
                        )
                        self._view_meta[table_name] = {
                            "grain": view_desc,
                            "primary_key": pk_dim,
                        }

                        # Index derived_table (PDT) definition if present
                        derived_table = view.get("derived_table")
                        if isinstance(derived_table, dict) and derived_table.get("sql"):
                            dt_cols = [d.get("name", "") for d in view.get("dimensions", []) if d.get("name")]
                            pdt_desc_parts = [view_desc or f"LookML Persistent Derived Table (PDT) '{view_name}' registered in DuckDB."]
                            if pk_dim:
                                pdt_desc_parts.append(f"Primary Key: {pk_dim}")
                            pdt_desc_parts.append(f"Available columns: {', '.join(dt_cols)}")
                            items.append(
                                CatalogItem(
                                    id=f"view.{view_name}",
                                    item_type="view",
                                    name=view_name,
                                    view=view_name,
                                    description=" | ".join(pdt_desc_parts),
                                    sql_formula=f"SELECT * FROM {view_name}",
                                    table_name=view_name,
                                    examples=[f"Query {view_name}", f"Analyze category velocity in {view_name}"],
                                )
                            )

                        # Standard Dimensions (including CASE dimensions)
                        for dim in view.get("dimensions", []):
                            dim_name = dim.get("name", "")
                            dim_sql = self._format_dimension_sql(dim)
                            items.append(
                                CatalogItem(
                                    id=f"{view_name}.{dim_name}",
                                    item_type="dimension",
                                    name=dim_name,
                                    view=view_name,
                                    description=dim.get("description", f"Dimension {dim_name} in {view_name}"),
                                    sql_formula=dim_sql,
                                    table_name=table_name,
                                    examples=[
                                        f"Filter or group by {dim_name}",
                                        f"Break down metrics by {dim_name}"
                                    ]
                                )
                            )

                        # Dimension Groups (Timeframes)
                        for dgroup in view.get("dimension_groups", []):
                            dg_name = dgroup.get("name", "")
                            timeframes = dgroup.get("timeframes", [])
                            dg_sql = dgroup.get("sql", f"${{TABLE}}.{dg_name}")
                            tf_note = f" [Timeframes: {', '.join(timeframes)}]" if timeframes else ""
                            dg_desc = dgroup.get("description", f"Time dimension group {dg_name} in {view_name}") + tf_note
                            items.append(
                                CatalogItem(
                                    id=f"{view_name}.{dg_name}",
                                    item_type="dimension",
                                    name=dg_name,
                                    view=view_name,
                                    description=dg_desc,
                                    sql_formula=dg_sql,
                                    table_name=table_name,
                                    examples=[
                                        f"Filter or group by {dg_name} date/month/year",
                                        f"Trend metrics over {dg_name}"
                                    ]
                                )
                            )

                        # Measures
                        for measure in view.get("measures", []):
                            m_name = measure.get("name", "")
                            desc = measure.get("description", f"Measure {m_name} in {view_name}")
                            filter_strs = self._extract_measure_filters(measure)
                            if filter_strs:
                                desc += f" [Filters: {', '.join(filter_strs)}]"
                            formula = self._format_measure_sql(measure, filter_strs)

                            items.append(
                                CatalogItem(
                                    id=f"{view_name}.{m_name}",
                                    item_type="metric",
                                    name=m_name,
                                    view=view_name,
                                    description=desc,
                                    sql_formula=formula,
                                    table_name=table_name,
                                    examples=[
                                        f"Calculate {m_name}",
                                        f"What is the {m_name}?"
                                    ]
                                )
                            )
                except Exception as e:
                    logger.error("Error parsing LookML view file %s: %s", lkml_path, e)

        # 2. Parse LookML Models (Explores and Join Relationships)
        models_dir = self.lookml_dir / "models"
        if models_dir.exists():
            for model_path in models_dir.glob("**/*.model.lkml"):
                try:
                    with open(model_path, "r", encoding="utf-8") as f:
                        parsed = lkml.load(f)

                    for explore in parsed.get("explores", []):
                        explore_name = explore.get("name")
                        explore_label = explore.get("label", explore_name)
                        explore_desc = explore.get("description", f"Explore {explore_name}")

                        items.append(
                            CatalogItem(
                                id=f"explore.{explore_name}",
                                item_type="explore",
                                name=explore_name,
                                view=explore_name,
                                description=f"LookML Explore '{explore_label}': {explore_desc}",
                                sql_formula=None,
                                table_name=explore_name,
                                examples=[f"Query {explore_name}", f"Explore financial data in {explore_name}"]
                            )
                        )

                        for join in explore.get("joins", []):
                            join_name = join.get("name")
                            rel = join.get("relationship", "many_to_one")
                            sql_on = join.get("sql_on", "").strip()
                            join_type = join.get("type", "left_outer")

                            items.append(
                                CatalogItem(
                                    id=f"join.{explore_name}.{join_name}",
                                    item_type="join",
                                    name=f"{explore_name} -> {join_name}",
                                    view=explore_name,
                                    description=(
                                        f"LookML Join in explore '{explore_name}': {join_type} join to '{join_name}' "
                                        f"(cardinality: {rel}). Join condition: {sql_on}"
                                    ),
                                    sql_formula=sql_on,
                                    table_name=join_name,
                                    examples=[
                                        f"Join {explore_name} with {join_name}",
                                        f"Connect {explore_name} and {join_name} on {sql_on}"
                                    ]
                                )
                            )
                except Exception as e:
                    logger.error("Error parsing LookML model file %s: %s", model_path, e)

        return items

    def extract_from_lakehouse(self) -> List[CatalogItem]:
        """Inspect DuckDB Parquet schemas to index physical table structures without pandas dependency."""
        items: List[CatalogItem] = []
        tables = [
            ("master_records", settings.silver_records_path, "Cleaned transaction ledger with accounts, categories, and amounts"),
            ("dim_accounts", settings.silver_accounts_path, "Conformed financial accounts dimension with balances, credit limits, and active status"),
            ("dim_categories", settings.silver_categories_path, "Conformed spend and income categories dimension with parent hierarchy and cardinality"),
            ("dim_labels", settings.silver_labels_path, "Conformed transaction labels/tags dimension with cumulative inflows and outflows"),
            ("fx_rates", settings.silver_fx_path, "Daily foreign exchange conversion rates (USD, EUR, COP, GBP)"),
            ("mart_monthly_pnl", settings.gold_pnl_path, "Pre-aggregated monthly P&L financial mart with net cashflow and savings rate"),
            ("mart_expense_summary", settings.gold_expense_summary_path, "Monthly expense breakdown with budget caps and variances"),
        ]

        conn = duckdb.connect(":memory:")
        for table_name, file_path, desc in tables:
            if not file_path.exists():
                continue
            try:
                # Use native fetchall() without pandas
                posix_path = file_path.as_posix()
                res = conn.execute(f"DESCRIBE SELECT * FROM read_parquet('{posix_path}')").fetchall()
                col_names = [row[0] for row in res]
                cols = [f"{row[0]} ({row[1]})" for row in res]

                # Dynamic statistics: row count & date coverage
                total_rows = conn.execute(f"SELECT COUNT(*) FROM read_parquet('{posix_path}')").fetchone()[0]

                date_info = ""
                if "year_month" in col_names:
                    try:
                        min_ym, max_ym = conn.execute(
                            f"SELECT MIN(year_month), MAX(year_month) FROM read_parquet('{posix_path}')"
                        ).fetchone()
                        if min_ym and max_ym:
                            date_info = f"Date Coverage: {min_ym} to {max_ym}"
                    except Exception:
                        pass
                elif "date" in col_names:
                    try:
                        min_dt, max_dt = conn.execute(
                            f"SELECT MIN(date), MAX(date) FROM read_parquet('{posix_path}')"
                        ).fetchone()
                        if min_dt and max_dt:
                            date_info = f"Date Coverage: {min_dt} to {max_dt}"
                    except Exception:
                        pass

                view_meta = self._view_meta.get(table_name, {})
                grain_desc = view_meta.get("grain") or desc
                pk_name = view_meta.get("primary_key")

                desc_parts = [grain_desc]
                if pk_name:
                    desc_parts.append(f"Primary Key: {pk_name}")
                desc_parts.append(f"Stats: {total_rows:,} rows")
                if date_info:
                    desc_parts.append(date_info)
                desc_parts.append(f"Available columns: {', '.join(cols)}")

                full_desc = " | ".join(desc_parts)

                items.append(
                    CatalogItem(
                        id=f"table.{table_name}",
                        item_type="table",
                        name=table_name,
                        view=table_name,
                        description=full_desc,
                        sql_formula=f"read_parquet('{posix_path}')",
                        table_name=table_name,
                        examples=[f"Query {table_name}", f"Inspect data in {table_name}"]
                    )
                )
            except Exception as e:
                logger.error("Failed to describe parquet table %s: %s", table_name, e)

        return items

    def run_indexing(self) -> Dict[str, Any]:
        """Wipe ChromaDB collection and execute full indexing pipeline from LookML and Lakehouse."""
        vector_store.clear()

        lookml_items = self.extract_from_lookml()
        lakehouse_items = self.extract_from_lakehouse()
        all_items = lookml_items + lakehouse_items

        vector_store.add_items(all_items)

        metrics_count = sum(1 for i in all_items if i.item_type == "metric")
        dimensions_count = sum(1 for i in all_items if i.item_type == "dimension")
        explores_count = sum(1 for i in all_items if i.item_type == "explore")
        joins_count = sum(1 for i in all_items if i.item_type == "join")
        tables_count = sum(1 for i in all_items if i.item_type == "table")

        logger.info(
            "Indexing completed (clean slate): %d total items (%d metrics, %d dimensions, %d explores, %d joins, %d tables)",
            len(all_items), metrics_count, dimensions_count, explores_count, joins_count, tables_count
        )
        return {
            "total_indexed": len(all_items),
            "metrics": metrics_count,
            "dimensions": dimensions_count,
            "explores": explores_count,
            "joins": joins_count,
            "tables": tables_count,
            "items": all_items
        }


indexer = SemanticCatalogIndexer()
