import os
import logging
from pathlib import Path
from typing import List, Dict, Any, Optional
import duckdb
from app.core.config import settings
from app.schemas.catalog import CatalogItem
from app.rag.store import vector_store

logger = logging.getLogger(__name__)


class SemanticCatalogIndexer:
    """Extracts semantic definitions from LookML views and DuckDB Parquet lakehouse,

    indexing them into ChromaDB for agent RAG retrieval.
    """

    def __init__(self):
        self.lookml_dir = Path(settings.LOOKML_DIR)
        self.data_dir = Path(settings.WALLET_DATA_DIR)

    def extract_from_lookml(self) -> List[CatalogItem]:
        """Parse .lkml files to extract measures, dimensions, and semantic business logic."""
        items: List[CatalogItem] = []
        try:
            import lkml
        except ImportError:
            logger.warning("lkml package not installed. Generating catalog from predefined LookML semantics.")
            return self._get_fallback_lookml_catalog()

        views_dir = self.lookml_dir / "views"
        if not views_dir.exists():
            logger.warning("LookML views dir %s does not exist. Using fallback catalog.", views_dir)
            return self._get_fallback_lookml_catalog()

        for lkml_path in views_dir.glob("**/*.view.lkml"):
            try:
                with open(lkml_path, "r", encoding="utf-8") as f:
                    parsed = lkml.load(f)

                views = parsed.get("views", [])
                for view in views:
                    view_name = view.get("name", lkml_path.stem.replace(".view", ""))
                    
                    # Dimensions
                    for dim in view.get("dimensions", []):
                        items.append(
                            CatalogItem(
                                id=f"{view_name}.{dim.get('name')}",
                                item_type="dimension",
                                name=dim.get("name", ""),
                                view=view_name,
                                description=dim.get("description", f"Dimension {dim.get('name')} in {view_name}"),
                                sql_formula=dim.get("sql"),
                                table_name=view.get("sql_table_name"),
                                examples=[
                                    f"Filter or group by {dim.get('name')}",
                                    f"Break down metrics by {dim.get('name')}"
                                ]
                            )
                        )

                    # Measures
                    for measure in view.get("measures", []):
                        desc = measure.get("description", f"Measure {measure.get('name')} in {view_name}")
                        formula = measure.get("sql", f"{measure.get('type')}(*)")
                        filters = measure.get("filters", [])
                        if filters:
                            filter_desc = " Filters: " + ", ".join([f"{k} = {v}" for f in filters for k, v in f.items()])
                            desc += filter_desc
                            
                        items.append(
                            CatalogItem(
                                id=f"{view_name}.{measure.get('name')}",
                                item_type="metric",
                                name=measure.get("name", ""),
                                view=view_name,
                                description=desc,
                                sql_formula=formula,
                                table_name=view.get("sql_table_name"),
                                examples=[
                                    f"Calculate {measure.get('name')}",
                                    f"What is the {measure.get('name')} for this month?"
                                ]
                            )
                        )
            except Exception as e:
                logger.error("Error parsing LookML file %s: %s", lkml_path, e)

        if not items:
            return self._get_fallback_lookml_catalog()

        return items

    def _get_fallback_lookml_catalog(self) -> List[CatalogItem]:
        """Curated semantic catalog extracted directly from the verified LookML semantic layer."""
        return [
            CatalogItem(
                id="fct_transactions.total_spend_usd",
                item_type="metric",
                name="total_spend_usd",
                view="fct_transactions",
                description="Total cash outflow in USD (excluding transfers). Calculated as sum(abs(amount_usd)) where is_transfer = false and amount < 0.",
                sql_formula="SUM(ABS(amount)) FILTER (WHERE is_transfer = false AND type = 'expense')",
                table_name="master_records",
                examples=["What were my total expenses in August?", "How much did I spend on living costs?"]
            ),
            CatalogItem(
                id="fct_transactions.total_income_usd",
                item_type="metric",
                name="total_income_usd",
                view="fct_transactions",
                description="Total cash inflow in USD (excluding transfers). Calculated as sum(amount_usd) where is_transfer = false and amount > 0.",
                sql_formula="SUM(amount) FILTER (WHERE is_transfer = false AND type = 'income')",
                table_name="master_records",
                examples=["What was my total income this month?", "How much money did I earn?"]
            ),
            CatalogItem(
                id="fct_transactions.net_cash_flow_usd",
                item_type="metric",
                name="net_cash_flow_usd",
                view="fct_transactions",
                description="Net cash flow (Total Income - Total Spend) in USD. Essential KPI measuring monthly surplus or burn.",
                sql_formula="total_income_usd - total_spend_usd",
                table_name="mart_monthly_pnl",
                examples=["What is my net cashflow?", "Did I save money in July?"]
            ),
            CatalogItem(
                id="fct_transactions.tier1_operational_spend_usd",
                item_type="metric",
                name="tier1_operational_spend_usd",
                view="fct_transactions",
                description="Two-Tier Accounting: Essential routine living burn rate (Tier 1 Needs, rent, groceries, utilities).",
                sql_formula="SUM(ABS(amount)) FILTER (WHERE tier_group LIKE '%Tier 1%')",
                table_name="master_records",
                examples=["What is my Tier 1 burn rate?", "How much do essential living expenses cost?"]
            ),
            CatalogItem(
                id="fct_transactions.tier2_sinking_fund_spend_usd",
                item_type="metric",
                name="tier2_sinking_fund_spend_usd",
                view="fct_transactions",
                description="Two-Tier Accounting: Capital sinking funds, savings allocations, and irregular asset investments (Tier 2).",
                sql_formula="SUM(ABS(amount)) FILTER (WHERE tier_group LIKE '%Tier 2%')",
                table_name="master_records",
                examples=["How much did I allocate to sinking funds?", "What was my Tier 2 capital expense?"]
            ),
            CatalogItem(
                id="fct_transactions.savings_rate_percentage",
                item_type="metric",
                name="savings_rate_percentage",
                view="fct_transactions",
                description="Percentage of total income retained or allocated to Tier 2 capital funds. savings_rate = net_cashflow / total_income * 100.",
                sql_formula="net_cashflow / operating_income * 100",
                table_name="mart_monthly_pnl",
                examples=["What is my savings rate?", "Am I meeting my 30% savings rate target?"]
            ),
            CatalogItem(
                id="dim_categories.category_group",
                item_type="dimension",
                name="category_group",
                view="dim_categories",
                description="High-level category classification (e.g. Housing, Food & Dining, Transportation, Utilities, Shopping, Financial).",
                sql_formula="category_group",
                table_name="master_records",
                examples=["Break down spending by category group", "Top spending categories"]
            ),
            CatalogItem(
                id="dim_accounts.account_class",
                item_type="dimension",
                name="account_class",
                view="dim_accounts",
                description="Classification of financial accounts: checking, savings, credit, cash, investment.",
                sql_formula="account_class",
                table_name="master_records",
                examples=["Show balances by account class", "Spending on credit cards"]
            ),
        ]

    def extract_from_lakehouse(self) -> List[CatalogItem]:
        """Inspect DuckDB Parquet schemas to index physical table structures."""
        items: List[CatalogItem] = []
        tables = [
            ("master_records", settings.silver_records_path, "Cleaned transaction ledger with accounts, categories, and amounts"),
            ("fx_rates", settings.silver_fx_path, "Daily foreign exchange conversion rates (USD, EUR, COP, GBP)"),
            ("mart_monthly_pnl", settings.gold_pnl_path, "Pre-aggregated monthly P&L financial mart with net cashflow and savings rate"),
            ("mart_expense_summary", settings.gold_expense_summary_path, "Monthly expense breakdown with budget caps and variances"),
        ]

        conn = duckdb.connect(":memory:")
        for table_name, file_path, desc in tables:
            if not file_path.exists():
                continue
            try:
                df = conn.execute(f"DESCRIBE SELECT * FROM read_parquet('{file_path.as_posix()}')").df()
                cols = [f"{row['column_name']} ({row['column_type']})" for _, row in df.iterrows()]
                items.append(
                    CatalogItem(
                        id=f"table.{table_name}",
                        item_type="table",
                        name=table_name,
                        view=table_name,
                        description=f"{desc}. Available columns: {', '.join(cols)}",
                        sql_formula=f"read_parquet('{file_path.as_posix()}')",
                        table_name=table_name,
                        examples=[f"Query {table_name}", f"Inspect data in {table_name}"]
                    )
                )
            except Exception as e:
                logger.error("Failed to describe parquet table %s: %s", table_name, e)

        return items

    def run_indexing(self) -> Dict[str, Any]:
        """Execute full indexing pipeline and upsert into ChromaDB."""
        lookml_items = self.extract_from_lookml()
        lakehouse_items = self.extract_from_lakehouse()
        all_items = lookml_items + lakehouse_items

        vector_store.add_items(all_items)

        metrics_count = sum(1 for i in all_items if i.item_type == "metric")
        dimensions_count = sum(1 for i in all_items if i.item_type == "dimension")
        tables_count = sum(1 for i in all_items if i.item_type == "table")

        logger.info(
            "Indexing completed: %d total items (%d metrics, %d dimensions, %d tables)",
            len(all_items), metrics_count, dimensions_count, tables_count
        )
        return {
            "total_indexed": len(all_items),
            "metrics": metrics_count,
            "dimensions": dimensions_count,
            "tables": tables_count,
            "items": all_items
        }


indexer = SemanticCatalogIndexer()
