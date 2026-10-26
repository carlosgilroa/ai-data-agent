"""Semantic layer service: runs compiled LookML queries on DuckDB and describes the model for the agent."""

import logging
import math
from datetime import timedelta
from decimal import Decimal
from typing import Any, Callable, Dict, List, Optional

from app.agent.tools.duckdb_tool import duckdb_tool
from app.core.config import settings
from app.semantic.compiler import CompiledQuery, Compiler, SemanticQuery, SemanticQueryError
from app.semantic.dates import DateRange
from app.semantic.model import Field, SemanticModel

logger = logging.getLogger(__name__)


def _json_value(val: Any) -> Any:
    if val is None or isinstance(val, (bool, str, int)):
        return val
    if isinstance(val, float):
        return None if (math.isnan(val) or math.isinf(val)) else round(val, 4)
    if isinstance(val, Decimal):
        return float(val) if val.is_finite() else None
    if hasattr(val, "isoformat"):
        return val.isoformat()
    return str(val)


class SemanticLayer:
    """Loads the LookML model, compiles semantic queries and runs them on the read-only DuckDB connection."""

    def __init__(self):
        self._model: Optional[SemanticModel] = None
        self._compiler: Optional[Compiler] = None
        self._on_reload: List[Callable[[], None]] = []

    @property
    def model(self) -> SemanticModel:
        if self._model is None:
            self._model = SemanticModel(settings.LOOKML_DIR)
            self._compiler = Compiler(self._model, today=settings.today)
        return self._model

    @property
    def compiler(self) -> Compiler:
        self.model  # noqa: B018 - ensures the model is loaded
        return self._compiler

    def on_reload(self, callback: Callable[[], None]) -> None:
        """Register a callback (e.g. re-indexing the catalog) to run when the LookML changes."""
        self._on_reload.append(callback)

    def ensure_fresh(self) -> bool:
        """Reload the model when any .lkml file changed since it was loaded. Returns True on reload."""
        if self._model is not None and not self._model.is_stale():
            return False
        reloaded = self._model is not None
        self._model = SemanticModel(settings.LOOKML_DIR)
        self._compiler = Compiler(self._model, today=settings.today)
        if reloaded:
            logger.info("LookML changed on disk; semantic model reloaded.")
            for callback in self._on_reload:
                try:
                    callback()
                except Exception as e:
                    logger.error("Reload callback failed: %s", e)
        return reloaded

    # ------------------------------------------------------------------ queries

    def compile(self, query: SemanticQuery) -> CompiledQuery:
        return self.compiler.compile(query)

    def run(self, query: SemanticQuery, max_rows: int = 500) -> Dict[str, Any]:
        """Compile and execute a semantic query. Raises SemanticQueryError for queries that can't compile."""
        compiled = self.compile(query)
        try:
            rel = duckdb_tool.conn.sql(compiled.sql)
            rows = rel.limit(max_rows).fetchall()
        except Exception as e:
            logger.error("Semantic query failed: %s | SQL: %s", e, compiled.sql)
            raise SemanticQueryError(f"The compiled query failed on DuckDB: {e}") from e
        columns = compiled.columns
        data = [{c: _json_value(v) for c, v in zip(columns, row)} for row in rows]
        return {
            "explore": compiled.explore,
            "fields": [self.field_info(f) for f in compiled.fields],
            "filters": dict(query.filters),
            "sql": compiled.sql,
            "row_count": len(data),
            "rows": data,
            "notes": self.coverage_notes(compiled),
        }

    def field_values(self, explore: str, field: str, pattern: Optional[str] = None, limit: int = 50) -> List[str]:
        """Distinct values of a dimension in an explore, optionally matching a text pattern."""
        filters = {field: f"%{pattern.strip()}%"} if pattern and pattern.strip() else {}
        query = SemanticQuery(explore=explore, fields=[field], filters=filters, sorts=[field], limit=limit)
        compiled = self.compile(query)
        if compiled.fields[0].is_measure:
            raise SemanticQueryError(f"{field} is a measure; values can only be listed for dimensions.")
        return [str(r[0]) for r in duckdb_tool.conn.sql(compiled.sql).fetchall() if r[0] is not None]

    def coverage_notes(self, compiled: CompiledQuery) -> List[str]:
        """Plain-language notes when a date filter reaches past (or before) the data that exists."""
        first, last = duckdb_tool.get_data_coverage()
        if not first or not last:
            return []
        notes = []
        for field_id, ranges in compiled.date_filters.items():
            for r in ranges:
                notes.extend(self._range_notes(r, first, last))
        return list(dict.fromkeys(notes))

    @staticmethod
    def _range_notes(r: DateRange, first, last) -> List[str]:
        day_after_last = last + timedelta(days=1)
        if r.start and r.start > last:
            return [f"No data yet for {r.describe()}: the data runs through {last.isoformat()}."]
        if r.end and r.end <= first:
            return [f"No data for {r.describe()}: the data starts on {first.isoformat()}."]
        notes = []
        if r.end and r.end > day_after_last:
            notes.append(f"{r.describe()} is only partly covered: the data runs through {last.isoformat()}.")
        if r.start and r.start < first:
            notes.append(f"{r.describe()} starts before the data, which begins on {first.isoformat()}.")
        return notes

    # ------------------------------------------------------------------ descriptions

    @staticmethod
    def field_info(f: Field) -> Dict[str, str]:
        return {"name": f.id, "kind": f.kind, "type": f.type, "description": f.description}

    def explore_overview(self) -> str:
        """One line per explore, for the agent's system prompt."""
        lines = []
        for e in self.model.explores.values():
            lines.append(f"- {e.name}: {e.label}. {e.description}")
        return "\n".join(lines)

    def describe_explore(self, name: str) -> str:
        """All visible fields of an explore with their type and description."""
        explore = self.model.explores.get(name)
        if explore is None:
            raise SemanticQueryError(
                f"Unknown explore '{name}'. Available explores: {', '.join(self.model.explores)}."
            )
        fields = self.model.explore_fields(name)
        lines = [f"Explore {explore.name}: {explore.label}. {explore.description}", "", "Dimensions:"]
        groups_done = set()
        for f in fields:
            if f.is_measure:
                continue
            if f.is_time:
                key = (f.view, f.group)
                if key in groups_done:
                    continue
                groups_done.add(key)
                timeframes = [g.name for g in fields if g.view == f.view and g.group == f.group]
                desc = f.description.rsplit(" (", 1)[0]
                lines.append(f"- {', '.join(f'{f.view}.{t}' for t in timeframes)} (time): {desc}")
            else:
                lines.append(f"- {f.id} ({f.type}): {f.description}")
        lines += ["", "Measures:"]
        lines += [f"- {f.id} ({f.type}): {f.description}" for f in fields if f.is_measure]
        return "\n".join(lines)


semantic_layer = SemanticLayer()
