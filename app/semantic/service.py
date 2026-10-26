"""Semantic layer service: runs compiled LookML queries on DuckDB and describes the model for the agent."""

import logging
import math
from datetime import timedelta
from decimal import Decimal
from typing import Any, Callable, Dict, List, Optional

from app.agent.tools.duckdb_tool import duckdb_tool
from app.core.config import settings
import duckdb

from app.semantic.compiler import MAX_LIMIT, CompiledQuery, Compiler, SemanticQuery, SemanticQueryError
from app.semantic.dates import DateRange
from app.semantic.instructions import AgentInstructions, instructions_path, load_instructions
from app.semantic.model import Field, SemanticModel

logger = logging.getLogger(__name__)

MAX_FIELD_VALUES = 100  # Looker samples up to 100 values from a field
_STOPWORDS = {"the", "and", "for", "with", "from", "spend", "spent", "my", "on", "at", "in", "of"}


def _contains_filter(term: str) -> str:
    """Looker filter expression for "contains term", escaping the grammar's special characters."""
    clean = term.strip().replace("%", "").replace("^", "").replace(",", "^,")
    return f"%{clean}%"


def _search_terms(term: str) -> List[str]:
    """Search variants for a word or phrase: whole, singular, and each meaningful word."""
    base = " ".join(term.strip().split())
    if not base:
        return []
    variants = [base]
    # Single words of a phrase must be 4+ letters, so "eating out" doesn't match "outlay"
    words = [w for w in base.split(" ") if len(w) >= 4 and w.lower() not in _STOPWORDS]
    for w in [base] + words:
        for suffix in ("ies", "es", "s"):
            if w.lower().endswith(suffix) and len(w) - len(suffix) >= 3:
                variants.append(w[: -len(suffix)] + ("y" if suffix == "ies" else ""))
                break
    if len(words) > 1:
        variants += words
    return list(dict.fromkeys(v for v in variants if v))


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
        self._cardinalities: Dict[tuple, int] = {}
        self._instructions: Optional[AgentInstructions] = None
        self._instructions_mtime: Optional[float] = None

    @property
    def model(self) -> SemanticModel:
        if self._model is None:
            self._model = SemanticModel(settings.LOOKML_DIR)
            self._compiler = Compiler(self._model, today=settings.today)
        return self._model

    @property
    def instructions(self) -> AgentInstructions:
        if self._instructions is None:
            self._instructions = load_instructions()
            self._instructions_mtime = self._instructions_file_mtime()
        return self._instructions

    @staticmethod
    def _instructions_file_mtime() -> Optional[float]:
        path = instructions_path()
        return path.stat().st_mtime if path.exists() else None

    @property
    def compiler(self) -> Compiler:
        self.model  # noqa: B018 - ensures the model is loaded
        return self._compiler

    def on_reload(self, callback: Callable[[], None]) -> None:
        """Register a callback (e.g. re-indexing the catalog) to run when the LookML changes."""
        self._on_reload.append(callback)

    def ensure_fresh(self) -> bool:
        """Reload the model when any .lkml file or the agent instructions changed. Returns True on reload."""
        instructions_changed = (
            self._instructions is not None and self._instructions_file_mtime() != self._instructions_mtime
        )
        if instructions_changed:
            self._instructions = None
        if self._model is not None and not self._model.is_stale() and not instructions_changed:
            return False
        reloaded = self._model is not None
        self._cardinalities.clear()
        self._model = SemanticModel(settings.LOOKML_DIR)
        self._compiler = Compiler(self._model, today=settings.today)
        if reloaded:
            logger.info("LookML or agent instructions changed on disk; semantic model reloaded.")
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
        """Compile and execute a semantic query. Raises SemanticQueryError for queries that can't compile.

        Default filters from the agent instructions apply unless the query filters that field itself.
        """
        defaults = [d for d in self.instructions.defaults_for(query.explore) if d.field not in query.filters]
        if defaults:
            query = query.model_copy(update={"filters": {**query.filters, **{d.field: d.value for d in defaults}}})
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
            "notes": [f"Default filter applied: {d.field} = {d.value}" for d in defaults] + self.coverage_notes(compiled),
        }

    def field_values(
        self, explore: str, field: str, search: Optional[str] = None, limit: int = MAX_FIELD_VALUES
    ) -> Dict[str, Any]:
        """Distinct values of one dimension, like Looker's filter suggestions: up to `limit` values,
        optionally only those containing a search term, and whether the list was cut off.
        """
        f = self._resolve(explore, field)
        if f.is_measure:
            raise SemanticQueryError(f"{field} is a measure; values can only be listed for dimensions.")
        # LookML suggest_explore / suggest_dimension point the lookup at another field
        lookup_explore = f.suggest_explore or explore
        lookup_field = f.suggest_dimension or f.id
        filters = {lookup_field: _contains_filter(search)} if search and search.strip() else {}
        query = SemanticQuery(
            explore=lookup_explore, fields=[lookup_field], filters=filters, sorts=[lookup_field], limit=MAX_LIMIT
        )
        sql = self.compile(query).sql
        rows = duckdb_tool.conn.sql(sql).fetchall()
        values = [str(r[0]) for r in rows if r[0] is not None]
        return {
            "field": f.id,
            "search": search or None,
            "total": len(values),
            "values": values[:limit],
            "truncated": len(values) > limit,
        }

    def search_values(
        self, explore: str, term: str, per_field: int = 5, only: Optional[List[str]] = None
    ) -> List[Dict[str, Any]]:
        """Find which searchable dimensions of an explore contain a term, like Looker's fuzzy value search.

        The term is tried whole, singular and word by word, so "restaurants" also matches
        "Restaurant & fast food". Fields with an exact (case-insensitive) match come first.
        """
        terms = _search_terms(term)
        if not terms:
            return []
        expression = ",".join(_contains_filter(t) for t in terms)
        matches = []
        for f in self.model.searchable_fields(explore):
            if only is not None and f.id not in only:
                continue
            query = SemanticQuery(explore=explore, fields=[f.id], filters={f.id: expression}, sorts=[f.id], limit=MAX_LIMIT)
            try:
                rows = duckdb_tool.conn.sql(self.compile(query).sql).fetchall()
            except (SemanticQueryError, duckdb.Error) as e:
                logger.debug("Value search skipped %s: %s", f.id, e)
                continue
            values = [str(r[0]) for r in rows if r[0] is not None]
            if not values:
                continue
            exact = any(v.lower() == term.strip().lower() for v in values)
            matches.append({
                "_cardinality": self._cardinality(explore, f.id),
                "field": f.id,
                "values": values[:per_field],
                "total": len(values),
                "truncated": len(values) > per_field,
                "exact": exact,
            })
        # Exact matches first, then the most specific match, then the most categorical field
        # (a category list beats a long merchant list when both contain the word)
        matches.sort(key=lambda m: (not m["exact"], m["total"], m.pop("_cardinality")))
        return matches

    def search_values_everywhere(self, term: str, per_field: int = 5) -> List[Dict[str, Any]]:
        """search_values over every explore, each field searched once, with the explores it is in."""
        explores_by_field: Dict[str, List[str]] = {}
        for explore in self.model.explores:
            for f in self.model.searchable_fields(explore):
                explores_by_field.setdefault(f.id, []).append(explore)
        matches: List[Dict[str, Any]] = []
        searched = set()
        for explore in self.model.explores:
            fields = [f.id for f in self.model.searchable_fields(explore) if f.id not in searched]
            if not fields:
                continue
            searched.update(fields)
            for match in self.search_values(explore, term, per_field=per_field, only=fields):
                match["explores"] = explores_by_field[match["field"]]
                matches.append(match)
        matches.sort(key=lambda m: (not m["exact"], m["total"], self._cardinality(m["explores"][0], m["field"])))
        return matches

    def _cardinality(self, explore: str, field_id: str) -> int:
        key = (explore, field_id)
        if key not in self._cardinalities:
            sql = self.compile(SemanticQuery(explore=explore, fields=[field_id], limit=MAX_LIMIT)).sql
            self._cardinalities[key] = duckdb_tool.conn.sql(f"SELECT COUNT(*) FROM ({sql})").fetchall()[0][0]
        return self._cardinalities[key]

    def _resolve(self, explore: str, field: str) -> Field:
        if explore not in self.model.explores:
            raise SemanticQueryError(f"Unknown explore '{explore}'. Available explores: {', '.join(self.model.explores)}.")
        for f in self.model.explore_fields(explore):
            if f.id == field:
                return f
        # Let the compiler produce its usual error with suggestions
        self.compile(SemanticQuery(explore=explore, fields=[field]))
        raise SemanticQueryError(f"No field '{field}' in explore '{explore}'.")

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

    def describe_fields(self, name: str, kind: str) -> str:
        """Visible dimensions or measures of an explore, one per line (Looker's get_dimensions / get_measures)."""
        explore = self.model.explores.get(name)
        if explore is None:
            raise SemanticQueryError(
                f"Unknown explore '{name}'. Available explores: {', '.join(self.model.explores)}."
            )
        text = self.describe_explore(name)
        dims, _, measures = text.partition("\n\nMeasures:\n")
        if kind == "measure":
            return f"Measures of {name}:\n{measures}"
        return dims.split("\n\n", 1)[1] if "\n\n" in dims else dims

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
