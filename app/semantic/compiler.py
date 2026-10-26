"""Deterministic LookML-to-SQL compiler (no LLM involved).

A semantic query names an explore, the fields to return, filters, sorts and a limit:

    {"explore": "fct_transactions",
     "fields": ["dim_categories.category_group", "fct_transactions.total_spend_cop"],
     "filters": {"fct_transactions.transaction_month": "2026-09"},
     "sorts": ["fct_transactions.total_spend_cop desc"]}

The compiler resolves every field to the SQL defined in LookML (measure filters, `type: number`
measures built from other measures, CASE and yesno dimensions, dimension_group timeframes,
derived tables), adds only the joins the query needs, and returns DuckDB SQL. Measures from a
joined view are computed with symmetric aggregates (deduplicated on the view's primary key), so a
join never double counts them.
"""

import difflib
import re
from dataclasses import dataclass, field
from datetime import date
from typing import Callable, Dict, List, Optional, Set, Tuple

from pydantic import BaseModel, Field as PydanticField

from app.semantic.dates import DateFilterError, DateRange, parse_date_filter
from app.semantic.model import Explore, Field, SemanticModel

DEFAULT_LIMIT = 500
MAX_LIMIT = 5000
REF = re.compile(r"\$\{([a-zA-Z0-9_]+)(?:\.([a-zA-Z0-9_]+))?\}")
AGGREGATES = {"sum": "SUM", "average": "AVG", "min": "MIN", "max": "MAX"}
SYMMETRIC_LIST_FN = {"sum": "LIST_SUM", "average": "LIST_AVG"}
JOIN_SQL = {"left_outer": "LEFT JOIN", "inner": "INNER JOIN", "full_outer": "FULL OUTER JOIN", "cross": "CROSS JOIN"}


class SemanticQueryError(ValueError):
    """A semantic query that cannot be compiled; the message tells the caller how to fix it."""


class SemanticQuery(BaseModel):
    explore: str = PydanticField(..., description="Explore name, e.g. fct_transactions")
    fields: List[str] = PydanticField(..., description="Fields as view.field, dimensions and measures")
    filters: Dict[str, str] = PydanticField(default_factory=dict, description="Field -> filter expression")
    sorts: List[str] = PydanticField(default_factory=list, description="'view.field' or 'view.field desc'")
    limit: int = PydanticField(default=DEFAULT_LIMIT, description="Maximum rows to return")


@dataclass
class CompiledQuery:
    sql: str
    explore: str
    fields: List[Field]
    date_filters: Dict[str, List[DateRange]] = field(default_factory=dict)

    @property
    def columns(self) -> List[str]:
        return [f.id for f in self.fields]

    @property
    def measures(self) -> List[Field]:
        return [f for f in self.fields if f.is_measure]


def _quote(value: str) -> str:
    return "'" + value.replace("'", "''") + "'"


def _split_values(expression: str) -> List[str]:
    """Split a filter on commas, keeping Looker's `^,` escape for commas inside a value."""
    placeholder = "\x00"
    parts = expression.replace("^,", placeholder).split(",")
    return [p.replace(placeholder, ",").strip() for p in parts if p.strip()]


class Compiler:
    def __init__(self, model: SemanticModel, today: Callable[[], date] = date.today):
        self.model = model
        self.today = today

    # ------------------------------------------------------------------ public

    def compile(self, query: SemanticQuery) -> CompiledQuery:
        explore = self._explore(query.explore)
        ctx = _Context(self.model, explore)

        if not query.fields:
            raise SemanticQueryError("Select at least one field.")
        selected = [ctx.field(name) for name in dict.fromkeys(query.fields)]
        filter_fields = {name: ctx.field(name) for name in query.filters}
        dims = [f for f in selected if not f.is_measure]
        measures = [f for f in selected if f.is_measure]

        select_parts = [f"{ctx.dimension_sql(f)} AS \"{f.id}\"" for f in dims]
        select_parts += [f"{ctx.measure_sql(f)} AS \"{f.id}\"" for f in measures]

        where, having, date_filters = [], [], {}
        for name, expression in query.filters.items():
            f = filter_fields[name]
            if f.is_measure:
                having.append(self._value_condition(ctx.measure_sql(f), "number", expression, f.id))
            elif f.is_time:
                ranges = self._date_ranges(expression, f.id)
                date_filters[f.id] = ranges
                where.append(self._date_condition(ctx.time_base_sql(f), ranges))
            else:
                where.append(self._value_condition(ctx.dimension_sql(f), f.type, expression, f.id))

        sql = [f"SELECT\n  " + ",\n  ".join(select_parts), ctx.from_clause()]
        if where:
            sql.append("WHERE " + "\n  AND ".join(where))
        if dims and measures:
            sql.append("GROUP BY " + ", ".join(str(i + 1) for i in range(len(dims))))
        elif dims:
            sql.append("GROUP BY ALL")
        if having:
            sql.append("HAVING " + "\n  AND ".join(having))
        order = self._order_by(query.sorts, selected, dims, measures)
        if order:
            sql.append("ORDER BY " + ", ".join(order))
        limit = max(1, min(int(query.limit or DEFAULT_LIMIT), MAX_LIMIT))
        sql.append(f"LIMIT {limit}")

        return CompiledQuery(sql="\n".join(sql), explore=explore.name, fields=selected, date_filters=date_filters)

    # ------------------------------------------------------------------ helpers

    def _explore(self, name: str) -> Explore:
        explore = self.model.explores.get(name)
        if explore is None:
            raise SemanticQueryError(
                f"Unknown explore '{name}'.{_suggest(name, self.model.explores)} "
                f"Available explores: {', '.join(self.model.explores)}."
            )
        return explore

    def _date_ranges(self, expression: str, field_id: str) -> List[DateRange]:
        try:
            return parse_date_filter(expression, self.today())
        except DateFilterError as e:
            raise SemanticQueryError(f"Filter on {field_id}: {e}") from e

    @staticmethod
    def _date_condition(expr: str, ranges: List[DateRange]) -> str:
        parts = []
        for r in ranges:
            conds = []
            if r.start:
                conds.append(f"{expr} >= DATE '{r.start.isoformat()}'")
            if r.end:
                conds.append(f"{expr} < DATE '{r.end.isoformat()}'")
            parts.append("(" + " AND ".join(conds or ["TRUE"]) + ")")
        return "(" + " OR ".join(parts) + ")"

    def _value_condition(self, expr: str, ftype: str, expression: str, field_id: str) -> str:
        expression = str(expression).strip()
        if ftype == "yesno":
            if expression.lower() not in ("yes", "no"):
                raise SemanticQueryError(f"Filter on {field_id} must be 'yes' or 'no'.")
            return f"({expr}) = {'TRUE' if expression.lower() == 'yes' else 'FALSE'}"
        if ftype == "number":
            return self._number_condition(expr, expression, field_id)
        return self._string_condition(expr, expression)

    @staticmethod
    def _number_condition(expr: str, expression: str, field_id: str) -> str:
        ors = []
        for part in _split_values(expression):
            low = part.lower()
            if low in ("null", "is null"):
                ors.append(f"({expr}) IS NULL")
                continue
            if low in ("-null", "not null", "is not null"):
                ors.append(f"({expr}) IS NOT NULL")
                continue
            m = re.fullmatch(r"\[?\s*(-?[\d.]+)\s*(?:,|to)\s*(-?[\d.]+)\s*\]?", part, re.I)
            if m and ("[" in part or " to " in low):
                ors.append(f"({expr}) BETWEEN {float(m.group(1))} AND {float(m.group(2))}")
                continue
            ands = []
            for clause in re.split(r"\s+and\s+", part, flags=re.I):
                m = re.fullmatch(r"(>=|<=|!=|<>|>|<|=|not)?\s*(-?\d+(?:\.\d+)?)", clause.strip(), re.I)
                if not m:
                    raise SemanticQueryError(
                        f"Filter on {field_id}: cannot read '{part}'. Use e.g. '>1000000', '<=0', '[100,500]' or '42'."
                    )
                op = (m.group(1) or "=").lower()
                op = {"not": "<>", "!=": "<>"}.get(op, op)
                ands.append(f"({expr}) {op} {float(m.group(2))}")
            ors.append("(" + " AND ".join(ands) + ")")
        if not ors:
            raise SemanticQueryError(f"Empty filter on {field_id}.")
        return "(" + " OR ".join(ors) + ")"

    @staticmethod
    def _string_condition(expr: str, expression: str) -> str:
        includes, excludes = [], []
        for value in _split_values(expression):
            negate = value.startswith("-")
            raw = value[1:].strip() if negate else value
            upper = raw.upper()
            if upper == "NULL":
                cond = f"({expr}) IS NULL"
            elif upper == "EMPTY":
                cond = f"(({expr}) IS NULL OR ({expr}) = '')"
            elif "%" in raw:
                cond = f"({expr}) ILIKE {_quote(raw)}"
            else:
                cond = f"LOWER({expr}) = LOWER({_quote(raw)})"
            if negate:
                excludes.append(f"NOT COALESCE({cond}, FALSE)")
            else:
                includes.append(cond)
        parts = []
        if includes:
            parts.append("(" + " OR ".join(includes) + ")")
        parts.extend(excludes)
        return "(" + " AND ".join(parts or ["TRUE"]) + ")"

    @staticmethod
    def _order_by(sorts: List[str], selected: List[Field], dims: List[Field], measures: List[Field]) -> List[str]:
        ids = {f.id for f in selected}
        order = []
        for s in sorts:
            parts = s.strip().split()
            if not parts:
                continue
            name = parts[0]
            direction = parts[1].upper() if len(parts) > 1 else "ASC"
            if direction not in ("ASC", "DESC"):
                raise SemanticQueryError(f"Sort direction must be asc or desc, got '{parts[1]}'.")
            if name not in ids:
                raise SemanticQueryError(f"Sort field '{name}' must also be one of the selected fields.")
            order.append(f"\"{name}\" {direction} NULLS LAST")
        if order:
            return order
        time_dims = [f for f in dims if f.is_time]
        if time_dims:
            return [f"\"{f.id}\" ASC" for f in time_dims]
        if measures:
            return [f"\"{measures[0].id}\" DESC NULLS LAST"]
        return [f"\"{f.id}\" ASC" for f in dims]


class _Context:
    """Per-query state: field resolution inside one explore and the joins the SQL needs."""

    def __init__(self, model: SemanticModel, explore: Explore):
        self.model = model
        self.explore = explore
        self.aliases = explore.alias_to_view()
        self.used_aliases: Set[str] = {explore.base_alias}

    # ------------------------------------------------------------------ fields

    def field(self, name: str) -> Field:
        if "." not in name:
            raise SemanticQueryError(
                f"Field '{name}' must be written as view.field.{self._suggest_field(name)}"
            )
        alias, fname = name.split(".", 1)
        if alias not in self.aliases:
            raise SemanticQueryError(
                f"View '{alias}' is not part of explore '{self.explore.name}'. "
                f"Views in this explore: {', '.join(self.aliases)}.{self._suggest_field(name)}"
            )
        view = self.model.views[self.aliases[alias]]
        f = view.fields.get(fname)
        if f is None:
            raise SemanticQueryError(f"No field '{name}' in explore '{self.explore.name}'.{self._suggest_field(name)}")
        if f.hidden:
            raise SemanticQueryError(
                f"Field '{name}' is hidden in the semantic layer and can't be used. {f.description}"
                f"{self._suggest_field(name)}"
            )
        self.used_aliases.add(alias)
        return Field(**{**f.__dict__, "view": alias})

    def _suggest_field(self, name: str) -> str:
        candidates = [f.id for f in self.model.explore_fields(self.explore.name)]
        return _suggest(name, candidates)

    def _raw(self, alias: str, fname: str) -> Field:
        view = self.model.views[self.aliases[alias]]
        f = view.fields.get(fname)
        if f is None:
            raise SemanticQueryError(f"LookML references unknown field '{alias}.{fname}'.")
        self.used_aliases.add(alias)
        return f

    def _substitute(self, sql: str, alias: str, resolve: Callable[[str, str], str]) -> str:
        sql = sql.replace("${TABLE}", alias)

        def repl(m: re.Match) -> str:
            if m.group(2):
                return f"({resolve(m.group(1), m.group(2))})"
            return f"({resolve(alias, m.group(1))})"

        return REF.sub(repl, sql)

    def dimension_sql(self, f: Field, alias: Optional[str] = None) -> str:
        alias = alias or f.view
        if f.is_time:
            return _timeframe_sql(self.time_base_sql(f, alias), f.timeframe)
        if f.case:
            whens = " ".join(
                f"WHEN {self._substitute(w['sql'], alias, self._dimension_ref)} THEN {_quote(w['label'])}"
                for w in f.case.get("whens", [])
            )
            other = f.case.get("else")
            return f"CASE {whens}{' ELSE ' + _quote(other) if other is not None else ''} END"
        if not f.sql:
            return f"{alias}.{f.name}"
        return self._substitute(f.sql, alias, self._dimension_ref)

    def time_base_sql(self, f: Field, alias: Optional[str] = None) -> str:
        alias = alias or f.view
        group = self.model.views[self.aliases[alias]].dimension_groups[f.group]
        return f"CAST({self._substitute(group['sql'], alias, self._dimension_ref)} AS DATE)"

    def _dimension_ref(self, alias: str, fname: str) -> str:
        if alias not in self.aliases:
            raise SemanticQueryError(f"LookML references view '{alias}', which is not joined in this explore.")
        return self.dimension_sql(self._raw(alias, fname), alias)

    def measure_sql(self, f: Field, alias: Optional[str] = None) -> str:
        alias = alias or f.view
        if f.type == "number":
            return self._substitute(f.sql or "NULL", alias, self._measure_ref)

        conds = [self._measure_filter(alias, fname, value) for item in f.filters for fname, value in item.items()]
        cond = " AND ".join(conds)
        view = self.model.views[self.aliases[alias]]
        pk = next((d for d in view.fields.values() if d.primary_key), None)
        pk_sql = self.dimension_sql(pk, alias) if pk else None
        fanned_out = alias != self.explore.base_alias or self._base_fanned_out()

        if f.type in ("count", "count_distinct"):
            if f.type == "count":
                target = pk_sql if (pk_sql and fanned_out) else None
            else:
                target = self._substitute(f.sql, alias, self._dimension_ref)
            if target is None:
                return f"COUNT(CASE WHEN {cond} THEN 1 END)" if cond else "COUNT(*)"
            return f"COUNT(DISTINCT CASE WHEN {cond} THEN {target} END)" if cond else f"COUNT(DISTINCT {target})"

        if f.type not in AGGREGATES:
            raise SemanticQueryError(f"Measure type '{f.type}' of {f.id} is not supported.")
        expr = self._substitute(f.sql, alias, self._dimension_ref)
        if cond:
            expr = f"CASE WHEN {cond} THEN {expr} END"
        if fanned_out and f.type in SYMMETRIC_LIST_FN:
            if not pk_sql:
                raise SemanticQueryError(
                    f"{f.id} can't be aggregated through a join because view '{view.name}' has no primary key. "
                    f"Query it from its own explore instead."
                )
            # Symmetric aggregate: keep one value per primary key so the join's fan-out can't double count
            return (
                f"{SYMMETRIC_LIST_FN[f.type]}([x.v FOR x IN "
                f"LIST(DISTINCT {{'k': {pk_sql}, 'v': {expr}}}) FILTER (WHERE {expr} IS NOT NULL)])"
            )
        return f"{AGGREGATES[f.type]}({expr})"

    def _base_fanned_out(self) -> bool:
        """True when a used join can repeat base rows (one_to_many or many_to_many)."""
        return any(
            self.explore.joins[a].relationship in ("one_to_many", "many_to_many")
            for a in self.used_aliases
            if a in self.explore.joins
        )

    def _measure_ref(self, alias: str, fname: str) -> str:
        if alias not in self.aliases:
            raise SemanticQueryError(f"LookML references view '{alias}', which is not joined in this explore.")
        f = self._raw(alias, fname)
        return self.measure_sql(f, alias) if f.is_measure else self.dimension_sql(f, alias)

    def _measure_filter(self, alias: str, fname: str, value: str) -> str:
        f = self._raw(alias, fname)
        expr = self.dimension_sql(f, alias)
        if f.type == "yesno":
            return f"({expr}) = {'TRUE' if str(value).lower() == 'yes' else 'FALSE'}"
        return Compiler._string_condition(expr, str(value))

    # ------------------------------------------------------------------ joins

    def from_clause(self) -> str:
        base_view = self.model.views[self.explore.base_view]
        lines = [f"FROM {base_view.table_sql} AS {self.explore.base_alias}"]
        needed = self._needed_joins()
        for alias, join in self.explore.joins.items():
            if alias not in needed:
                continue
            view = self.model.views[join.view]
            on = self._substitute(join.sql_on, alias, self._dimension_ref) if join.sql_on else "TRUE"
            keyword = JOIN_SQL.get(join.type, "LEFT JOIN")
            if keyword == "CROSS JOIN":
                lines.append(f"{keyword} {view.table_sql} AS {alias}")
            else:
                lines.append(f"{keyword} {view.table_sql} AS {alias} ON {on}")
        return "\n".join(lines)

    def _needed_joins(self) -> Set[str]:
        needed = {a for a in self.used_aliases if a != self.explore.base_alias}
        changed = True
        while changed:
            changed = False
            for alias in list(needed):
                join = self.explore.joins[alias]
                for ref_alias, _ in REF.findall(join.sql_on):
                    if ref_alias in self.explore.joins and ref_alias not in needed:
                        needed.add(ref_alias)
                        changed = True
            # inner joins always apply, because they filter the base rows
            for alias, join in self.explore.joins.items():
                if join.type == "inner" and alias not in needed:
                    needed.add(alias)
                    changed = True
        return needed


def _timeframe_sql(base: str, timeframe: str) -> str:
    if timeframe in ("raw", "date"):
        return base
    if timeframe == "week":
        return f"CAST(DATE_TRUNC('week', {base}) AS DATE)"
    if timeframe == "month":
        return f"STRFTIME({base}, '%Y-%m')"
    if timeframe == "month_name":
        return f"STRFTIME({base}, '%B')"
    if timeframe == "quarter":
        return f"CONCAT(CAST(YEAR({base}) AS VARCHAR), '-Q', CAST(QUARTER({base}) AS VARCHAR))"
    if timeframe == "year":
        return f"YEAR({base})"
    if timeframe == "day_of_week":
        return f"STRFTIME({base}, '%A')"
    if timeframe == "day_of_month":
        return f"DAY({base})"
    raise SemanticQueryError(f"Timeframe '{timeframe}' is not supported.")


def _suggest(name: str, candidates) -> str:
    candidates = [c for c in candidates if c != name]
    matches = []
    if "." in name:
        short = name.split(".", 1)[1]
        matches = [c for c in candidates if c.endswith("." + short)]
    matches += [m for m in difflib.get_close_matches(name, candidates, n=3, cutoff=0.6) if m not in matches]
    return f" Did you mean: {', '.join(matches[:3])}?" if matches else ""
