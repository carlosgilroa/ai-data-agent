"""Typed, in-memory representation of the LookML project (views, fields and explores).

The model is parsed once with `lkml` and is what the compiler reads to turn a semantic query
into SQL. Field names follow Looker's convention: `<view>.<field>`, and a dimension_group
expands into one field per timeframe (`fct_transactions.transaction_month`).
"""

import hashlib
import logging
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional

import lkml

logger = logging.getLogger(__name__)

SCHEMA_MACRO = re.compile(r"`?@\{schema_name\}\.([a-zA-Z0-9_]+)`?")


def strip_schema_macros(sql: str) -> str:
    """Replace `@{schema_name}.table` with the bare table name registered in DuckDB."""
    return SCHEMA_MACRO.sub(r"\1", sql).strip().rstrip(";").strip()


def _is_yes(value) -> bool:
    return str(value).strip().lower() in ("yes", "true")


@dataclass
class Field:
    """A selectable LookML field: a dimension, one timeframe of a dimension_group, or a measure."""

    view: str
    name: str
    kind: str  # "dimension" | "measure"
    type: str  # string, number, yesno, time, sum, count, number (measure), ...
    sql: Optional[str]
    description: str = ""
    hidden: bool = False
    primary_key: bool = False
    filters: List[Dict[str, str]] = field(default_factory=list)  # measure filters
    case: Optional[dict] = None  # CASE dimension definition
    group: Optional[str] = None  # dimension_group name for timeframe fields
    timeframe: Optional[str] = None  # month, date, year...

    @property
    def id(self) -> str:
        return f"{self.view}.{self.name}"

    @property
    def is_measure(self) -> bool:
        return self.kind == "measure"

    @property
    def is_time(self) -> bool:
        return self.timeframe is not None


@dataclass
class View:
    name: str
    description: str
    table_sql: str  # bare table name, or "(<derived sql>)"
    fields: Dict[str, Field]
    dimension_groups: Dict[str, dict]

    @property
    def is_derived(self) -> bool:
        return self.table_sql.startswith("(")


@dataclass
class Join:
    name: str  # alias used in field names
    view: str  # underlying view (join `from:`), usually equal to name
    type: str
    relationship: str
    sql_on: str


@dataclass
class Explore:
    name: str
    label: str
    description: str
    base_alias: str  # alias of the base view in field names (view_name or explore name)
    base_view: str  # underlying view (`from:`)
    joins: Dict[str, Join]

    def alias_to_view(self) -> Dict[str, str]:
        mapping = {self.base_alias: self.base_view}
        mapping.update({alias: j.view for alias, j in self.joins.items()})
        return mapping


class SemanticModel:
    """All views and explores of a LookML project directory."""

    def __init__(self, lookml_dir: Path):
        self.lookml_dir = Path(lookml_dir)
        self.views: Dict[str, View] = {}
        self.explores: Dict[str, Explore] = {}
        self.fingerprint = ""
        self.load()

    # ------------------------------------------------------------------ loading

    def lookml_files(self) -> List[Path]:
        return sorted(self.lookml_dir.glob("**/*.lkml"))

    def compute_fingerprint(self) -> str:
        digest = hashlib.sha256()
        for path in self.lookml_files():
            stat = path.stat()
            digest.update(f"{path}:{stat.st_mtime_ns}:{stat.st_size}".encode())
        return digest.hexdigest()

    def is_stale(self) -> bool:
        return self.compute_fingerprint() != self.fingerprint

    def load(self) -> None:
        self.views.clear()
        self.explores.clear()
        for path in sorted((self.lookml_dir / "views").glob("**/*.view.lkml")):
            parsed = lkml.load(path.read_text(encoding="utf-8"))
            for raw in parsed.get("views", []):
                view = self._parse_view(raw)
                self.views[view.name] = view
        for path in sorted((self.lookml_dir / "models").glob("**/*.model.lkml")):
            parsed = lkml.load(path.read_text(encoding="utf-8"))
            for raw in parsed.get("explores", []):
                explore = self._parse_explore(raw)
                self.explores[explore.name] = explore
        self.fingerprint = self.compute_fingerprint()
        logger.info("Loaded LookML model: %d views, %d explores", len(self.views), len(self.explores))

    @staticmethod
    def _parse_view(raw: dict) -> View:
        name = raw["name"]
        if raw.get("sql_table_name"):
            table_sql = strip_schema_macros(raw["sql_table_name"])
        elif raw.get("derived_table", {}).get("sql"):
            table_sql = f"({strip_schema_macros(raw['derived_table']['sql'])})"
        else:
            table_sql = name

        fields: Dict[str, Field] = {}
        for d in raw.get("dimensions", []):
            fields[d["name"]] = Field(
                view=name,
                name=d["name"],
                kind="dimension",
                type=d.get("type", "string"),
                sql=d.get("sql"),
                description=d.get("description", ""),
                hidden=_is_yes(d.get("hidden", "no")),
                primary_key=_is_yes(d.get("primary_key", "no")),
                case=d.get("case"),
            )

        groups: Dict[str, dict] = {}
        for g in raw.get("dimension_groups", []):
            groups[g["name"]] = g
            for tf in g.get("timeframes", []):
                fname = f"{g['name']}_{tf}"
                fields[fname] = Field(
                    view=name,
                    name=fname,
                    kind="dimension",
                    type="time",
                    sql=g.get("sql"),
                    description=f"{g.get('description', '')} ({tf})".strip(),
                    hidden=_is_yes(g.get("hidden", "no")),
                    group=g["name"],
                    timeframe=tf,
                )

        for m in raw.get("measures", []):
            filters = [
                {k: v}
                for group in m.get("filters__all", [])
                for item in (group if isinstance(group, list) else [group])
                for k, v in item.items()
            ]
            fields[m["name"]] = Field(
                view=name,
                name=m["name"],
                kind="measure",
                type=m.get("type", "count"),
                sql=m.get("sql"),
                description=m.get("description", ""),
                hidden=_is_yes(m.get("hidden", "no")),
                filters=filters,
            )

        return View(
            name=name,
            description=raw.get("description", ""),
            table_sql=table_sql,
            fields=fields,
            dimension_groups=groups,
        )

    @staticmethod
    def _parse_explore(raw: dict) -> Explore:
        name = raw["name"]
        base_view = raw.get("from") or raw.get("view_name") or name
        base_alias = raw.get("view_name") or name
        joins = {}
        for j in raw.get("joins", []):
            joins[j["name"]] = Join(
                name=j["name"],
                view=j.get("from") or j["name"],
                type=j.get("type", "left_outer"),
                relationship=j.get("relationship", "many_to_one"),
                sql_on=j.get("sql_on", ""),
            )
        return Explore(
            name=name,
            label=raw.get("label", name),
            description=raw.get("description", ""),
            base_alias=base_alias,
            base_view=base_view,
            joins=joins,
        )

    # ------------------------------------------------------------------ lookups

    def explore_fields(self, explore_name: str, include_hidden: bool = False) -> List[Field]:
        """Every field reachable in an explore, keyed by the explore's aliases."""
        explore = self.explores[explore_name]
        result = []
        for alias, view_name in explore.alias_to_view().items():
            view = self.views.get(view_name)
            if view is None:
                continue
            for f in view.fields.values():
                if f.hidden and not include_hidden:
                    continue
                result.append(f if alias == view_name else _aliased(f, alias))
        return result

    def explores_for_view(self, view_name: str) -> List[str]:
        return [
            e.name for e in self.explores.values() if view_name in e.alias_to_view().values()
        ]

    def referenced_tables(self) -> List[str]:
        return referenced_tables(self.lookml_dir)


def referenced_tables(lookml_dir: Path) -> List[str]:
    """Physical tables referenced by views through sql_table_name or derived-table SQL."""
    tables = set()
    for path in sorted((Path(lookml_dir) / "views").glob("**/*.view.lkml")):
        tables.update(SCHEMA_MACRO.findall(path.read_text(encoding="utf-8")))
    return sorted(tables)


def _aliased(f: Field, alias: str) -> Field:
    return Field(**{**f.__dict__, "view": alias})
