"""Agent instructions, modelled on a Looker Conversational Analytics data agent.

`agent_instructions.yaml` holds a persona, a business glossary, default filters and golden queries
(verified question and query pairs). The glossary and golden queries are indexed in Chroma and
retrieved per question; default filters are applied by the semantic layer to every query.
"""

import logging
from pathlib import Path
from typing import Any, Dict, List, Optional

import yaml
from pydantic import BaseModel, Field

from app.core.config import settings

logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parents[2]


class GlossaryEntry(BaseModel):
    term: str
    synonyms: List[str] = Field(default_factory=list)
    meaning: str
    field: Optional[str] = None  # dimension name without its view, e.g. category_name
    value: Optional[str] = None  # exact value of that dimension

    def describe(self) -> str:
        text = f'"{self.term}"'
        if self.synonyms:
            text += f" (also: {', '.join(self.synonyms)})"
        text += f": {self.meaning}"
        if self.field and self.value:
            text += f'. Filter {self.field} = "{self.value}"'
        return text


class DefaultFilter(BaseModel):
    explore: str
    field: str
    value: str


class GoldenQuery(BaseModel):
    question: str
    query: Dict[str, Any]


class AgentInstructions(BaseModel):
    persona: str = ""
    # One or two sentences on what the data covers; the router uses it to tell in-scope questions from others
    data_description: str = ""
    glossary: List[GlossaryEntry] = Field(default_factory=list)
    default_filters: List[DefaultFilter] = Field(default_factory=list)
    golden_queries: List[GoldenQuery] = Field(default_factory=list)

    def defaults_for(self, explore: str) -> List[DefaultFilter]:
        return [f for f in self.default_filters if f.explore == explore]


def instructions_path() -> Path:
    path = Path(settings.AGENT_INSTRUCTIONS_PATH)
    return path if path.is_absolute() else PROJECT_ROOT / path


def load_instructions() -> AgentInstructions:
    """Read the instructions file; a missing file means no instructions."""
    path = instructions_path()
    if not path.exists():
        logger.info("No agent instructions file at %s", path)
        return AgentInstructions()
    data = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    return AgentInstructions(**data)
