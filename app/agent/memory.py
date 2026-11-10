"""Conversation memory that survives a server restart.

Each answered turn is stored in a small SQLite file, keyed by the session id (later, the Telegram chat id):
the question, the router's standalone version of it, the answer, and the semantic query (or raw SQL) the
answer used. Only the last SESSION_MAX_TURNS turns of a session are kept, and sessions with no activity
for SESSION_TTL_HOURS are deleted. A new connection is opened per call, so concurrent requests are safe.
"""

import json
import logging
import sqlite3
import time
from contextlib import closing
from pathlib import Path
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)

_SCHEMA = """
CREATE TABLE IF NOT EXISTS session_turns (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    session_id TEXT NOT NULL,
    created_at REAL NOT NULL,
    question TEXT NOT NULL,
    standalone_question TEXT,
    answer TEXT NOT NULL,
    semantic_query TEXT,
    sql TEXT
);
CREATE INDEX IF NOT EXISTS idx_session_turns_session ON session_turns (session_id, id);
"""


class SessionStore:
    """The last few turns of each conversation, in SQLite."""

    def __init__(self, path: Path, max_turns: int = 10, ttl_hours: float = 168.0):
        self.path = Path(path)
        self.max_turns = max(1, max_turns)
        self.ttl_seconds = ttl_hours * 3600
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with closing(self._connect()) as conn, conn:
            conn.executescript(_SCHEMA)
        self.expire()

    def _connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self.path, timeout=10)
        conn.row_factory = sqlite3.Row
        return conn

    def add_turn(
        self,
        session_id: str,
        question: str,
        answer: str,
        standalone_question: Optional[str] = None,
        semantic_query: Optional[Dict[str, Any]] = None,
        sql: Optional[str] = None,
    ) -> None:
        """Store one turn, keep only the session's last max_turns, and drop expired sessions."""
        with closing(self._connect()) as conn, conn:
            conn.execute(
                "INSERT INTO session_turns (session_id, created_at, question, standalone_question, answer, semantic_query, sql)"
                " VALUES (?, ?, ?, ?, ?, ?, ?)",
                (
                    session_id,
                    time.time(),
                    question,
                    standalone_question,
                    answer,
                    json.dumps(semantic_query, default=str) if semantic_query else None,
                    sql,
                ),
            )
            conn.execute(
                "DELETE FROM session_turns WHERE session_id = ? AND id NOT IN"
                " (SELECT id FROM session_turns WHERE session_id = ? ORDER BY id DESC LIMIT ?)",
                (session_id, session_id, self.max_turns),
            )
        self.expire()

    def turns(self, session_id: str) -> List[Dict[str, Any]]:
        """The session's stored turns, oldest first; empty for an unknown or expired session."""
        cutoff = time.time() - self.ttl_seconds
        with closing(self._connect()) as conn:
            rows = conn.execute(
                "SELECT * FROM session_turns WHERE session_id = ? ORDER BY id",
                (session_id,),
            ).fetchall()
        if not rows or rows[-1]["created_at"] < cutoff:
            return []
        return [
            {
                "question": row["question"],
                "standalone_question": row["standalone_question"],
                "answer": row["answer"],
                "semantic_query": json.loads(row["semantic_query"]) if row["semantic_query"] else None,
                "sql": row["sql"],
                "created_at": row["created_at"],
            }
            for row in rows
        ]

    def last_turn(self, session_id: str) -> Optional[Dict[str, Any]]:
        turns = self.turns(session_id)
        return turns[-1] if turns else None

    def has(self, session_id: str) -> bool:
        return bool(self.turns(session_id))

    def clear(self, session_id: str) -> bool:
        """Delete a session's turns; True if it had any."""
        with closing(self._connect()) as conn, conn:
            deleted = conn.execute("DELETE FROM session_turns WHERE session_id = ?", (session_id,)).rowcount
        return deleted > 0

    def expire(self) -> int:
        """Delete every session whose last turn is older than the time to live; returns the turns removed."""
        cutoff = time.time() - self.ttl_seconds
        with closing(self._connect()) as conn, conn:
            removed = conn.execute(
                "DELETE FROM session_turns WHERE session_id IN"
                " (SELECT session_id FROM session_turns GROUP BY session_id HAVING MAX(created_at) < ?)",
                (cutoff,),
            ).rowcount
        if removed:
            logger.info("Expired %d stored session turns", removed)
        return removed
