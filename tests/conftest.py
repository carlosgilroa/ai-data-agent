"""Test setup: conversation memory goes to a temporary SQLite file, never to data/sessions.db."""

import os
import tempfile
from pathlib import Path

os.environ.setdefault("SESSION_DB_PATH", str(Path(tempfile.mkdtemp(prefix="agent-sessions-")) / "sessions.db"))
