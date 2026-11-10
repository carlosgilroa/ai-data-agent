"""Conversation memory in SQLite: survives a restart, keeps the last turns, expires idle sessions."""

import time

from app.agent.memory import SessionStore
from app.agent.runner import agent_runner
from app.schemas.query import QueryRequest


def test_turns_survive_a_restart(tmp_path):
    db = tmp_path / "sessions.db"
    store = SessionStore(db)
    query = {"explore": "e", "fields": ["v.m"], "filters": {"v.d": "x"}}
    store.add_turn("chat-1", "Spend on x in September?", "1,000 COP", standalone_question="Spend on x in September 2026?",
                   semantic_query=query, sql="SELECT 1")

    restarted = SessionStore(db)  # a new process opening the same file
    turn = restarted.last_turn("chat-1")
    assert turn["question"] == "Spend on x in September?"
    assert turn["standalone_question"] == "Spend on x in September 2026?"
    assert turn["semantic_query"] == query
    assert turn["sql"] == "SELECT 1"
    assert restarted.last_turn("another-chat") is None


def test_only_the_last_turns_are_kept(tmp_path):
    store = SessionStore(tmp_path / "s.db", max_turns=3)
    for i in range(5):
        store.add_turn("chat", f"q{i}", f"a{i}")
    assert [t["question"] for t in store.turns("chat")] == ["q2", "q3", "q4"]


def test_idle_sessions_expire(tmp_path, monkeypatch):
    store = SessionStore(tmp_path / "s.db", ttl_hours=1)
    store.add_turn("old", "q", "a")
    store.add_turn("fresh", "q", "a")
    real_time = time.time
    monkeypatch.setattr(time, "time", lambda: real_time() + 2 * 3600)
    assert store.turns("old") == []  # an expired session reads as empty even before it is deleted
    store.add_turn("fresh", "q2", "a2")  # any write deletes expired sessions; "fresh" is active again
    import sqlite3
    with sqlite3.connect(store.path) as conn:
        remaining = {row[0] for row in conn.execute("SELECT DISTINCT session_id FROM session_turns")}
    assert remaining == {"fresh"}
    assert [t["question"] for t in store.turns("fresh")] == ["q", "q2"]


def test_clear_removes_a_session(tmp_path):
    store = SessionStore(tmp_path / "s.db")
    store.add_turn("chat", "q", "a")
    assert store.clear("chat") is True
    assert store.clear("chat") is False
    assert not store.has("chat")


def test_follow_up_works_after_a_restart(tmp_path, monkeypatch):
    """The first turn is answered, the server "restarts" (a new store on the same file), and the follow-up still
    reaches the router with the previous standalone question and semantic query."""
    from app.agent.router import intent_router
    from app.schemas.query import RouterDecision
    from tests.test_agent import _FakeLLM

    db = tmp_path / "sessions.db"
    monkeypatch.setattr(agent_runner, "memory", SessionStore(db))
    first_query = {
        "explore": "fct_transactions",
        "fields": ["fct_transactions.total_spend_cop"],
        "filters": [{"field": "fct_transactions.transaction_month", "value": "2026-09"}],
    }
    fake = _FakeLLM(first_query)
    monkeypatch.setattr(agent_runner, "api_key", "test-key")
    monkeypatch.setattr(agent_runner, "llm_with_tools", fake)
    monkeypatch.setattr(agent_runner, "_get_or_create_llm", lambda model, tools: (fake, fake))
    router_calls = []

    def classify(query, has_session_context=False, previous=None):
        router_calls.append(previous)
        standalone = "How much did I spend in August 2026?" if previous else "How much did I spend in September 2026?"
        return RouterDecision(intent="simple_query", reasoning="test", standalone_question=standalone,
                              measures=["total spend"])

    monkeypatch.setattr(intent_router, "classify", classify)
    agent_runner.run_query(QueryRequest(query="How much did I spend in September?", session_id="chat-42"))

    monkeypatch.setattr(agent_runner, "memory", SessionStore(db))  # restart
    fake.calls = 0
    agent_runner.run_query(QueryRequest(query="and in August?", session_id="chat-42"))

    assert router_calls[0] is None
    assert router_calls[1]["question"] == "How much did I spend in September 2026?"
    assert router_calls[1]["semantic_query"]["filters"] == {"fct_transactions.transaction_month": "2026-09"}
    turns = agent_runner.memory.turns("chat-42")
    assert [t["question"] for t in turns] == ["How much did I spend in September?", "and in August?"]
    assert turns[1]["standalone_question"] == "How much did I spend in August 2026?"
