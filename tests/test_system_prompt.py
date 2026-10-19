import duckdb
import pytest

from duckdb_agent.config import AgentConfig, MAX_RESULT_CHARS, MAX_RESULT_ROWS, QUERY_TIMEOUT_SECONDS
from duckdb_agent.loop import run_agent
from duckdb_agent.prompts import SYSTEM_PROMPT
from test_stop_reasons import FakeClient, response, text, tool_use


@pytest.fixture
def conn():
    conn = duckdb.connect()
    conn.execute("CREATE TABLE events AS SELECT 1 AS id, DATE '2026-10-01' AS event_date")
    return conn


def run(conn, client, **settings):
    return run_agent(conn, "What changed yesterday?", tables="events", client=client, config=AgentConfig(**settings))


def test_system_prompt_is_sent_on_every_call(conn):
    client = FakeClient(response("tool_use", tool_use("SELECT 1")), response("end_turn", text("ok")))
    calls = []
    original_create = client.create
    client.messages.create = lambda **kw: calls.append(kw) or original_create(**kw)
    run(conn, client)
    assert len(calls) == 2
    assert all(call["system"][0]["text"] == SYSTEM_PROMPT for call in calls)


def test_system_prompt_states_the_real_limits():
    assert f"{MAX_RESULT_ROWS} rows" in SYSTEM_PROMPT
    assert f"{MAX_RESULT_CHARS:,} characters" in SYSTEM_PROMPT
    assert f"{QUERY_TIMEOUT_SECONDS} seconds" in SYSTEM_PROMPT


def test_system_prompt_has_no_per_run_content():
    # Must stay byte-identical across runs so it can be prompt-cached.
    assert "events" not in SYSTEM_PROMPT
    assert "2026" not in SYSTEM_PROMPT


def test_user_message_holds_table_schema_and_question(conn):
    client = FakeClient(response("end_turn", text("ok")))
    run(conn, client)
    first_message = client.sent_messages[0][0]["content"]
    assert '<table name="main.events">' in first_message
    assert "event_date (DATE)" in first_message
    assert "<question>\nWhat changed yesterday?\n</question>" in first_message
    assert "query_database" not in first_message  # instructions live in the system prompt


def test_system_prompt_can_be_overridden(conn):
    client = FakeClient(response("end_turn", text("ok")))
    run(conn, client, system_prompt="Custom prompt")
    assert client.last_kwargs["system"][0]["text"] == "Custom prompt"
