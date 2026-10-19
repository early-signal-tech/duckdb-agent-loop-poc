import json
from types import SimpleNamespace

import duckdb
import pytest

from duckdb_agent.config import AgentConfig, CONTEXT_MANAGEMENT, CONTEXT_MANAGEMENT_BETA
from duckdb_agent.loop import run_agent
from duckdb_agent.prompts import SYSTEM_PROMPT
from duckdb_agent.tools import TOOLS
from duckdb_agent.tracing import init_traces_db
from test_stop_reasons import FakeClient, response, text, tool_use


@pytest.fixture
def conn():
    conn = duckdb.connect()
    conn.execute("CREATE TABLE t AS SELECT 1 AS a")
    return conn


def run(conn, client, traces_conn=None, **settings):
    return run_agent(conn, "analyze", tables="t", client=client, traces_conn=traces_conn, config=AgentConfig(**settings))


def edit(cleared_tool_uses=4, cleared_input_tokens=30_000):
    return SimpleNamespace(
        type="clear_tool_uses_20250919",
        cleared_tool_uses=cleared_tool_uses,
        cleared_input_tokens=cleared_input_tokens,
    )


def test_context_editing_is_requested_on_every_call(conn):
    client = FakeClient(response("tool_use", tool_use("SELECT 1")), response("end_turn", text("ok")))
    calls = []
    original = client.create
    client.messages.create = lambda **kw: calls.append(kw) or original(**kw)
    run(conn, client)
    assert len(calls) == 2
    for call in calls:
        assert call["betas"] == [CONTEXT_MANAGEMENT_BETA]
        assert call["context_management"] == CONTEXT_MANAGEMENT


def test_context_editing_can_be_disabled(conn):
    client = FakeClient(response("end_turn", text("ok")))
    run(conn, client, context_editing=False)
    assert "betas" not in client.last_kwargs
    assert "context_management" not in client.last_kwargs


def test_clearing_settings():
    (strategy,) = CONTEXT_MANAGEMENT["edits"]
    assert strategy["type"] == "clear_tool_uses_20250919"
    # Below the 100K-token prompt size where Haiku 5.5 input pricing rises 5x.
    assert strategy["trigger"] == {"type": "input_tokens", "value": 80_000}
    assert strategy["clear_at_least"]["value"] > 0  # clears must be worth the cache rewrite
    assert "clear_tool_inputs" not in strategy  # SQL stays visible
    tool_names = {tool["name"] for tool in TOOLS}
    assert set(strategy["exclude_tools"]) <= tool_names
    assert "query_database" not in strategy["exclude_tools"]


def test_system_prompt_explains_clearing():
    assert "cleared" in SYSTEM_PROMPT


def test_applied_edits_are_logged(conn, tmp_path):
    traces_conn = init_traces_db(str(tmp_path / "traces.duckdb"))
    cleared = response("end_turn", text("ok"))
    cleared.context_management = SimpleNamespace(applied_edits=[edit()])
    result = run(conn, FakeClient(cleared), traces_conn=traces_conn)
    rows = traces_conn.execute(
        "SELECT iteration, claude_response FROM agent_traces WHERE run_id = ? AND action = 'context_edit'",
        [result.run_id]
    ).fetchall()
    assert len(rows) == 1
    iteration, details = rows[0]
    assert iteration == 1
    assert json.loads(details) == {
        "type": "clear_tool_uses_20250919",
        "cleared_tool_uses": 4,
        "cleared_input_tokens": 30_000,
    }


def test_no_edits_logs_nothing(conn, tmp_path):
    traces_conn = init_traces_db(str(tmp_path / "traces.duckdb"))
    plain = response("end_turn", text("ok"))
    plain.context_management = SimpleNamespace(applied_edits=[])
    result = run(conn, FakeClient(plain), traces_conn=traces_conn)
    count = traces_conn.execute(
        "SELECT count(*) FROM agent_traces WHERE run_id = ? AND action = 'context_edit'", [result.run_id]
    ).fetchone()[0]
    assert count == 0


def test_history_we_send_is_never_trimmed(conn):
    # Clearing happens server-side. Our own history must stay complete and
    # append-only, or Haiku 5.5's later thinking blocks become invalid (a 400).
    big_result_sql = "SELECT repeat('x', 400) AS v FROM range(40)"
    client = FakeClient(
        response("tool_use", tool_use(big_result_sql)),
        response("tool_use", tool_use("SELECT 1", id="toolu_2")),
        response("end_turn", text("ok")),
    )
    run(conn, client)
    first_result = client.sent_messages[2][2]["content"][0]
    assert first_result["tool_use_id"] == "toolu_1"
    assert "xxxx" in first_result["content"]
