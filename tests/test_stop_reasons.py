from types import SimpleNamespace

import duckdb
import pytest

from duckdb_agent.config import AgentConfig
from duckdb_agent.loop import run_agent
from duckdb_agent.tracing import init_traces_db


def text(value):
    return SimpleNamespace(type="text", text=value)


def tool_use(sql, id="toolu_1"):
    return SimpleNamespace(type="tool_use", id=id, name="query_database", input={"sql": sql})


def response(stop_reason, *content, stop_details=None):
    return SimpleNamespace(stop_reason=stop_reason, content=list(content), stop_details=stop_details)


class FakeClient:
    """Returns scripted responses and records the messages sent on each call."""

    def __init__(self, *responses):
        self.responses = list(responses)
        self.sent_messages = []
        self.messages = SimpleNamespace(create=self.create)
        self.beta = SimpleNamespace(messages=self.messages)  # the loop calls client.beta.messages

    def create(self, **kwargs):
        self.last_kwargs = kwargs
        self.sent_messages.append(list(kwargs["messages"]))
        return self.responses.pop(0)


@pytest.fixture
def conn():
    conn = duckdb.connect()
    conn.execute("CREATE TABLE t AS SELECT 1 AS a")
    return conn


@pytest.fixture
def traces_conn(tmp_path):
    return init_traces_db(str(tmp_path / "traces.duckdb"))


def run(conn, client, traces_conn=None, **settings):
    return run_agent(conn, "analyze", tables="t", traces_conn=traces_conn, client=client, config=AgentConfig(**settings))


def test_end_turn_completes_and_joins_all_text_blocks(conn):
    client = FakeClient(response("end_turn", text("part one"), text("part two")))
    result = run(conn, client)
    assert result.status == "completed"
    assert result.text == "part one\npart two"
    assert result.iterations == 1


def test_tool_use_runs_tool_then_completes(conn):
    client = FakeClient(
        response("tool_use", text("checking"), tool_use("SELECT count(*) FROM t")),
        response("end_turn", text("one row")),
    )
    result = run(conn, client)
    assert result.status == "completed"
    assert result.iterations == 2
    tool_result_message = client.sent_messages[1][-1]
    assert tool_result_message["content"][0]["tool_use_id"] == "toolu_1"


def test_max_tokens_does_not_execute_truncated_tool_call(conn):
    # A truncated tool input can still look valid -- it must not run.
    client = FakeClient(response("max_tokens", text("partial"), tool_use("SELECT * FROM t")))
    result = run(conn, client)
    assert result.status == "max_tokens"
    assert result.text == "partial"
    assert len(client.sent_messages) == 1


def test_refusal_reports_category_and_skips_tools(conn):
    details = SimpleNamespace(category="cyber", explanation=None)
    client = FakeClient(response("refusal", tool_use("SELECT 1"), stop_details=details))
    result = run(conn, client)
    assert result.status == "refusal"
    assert "cyber" in result.text
    assert len(client.sent_messages) == 1


def test_pause_turn_resends_without_extra_user_message(conn):
    paused = response("pause_turn", text("working"))
    client = FakeClient(paused, response("end_turn", text("done")))
    result = run(conn, client)
    assert result.status == "completed"
    resumed = client.sent_messages[1]
    assert resumed[-1] == {"role": "assistant", "content": paused.content}


def test_max_tokens_default_and_override(conn):
    client = FakeClient(response("end_turn", text("ok")), response("end_turn", text("ok")))
    run(conn, client)
    assert client.last_kwargs["max_tokens"] == 16_000
    run(conn, client, max_tokens=4_000)
    assert client.last_kwargs["max_tokens"] == 4_000


def test_context_window_exceeded_is_its_own_status(conn):
    client = FakeClient(response("model_context_window_exceeded", text("cut")))
    assert run(conn, client).status == "context_window_exceeded"


def test_unknown_stop_reason_is_unexpected_stop(conn):
    client = FakeClient(response("something_new", text("?")))
    assert run(conn, client).status == "unexpected_stop"


def test_max_iterations(conn):
    client = FakeClient(
        *[response("tool_use", tool_use("SELECT 1", id=f"toolu_{i}")) for i in range(2)],
        response("end_turn", text("best answer so far")),
    )
    result = run(conn, client, max_iterations=2)
    assert result.status == "max_iterations"
    assert result.text == "best answer so far"


def test_status_and_stop_reason_are_traced(conn, traces_conn):
    client = FakeClient(response("max_tokens", text("partial")))
    result = run(conn, client, traces_conn=traces_conn)
    row = traces_conn.execute(
        "SELECT status, stop_reason FROM agent_runs WHERE run_id = ?", [result.run_id]
    ).fetchone()
    assert row == ("max_tokens", "max_tokens")
    actions = traces_conn.execute(
        "SELECT action FROM agent_traces WHERE run_id = ? ORDER BY id", [result.run_id]
    ).fetchall()
    assert actions == [("schema_lookup",), ("max_tokens",)]
