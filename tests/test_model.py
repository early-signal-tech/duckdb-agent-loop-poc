from types import SimpleNamespace

import duckdb
import pytest

from duckdb_agent.config import AgentConfig, DEFAULT_EFFORT, DEFAULT_MODEL
from duckdb_agent.loop import run_agent
from duckdb_agent.tracing import init_traces_db
from test_stop_reasons import FakeClient, response, text, tool_use


def thinking(signature="sig_1"):
    # Haiku 5.5 omits thinking text by default: the block has only a signature.
    return SimpleNamespace(type="thinking", thinking="", signature=signature)


@pytest.fixture
def conn():
    conn = duckdb.connect()
    conn.execute("CREATE TABLE t AS SELECT 1 AS a")
    return conn


def run(conn, client, traces_conn=None, **settings):
    return run_agent(conn, "analyze", tables="t", client=client, traces_conn=traces_conn, config=AgentConfig(**settings))


def test_default_model_is_haiku_5_5(conn):
    client = FakeClient(response("end_turn", text("ok")))
    run(conn, client)
    assert DEFAULT_MODEL == "claude-haiku-5-5"
    assert client.last_kwargs["model"] == "claude-haiku-5-5"


def test_default_effort_is_sent_explicitly(conn):
    client = FakeClient(response("end_turn", text("ok")))
    run(conn, client)
    assert client.last_kwargs["output_config"] == {"effort": DEFAULT_EFFORT}


def test_model_and_effort_can_be_overridden(conn):
    client = FakeClient(response("end_turn", text("ok")))
    run(conn, client, model="claude-opus-5-5", effort="high")
    assert client.last_kwargs["model"] == "claude-opus-5-5"
    assert client.last_kwargs["output_config"] == {"effort": "high"}


def test_adaptive_thinking_with_summaries(conn):
    client = FakeClient(response("end_turn", text("ok")))
    run(conn, client)
    assert client.last_kwargs["thinking"] == {"type": "adaptive", "display": "summarized"}


def test_no_sampling_params(conn):
    # Haiku 5.5 returns a 400 for non-default temperature/top_p/top_k.
    client = FakeClient(response("end_turn", text("ok")))
    run(conn, client)
    assert not {"temperature", "top_p", "top_k"} & client.last_kwargs.keys()


def test_thinking_blocks_are_skipped_in_answer_text(conn):
    client = FakeClient(response("end_turn", thinking(), text("the answer")))
    result = run(conn, client)
    assert result.text == "the answer"


def test_thinking_blocks_are_passed_back_unchanged(conn):
    block = thinking()
    client = FakeClient(
        response("tool_use", block, tool_use("SELECT 1")),
        response("end_turn", text("ok")),
    )
    run(conn, client)
    assistant_turn = client.sent_messages[1][1]
    assert assistant_turn["role"] == "assistant"
    assert assistant_turn["content"][0] is block


def test_no_request_ends_with_an_assistant_turn(conn):
    # Haiku 5.5 rejects assistant prefill (a trailing assistant turn). Requests end
    # with a user turn, or the wrap-up system message that follows one.
    client = FakeClient(
        response("tool_use", tool_use("SELECT 1")),
        response("tool_use", tool_use("SELECT 2", id="toolu_2")),
        response("end_turn", text("summary")),
    )
    run(conn, client, max_iterations=2)
    assert [messages[-1]["role"] for messages in client.sent_messages] == ["user", "user", "system"]
    assert client.sent_messages[-1][-2]["role"] == "user"


def thinking_with(summary):
    return SimpleNamespace(type="thinking", thinking=summary, signature="sig")


def test_thinking_summaries_are_logged(conn, tmp_path):
    traces_conn = init_traces_db(str(tmp_path / "traces.duckdb"))
    client = FakeClient(
        response("tool_use", thinking_with("Check the date range first."), tool_use("SELECT 1")),
        response("end_turn", thinking_with("Compare the two days."), thinking_with("Then answer."), text("ok")),
    )
    result = run(conn, client, traces_conn=traces_conn)
    rows = traces_conn.execute(
        "SELECT iteration, thinking FROM agent_traces WHERE run_id = ? AND action = 'thinking' ORDER BY id",
        [result.run_id]
    ).fetchall()
    assert rows == [(1, "Check the date range first."), (2, "Compare the two days.\n\nThen answer.")]


def test_empty_thinking_is_not_logged(conn, tmp_path):
    traces_conn = init_traces_db(str(tmp_path / "traces.duckdb"))
    client = FakeClient(response("end_turn", thinking(), text("ok")))
    result = run(conn, client, traces_conn=traces_conn)
    count = traces_conn.execute(
        "SELECT count(*) FROM agent_traces WHERE run_id = ? AND action = 'thinking'", [result.run_id]
    ).fetchone()[0]
    assert count == 0


def test_existing_traces_db_gains_thinking_column(tmp_path):
    path = str(tmp_path / "old.duckdb")
    old = duckdb.connect(path)
    old.execute("CREATE TABLE agent_traces (id INTEGER, run_id VARCHAR, iteration INTEGER, timestamp TIMESTAMP, "
                "action VARCHAR, sql_query VARCHAR, query_result VARCHAR, claude_response VARCHAR, tool_calls VARCHAR)")
    old.close()
    traces_conn = init_traces_db(path)
    columns = [row[0] for row in traces_conn.execute("DESCRIBE agent_traces").fetchall()]
    assert "thinking" in columns
