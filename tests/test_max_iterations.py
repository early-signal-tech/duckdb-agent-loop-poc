import anthropic
import duckdb
import pytest

from duckdb_agent.config import AgentConfig
from duckdb_agent.loop import run_agent
from duckdb_agent.prompts import MAX_ITERATIONS_WRAP_UP_PROMPT
from duckdb_agent.tracing import init_traces_db
from test_api_errors import RaisingClient, status_error
from test_stop_reasons import FakeClient, response, text, tool_use


@pytest.fixture
def conn():
    conn = duckdb.connect()
    conn.execute("CREATE TABLE t AS SELECT 1 AS a")
    return conn


def looping_client(n_tool_turns, final):
    """Claude keeps calling tools for n turns, then returns `final` for the wrap-up call."""
    turns = [response("tool_use", tool_use("SELECT 1", id=f"toolu_{i}")) for i in range(n_tool_turns)]
    return RaisingClient(*turns, final)


def run(conn, client, traces_conn=None, max_iterations=2):
    return run_agent(conn, "analyze", tables="t", config=AgentConfig(max_iterations=max_iterations), traces_conn=traces_conn, client=client)


def test_wrap_up_call_disables_tools_and_returns_answer(conn):
    client = looping_client(2, response("end_turn", text("Partial answer: 1 row.")))
    result = run(conn, client)
    assert result.status == "max_iterations"
    assert result.text == "Partial answer: 1 row."
    assert result.iterations == 3  # 2 tool iterations + the wrap-up call
    assert client.last_kwargs["tool_choice"] == {"type": "none"}


def test_wrap_up_prompt_is_a_system_message_after_tool_results(conn):
    client = looping_client(2, response("end_turn", text("done")))
    run(conn, client)
    *_, tool_results, wrap_up = client.sent_messages[-1]
    assert tool_results["role"] == "user"
    assert all(block["type"] == "tool_result" for block in tool_results["content"])
    assert wrap_up == {"role": "system", "content": MAX_ITERATIONS_WRAP_UP_PROMPT}


def test_normal_iterations_do_not_disable_tools(conn):
    client = FakeClient(response("tool_use", tool_use("SELECT 1")), response("end_turn", text("ok")))
    run(conn, client)
    assert "tool_choice" not in client.last_kwargs


def test_wrap_up_after_pause_turn_follows_paused_assistant_turn(conn):
    # pause_turn only happens mid server-tool use, where a system message may follow
    # the assistant turn directly.
    client = RaisingClient(
        response("pause_turn", text("working")),
        response("end_turn", text("summary")),
    )
    result = run(conn, client, max_iterations=1)
    assert result.status == "max_iterations"
    sent = client.sent_messages[-1]
    assert sent[-2]["role"] == "assistant"
    assert sent[-1] == {"role": "system", "content": MAX_ITERATIONS_WRAP_UP_PROMPT}


def test_wrap_up_truncated_reports_max_tokens(conn):
    result = run(conn, looping_client(2, response("max_tokens", text("cut off"))))
    assert result.status == "max_tokens"


def test_wrap_up_api_error_reports_api_error(conn):
    result = run(conn, looping_client(2, status_error(anthropic.InternalServerError, 500)))
    assert result.status == "api_error"
    assert result.iterations == 3


def test_wrap_up_is_traced(conn, tmp_path):
    traces_conn = init_traces_db(str(tmp_path / "traces.duckdb"))
    result = run(conn, looping_client(2, response("end_turn", text("summary"))), traces_conn)
    row = traces_conn.execute(
        "SELECT status, final_response, total_iterations FROM agent_runs WHERE run_id = ?",
        [result.run_id]
    ).fetchone()
    assert row == ("max_iterations", "summary", 3)
    last_action = traces_conn.execute(
        "SELECT action, iteration FROM agent_traces WHERE run_id = ? ORDER BY id DESC LIMIT 1",
        [result.run_id]
    ).fetchone()
    assert last_action == ("max_iterations", 3)
