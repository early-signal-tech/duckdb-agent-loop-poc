from types import SimpleNamespace

import duckdb
import pytest

from duckdb_agent.config import AgentConfig
from duckdb_agent.loop import run_agent
from duckdb_agent.sql import execute_query
from duckdb_agent.tools import run_tool, tool_result_block
from duckdb_agent.tracing import init_traces_db
from test_stop_reasons import FakeClient, response, text
from test_tools import ANSWER


def call(name, id="toolu_1", **tool_input):
    return SimpleNamespace(type="tool_use", id=id, name=name, input=tool_input)


@pytest.fixture
def conn():
    conn = duckdb.connect()
    conn.execute("CREATE TABLE t AS SELECT range AS a FROM range(500)")
    return conn


@pytest.fixture
def traces_conn(tmp_path):
    return init_traces_db(str(tmp_path / "traces.duckdb"))


def run(conn, traces_conn, *calls, **settings):
    """Run one turn of the given tool calls, then finish. Returns the tool_calls rows."""
    client = FakeClient(response("tool_use", *calls), response("end_turn", text("ok")))
    result = run_agent(conn, "analyze", client=client, traces_conn=traces_conn, config=AgentConfig(**settings))
    rows = traces_conn.execute("""
        SELECT tool_call_id, success, truncated, output_rows, error_type, error, result_chars,
               duration_ms, started_at
        FROM tool_calls WHERE run_id = ? ORDER BY id
    """, [result.run_id]).fetchall()
    return client, rows


# --- error classes on tool results ---

@pytest.mark.parametrize("sql, error_type", [
    ("DELETE FROM t", "read_only_violation"),
    ("SELEC 1", "sql_parse_error"),
    ("SELECT nope FROM t", "BinderException"),
    ("SELECT * FROM missing_table", "CatalogException"),
    ("SELECT 'abc'::INTEGER", "ConversionException"),
])
def test_query_errors_are_classified(conn, sql, error_type):
    result = execute_query(conn, sql)
    assert result["success"] is False
    assert result["error_type"] == error_type


def test_timeout_is_classified(conn):
    result = execute_query(conn, "SELECT count(*) FROM range(10_000_000_000)", timeout_seconds=0.2)
    assert result["error_type"] == "timeout"
    assert result["timed_out"] is True


@pytest.mark.parametrize("name, tool_input, error_type", [
    ("describe_table", {"table": "nope"}, "table_not_found"),
    ("sample_rows", {"table": "t", "n": "many"}, "invalid_input"),
    ("describe_table", {}, "invalid_input"),
    ("drop_everything", {}, "unknown_tool"),
])
def test_tool_errors_are_classified(conn, name, tool_input, error_type):
    assert run_tool(conn, name, tool_input)["error_type"] == error_type


def test_error_type_is_not_sent_to_claude(conn):
    # error_type is for traces; Claude gets only the message.
    result = execute_query(conn, "SELECT nope FROM t")
    block = tool_result_block("toolu_1", result)
    assert block["is_error"] is True
    assert block["content"] == result["error"]


# --- tool_calls rows ---

def test_successful_call_is_logged_in_full(conn, traces_conn):
    client, rows = run(conn, traces_conn, call("query_database", sql="SELECT 42 AS answer"))
    (tool_call_id, success, truncated, output_rows, error_type, error, result_chars, duration_ms, started_at), = rows
    assert (tool_call_id, success, truncated, output_rows, error_type, error) == ("toolu_1", True, False, 1, None, None)
    assert duration_ms >= 0 and started_at is not None
    # result_chars is the size of what Claude actually received.
    sent = client.sent_messages[1][-1]["content"][0]["content"]
    assert result_chars == len(sent)


def test_truncated_result_is_flagged(conn, traces_conn):
    _, rows = run(conn, traces_conn, call("query_database", sql="FROM t"), max_result_rows=10)
    success, truncated, output_rows = rows[0][1:4]
    assert (success, truncated, output_rows) == (True, True, 10)


def test_failed_call_logs_class_and_message(conn, traces_conn):
    _, rows = run(conn, traces_conn, call("query_database", sql="SELECT nope FROM t"))
    success, truncated, _, error_type, error = rows[0][1:6]
    assert (success, truncated, error_type) == (False, False, "BinderException")
    assert "nope" in error


def test_every_tool_is_logged(conn, traces_conn):
    _, rows = run(
        conn, traces_conn,
        call("list_tables", id="toolu_1"),
        call("describe_table", id="toolu_2", table="t"),
        call("sample_rows", id="toolu_3", table="t", n=3),
        call("describe_table", id="toolu_4", table="nope"),
    )
    assert [(row[0], row[1], row[4]) for row in rows] == [
        ("toolu_1", True, None),
        ("toolu_2", True, None),
        ("toolu_3", True, None),
        ("toolu_4", False, "table_not_found"),
    ]


def test_rejected_submit_answer_is_classified(conn, traces_conn):
    _, rows = run(
        conn, traces_conn,
        call("query_database", id="toolu_1", sql="SELECT 1"),
        call("submit_answer", id="toolu_2", **ANSWER),
        structured_answer=True,
    )
    assert rows[1][4] == "submit_not_alone"


def test_invalid_submit_answer_is_classified(conn, traces_conn):
    bad = {**ANSWER, "summary": ""}
    _, rows = run(conn, traces_conn, call("submit_answer", **bad), structured_answer=True)
    assert rows[0][4] == "invalid_answer"


def test_error_breakdown_query(conn, traces_conn):
    # The kind of question these columns are for: which errors happen most?
    run(
        conn, traces_conn,
        call("query_database", id="toolu_1", sql="SELECT nope FROM t"),
        call("query_database", id="toolu_2", sql="SELECT also_nope FROM t"),
        call("query_database", id="toolu_3", sql="DELETE FROM t"),
    )
    breakdown = traces_conn.execute(
        "SELECT error_type, count(*) FROM tool_calls WHERE NOT success GROUP BY ALL ORDER BY 2 DESC"
    ).fetchall()
    assert breakdown == [("BinderException", 2), ("read_only_violation", 1)]


def test_existing_traces_db_gains_tool_call_columns(tmp_path):
    path = str(tmp_path / "old.duckdb")
    old = duckdb.connect(path)
    old.execute("CREATE TABLE tool_calls (id INTEGER, run_id VARCHAR, iteration INTEGER, tool_name VARCHAR, "
                "tool_call_id VARCHAR, input_sql VARCHAR, output_rows INTEGER, output_summary VARCHAR)")
    old.close()
    traces_conn = init_traces_db(path)
    columns = {row[0] for row in traces_conn.execute("DESCRIBE tool_calls").fetchall()}
    assert {"started_at", "duration_ms", "success", "truncated", "result_chars", "error_type", "error"} <= columns
