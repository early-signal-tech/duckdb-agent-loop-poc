import json
from types import SimpleNamespace

import duckdb
import pytest

from duckdb_agent.config import AgentConfig
from duckdb_agent.loop import run_agent
from duckdb_agent.tools import describe_table, list_tables, MAX_SAMPLE_ROWS, resolve_table, run_tool, sample_rows, SUBMIT_ANSWER_TOOL, TOOLS
from duckdb_agent.tracing import init_traces_db
from test_stop_reasons import FakeClient, response, text


def call(name, id="toolu_1", **tool_input):
    return SimpleNamespace(type="tool_use", id=id, name=name, input=tool_input)


ANSWER = {
    "summary": "Revenue rose 10%.",
    "key_findings": ["Day 2 revenue: 110", "Day 1 revenue: 100"],
    "caveats": ["Only two days of data."],
    "supporting_sql": ["SELECT day, sum(revenue) FROM sales GROUP BY ALL"],
}


@pytest.fixture
def conn():
    conn = duckdb.connect()
    conn.execute("CREATE TABLE sales AS SELECT range AS id, range * 10 AS revenue FROM range(100)")
    conn.execute("COMMENT ON TABLE sales IS 'daily sales'")
    conn.execute("COMMENT ON COLUMN sales.revenue IS 'in USD'")
    conn.execute("CREATE VIEW big_sales AS SELECT * FROM sales WHERE revenue > 500")
    conn.execute("CREATE SCHEMA staging")
    conn.execute("CREATE TABLE staging.sales AS SELECT 1 AS id")
    conn.execute("CREATE TABLE staging.raw AS SELECT 'x' AS payload")
    return conn


# --- list_tables ---

def test_list_tables_returns_tables_and_views(conn):
    result = list_tables(conn)
    names = {(row[0], row[1], row[2]) for row in result["rows"]}
    assert names == {
        ("main", "big_sales", "view"),
        ("main", "sales", "table"),
        ("staging", "raw", "table"),
        ("staging", "sales", "table"),
    }
    assert result["current_schema"] == "main"
    sales = next(row for row in result["rows"] if row[:2] == ["main", "sales"])
    assert sales[3:] == [2, 100, "daily sales"]  # column_count, estimated_rows, comment


def test_list_tables_filters_by_schema(conn):
    result = list_tables(conn, "STAGING")
    assert {row[1] for row in result["rows"]} == {"raw", "sales"}


def test_list_tables_excludes_other_databases(conn, tmp_path):
    conn.execute(f"ATTACH '{tmp_path / 'other.duckdb'}' AS other")
    conn.execute("CREATE TABLE other.secret AS SELECT 1 AS x")
    assert "secret" not in {row[1] for row in list_tables(conn)["rows"]}


# --- resolve_table ---

@pytest.mark.parametrize("name, expected", [
    ("sales", ("main", "sales")),  # current schema wins over staging.sales
    ("SALES", ("main", "sales")),
    ("staging.sales", ("staging", "sales")),
    ("memory.staging.sales", ("staging", "sales")),
    ('"staging"."raw"', ("staging", "raw")),
    ("raw", ("staging", "raw")),  # only exists in one schema
    ("big_sales", ("main", "big_sales")),
])
def test_resolve_table(conn, name, expected):
    assert resolve_table(conn, name) == (expected, None)


def test_resolve_table_not_found(conn):
    resolved, error = resolve_table(conn, "nope")
    assert resolved is None
    assert error["error_type"] == "table_not_found"
    assert "list_tables" in error["error"]


def test_resolve_table_ambiguous_outside_current_schema(conn):
    conn.execute("CREATE SCHEMA a; CREATE SCHEMA b")
    conn.execute("CREATE TABLE a.dup AS SELECT 1; CREATE TABLE b.dup AS SELECT 1")
    resolved, error = resolve_table(conn, "dup")
    assert resolved is None
    assert error["error_type"] == "ambiguous_table"
    assert "a.dup" in error["error"] and "b.dup" in error["error"]


def test_resolve_table_cannot_be_used_for_injection(conn):
    resolved, error = resolve_table(conn, "sales; DROP TABLE sales")
    assert resolved is None
    assert conn.execute("SELECT count(*) FROM sales").fetchone()[0] == 100


# --- describe_table ---

def test_describe_table(conn):
    result = describe_table(conn, "sales")
    assert result["table"] == "main.sales"
    assert result["columns"] == ["column", "type", "nullable", "comment"]
    assert result["rows"] == [["id", "BIGINT", True, None], ["revenue", "BIGINT", True, "in USD"]]


def test_describe_view(conn):
    result = describe_table(conn, "big_sales")
    assert [row[0] for row in result["rows"]] == ["id", "revenue"]


def test_describe_unknown_table(conn):
    result = describe_table(conn, "nope")
    assert result["success"] is False


# --- sample_rows ---

def test_sample_rows_default_and_clamped(conn):
    assert sample_rows(conn, "sales")["row_count"] == 5
    assert sample_rows(conn, "sales", 1000)["row_count"] == MAX_SAMPLE_ROWS
    assert sample_rows(conn, "sales", 0)["row_count"] == 1
    assert sample_rows(conn, "sales", "3")["row_count"] == 3


def test_sample_rows_rejects_bad_n(conn):
    result = sample_rows(conn, "sales", "lots")
    assert result["success"] is False


def test_sample_rows_quotes_odd_names(conn):
    conn.execute('CREATE TABLE "my table" AS SELECT 1 AS "weird ""col"""')
    result = sample_rows(conn, "my table")
    assert result["success"] is True
    assert result["table"] == "main.my table"


# --- run_tool dispatch ---

def test_run_tool_dispatches_every_data_tool(conn):
    assert run_tool(conn, "list_tables", {})["success"]
    assert run_tool(conn, "describe_table", {"table": "sales"})["success"]
    assert run_tool(conn, "sample_rows", {"table": "sales", "n": 2})["row_count"] == 2
    assert run_tool(conn, "query_database", {"sql": "SELECT 1"})["rows"] == [[1]]


def test_run_tool_missing_required_input(conn):
    result = run_tool(conn, "describe_table", {})
    assert result == {"success": False, "error": "describe_table requires: table", "error_type": "invalid_input"}


def test_run_tool_unknown_tool(conn):
    assert "Unknown tool" in run_tool(conn, "drop_everything", {})["error"]


def test_submit_answer_is_not_a_data_tool(conn):
    assert "Unknown tool" in run_tool(conn, "submit_answer", ANSWER)["error"]


# --- the loop ---

def test_runs_without_tables_and_explores_with_tools(conn):
    client = FakeClient(
        response("tool_use", call("list_tables")),
        response("tool_use", call("describe_table", table="sales")),
        response("end_turn", text("done")),
    )
    result = run_agent(conn, "analyze", client=client)
    assert result.status == "completed"
    first_message = client.sent_messages[0][0]["content"]
    assert first_message == "<question>\nanalyze\n</question>"
    describe_result = client.sent_messages[2][-1]["content"][0]
    assert "in USD" in describe_result["content"]


def test_preloads_several_tables(conn):
    client = FakeClient(response("end_turn", text("ok")))
    run_agent(conn, "analyze", tables=["sales", "staging.raw"], client=client)
    first_message = client.sent_messages[0][0]["content"]
    assert '<table name="main.sales">' in first_message
    assert "  - revenue (BIGINT): in USD" in first_message
    assert '<table name="staging.raw">' in first_message


def test_unknown_preload_table_is_reported_in_message(conn):
    client = FakeClient(response("end_turn", text("ok")))
    run_agent(conn, "analyze", tables="nope", client=client)
    assert "Error: Table not found: nope" in client.sent_messages[0][0]["content"]


def test_submit_answer_offered_only_when_requested(conn):
    client = FakeClient(response("end_turn", text("ok")))
    run_agent(conn, "analyze", client=client)
    assert client.last_kwargs["tools"] == TOOLS

    client = FakeClient(response("end_turn", text("ok")))
    run_agent(conn, "analyze", client=client, config=AgentConfig(structured_answer=True))
    assert client.last_kwargs["tools"] == TOOLS + [SUBMIT_ANSWER_TOOL]


def test_submit_answer_ends_run_with_structured_answer(conn, tmp_path):
    traces_conn = init_traces_db(str(tmp_path / "traces.duckdb"))
    client = FakeClient(
        response("tool_use", call("query_database", sql="SELECT 1")),
        response("tool_use", call("submit_answer", id="toolu_2", **ANSWER)),
    )
    result = run_agent(conn, "analyze", client=client, config=AgentConfig(structured_answer=True), traces_conn=traces_conn)
    assert result.status == "completed"
    assert result.answer == ANSWER
    assert result.iterations == 2
    assert result.text.startswith("Revenue rose 10%.\n\nKey findings:\n- Day 2 revenue: 110")
    assert "Caveats:\n- Only two days of data." in result.text

    action, logged = traces_conn.execute(
        "SELECT action, claude_response FROM agent_traces WHERE run_id = ? ORDER BY id DESC LIMIT 1",
        [result.run_id]
    ).fetchone()
    assert action == "submit_answer"
    assert json.loads(logged) == ANSWER


def test_invalid_submit_answer_gets_error_and_can_retry(conn):
    client = FakeClient(
        response("tool_use", call("submit_answer", summary="", key_findings=[], caveats=[], supporting_sql=[])),
        response("tool_use", call("submit_answer", id="toolu_2", **ANSWER)),
    )
    result = run_agent(conn, "analyze", client=client, config=AgentConfig(structured_answer=True))
    assert result.answer == ANSWER
    error_block = client.sent_messages[1][-1]["content"][0]
    assert error_block["is_error"] is True
    assert "summary" in error_block["content"]


def test_submit_answer_with_other_tools_is_rejected_but_others_run(conn):
    client = FakeClient(
        response("tool_use", call("query_database", sql="SELECT 42"), call("submit_answer", id="toolu_2", **ANSWER)),
        response("tool_use", call("submit_answer", id="toolu_3", **ANSWER)),
    )
    result = run_agent(conn, "analyze", client=client, config=AgentConfig(structured_answer=True))
    assert result.answer == ANSWER
    query_block, submit_block = client.sent_messages[1][-1]["content"]
    assert "42" in query_block["content"]
    assert submit_block["is_error"] is True
    assert "on its own" in submit_block["content"]


def test_text_answer_in_structured_mode_has_no_answer(conn):
    client = FakeClient(response("end_turn", text("plain text")))
    result = run_agent(conn, "analyze", client=client, config=AgentConfig(structured_answer=True))
    assert result.status == "completed"
    assert result.text == "plain text"
    assert result.answer is None


def test_submit_answer_ignored_when_not_requested(conn):
    client = FakeClient(
        response("tool_use", call("submit_answer", **ANSWER)),
        response("end_turn", text("ok")),
    )
    result = run_agent(conn, "analyze", client=client)
    assert result.answer is None
    assert "Unknown tool" in client.sent_messages[1][-1]["content"][0]["content"]
