import json

import duckdb
import pytest

from duckdb_agent.config import MAX_RESULT_ROWS
from duckdb_agent.loop import run_agent
from duckdb_agent.sql import execute_query
from duckdb_agent.tools import tool_result_block
from test_stop_reasons import FakeClient, response, text


@pytest.fixture
def conn():
    return duckdb.connect()


def test_small_result_is_not_marked_truncated(conn):
    result = execute_query(conn, "SELECT * FROM range(5)")
    assert result["row_count"] == 5
    assert "truncated" not in result


def test_exactly_max_rows_is_not_truncated(conn):
    result = execute_query(conn, "SELECT * FROM range(10)", max_rows=10)
    assert result["row_count"] == 10
    assert "truncated" not in result


def test_row_cap_marks_truncated_with_note(conn):
    result = execute_query(conn, "SELECT * FROM range(1000)")
    assert result["row_count"] == MAX_RESULT_ROWS
    assert result["truncated"] is True
    assert f"more than {MAX_RESULT_ROWS}" in result["note"]


def test_huge_table_returns_quickly_without_materializing(conn):
    result = execute_query(conn, "SELECT i, md5(i::VARCHAR) FROM range(500000000) t(i)")
    assert result["row_count"] == MAX_RESULT_ROWS


def test_char_cap_keeps_whole_rows_within_budget(conn):
    result = execute_query(conn, "SELECT repeat('x', 400) AS s FROM range(150)", max_chars=5_000)
    assert 0 < result["row_count"] < 150
    assert result["truncated"] is True
    assert "of 150 rows" in result["note"]
    assert len(tool_result_block("t", result)["content"]) <= 5_000


def test_long_text_is_clipped(conn):
    result = execute_query(conn, "SELECT repeat('x', 5000) AS s", max_cell_chars=100)
    value = result["rows"][0][0]
    assert value.startswith("x" * 100)
    assert "clipped, 5,000 chars total" in value


def test_row_too_large_for_budget_explains_what_to_do(conn):
    result = execute_query(conn, "SELECT [repeat('x', 400) FOR i IN range(100)] AS l", max_chars=2_000)
    assert result["row_count"] == 0
    assert result["truncated"] is True
    assert "first row" in result["note"]


def test_truncation_info_reaches_claude(conn):
    block = tool_result_block("t", execute_query(conn, "SELECT * FROM range(1000)"))
    payload = json.loads(block["content"])
    assert payload["truncated"] is True and "note" in payload
    assert "success" not in payload


def test_schema_lookup_is_not_capped(conn):
    columns = ", ".join(f"1 AS c{i}" for i in range(300))
    conn.execute(f"CREATE TABLE wide AS SELECT {columns}")
    client = FakeClient(response("end_turn", text("ok")))
    run_agent(conn, "analyze", tables="wide", client=client)
    first_message = client.sent_messages[0][0]["content"]
    assert "c299" in first_message
