import json

import duckdb
import pytest

from duckdb_agent.loop import run_agent
from duckdb_agent.serialization import json_dumps
from duckdb_agent.sql import execute_query
from duckdb_agent.tools import tool_result_block
from test_stop_reasons import FakeClient, response, text, tool_use


@pytest.fixture
def conn():
    return duckdb.connect()


def test_json_dumps_handles_duckdb_types(conn):
    result = execute_query(conn, """
        SELECT 1.50::DECIMAL(10,2) AS d, DATE '2026-10-14' AS dt,
               TIMESTAMP '2026-10-14 10:00' AS ts, 'héllo' AS s, [1, 2] AS l,
               {'a': 1} AS st, 'abc'::BLOB AS b, 'nan'::DOUBLE AS n, NULL AS x,
               INTERVAL 3 DAY AS iv, '00000000-0000-0000-0000-000000000001'::UUID AS u
    """)
    payload = json.loads(json_dumps(result))  # must be valid, strict JSON
    assert payload["rows"] == [[
        1.5, "2026-10-14", "2026-10-14T10:00:00", "héllo", [1, 2],
        {"a": 1}, "616263", "nan", None, "3 days, 0:00:00",
        "00000000-0000-0000-0000-000000000001",
    ]]
    assert "héllo" in json_dumps(result)  # not escaped to é


def test_success_result_is_json_without_is_error(conn):
    block = tool_result_block("toolu_1", execute_query(conn, "SELECT 1 AS a"))
    assert "is_error" not in block
    assert json.loads(block["content"]) == {"columns": ["a"], "rows": [[1]], "row_count": 1}


def test_failed_query_sets_is_error_with_message(conn):
    block = tool_result_block("toolu_1", execute_query(conn, "SELECT * FROM missing_table"))
    assert block["is_error"] is True
    assert "missing_table" in block["content"]


def test_blocked_query_sets_is_error(conn):
    block = tool_result_block("toolu_1", execute_query(conn, "DROP TABLE x"))
    assert block["is_error"] is True
    assert "read-only" in block["content"]


def test_unknown_tool_returns_error_result(conn):
    conn.execute("CREATE TABLE t AS SELECT 1 AS a")
    bad_call = tool_use("SELECT 1")
    bad_call.name = "drop_everything"
    client = FakeClient(response("tool_use", bad_call), response("end_turn", text("ok")))
    run_agent(conn, "analyze", tables="t", client=client)
    sent = client.sent_messages[1][-1]["content"][0]
    assert sent["is_error"] is True
    assert "Unknown tool" in sent["content"]
