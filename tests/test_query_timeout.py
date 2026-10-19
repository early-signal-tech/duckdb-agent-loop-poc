import threading
import time

import duckdb
import pytest

from duckdb_agent.sql import execute_query
from duckdb_agent.tools import tool_result_block

SLOW_SQL = "SELECT count(*) FROM range(5000000000) t(i) WHERE md5(i::VARCHAR) LIKE '%zz%'"


@pytest.fixture
def conn():
    return duckdb.connect()


def test_slow_query_is_cancelled_and_reported(conn):
    start = time.monotonic()
    result = execute_query(conn, SLOW_SQL, timeout_seconds=0.5)
    assert time.monotonic() - start < 5
    assert result["success"] is False
    assert result["timed_out"] is True
    assert "timed out after 0.5s" in result["error"]
    assert tool_result_block("t", result)["is_error"] is True


def test_slow_order_by_is_cancelled_even_with_row_cap(conn):
    # ORDER BY must sort everything before the first row, so fetchmany alone can't stop it.
    sql = "SELECT md5(i::VARCHAR) AS m FROM range(2000000000) t(i) ORDER BY m"
    assert execute_query(conn, sql, timeout_seconds=0.5)["timed_out"] is True


def test_connection_still_works_after_timeout(conn):
    execute_query(conn, SLOW_SQL, timeout_seconds=0.5)
    assert execute_query(conn, "SELECT 42 AS x")["rows"] == [[42]]


def test_timer_does_not_leak_into_later_queries(conn):
    # The first query finishes well before its timeout. The timer must be
    # cancelled, so it can't fire during the next (slower) query.
    execute_query(conn, "SELECT 1", timeout_seconds=0.3)
    result = execute_query(conn, "SELECT count(*) FROM range(300000000)", timeout_seconds=None)
    assert result["success"] is True
    time.sleep(0.4)
    assert execute_query(conn, "SELECT 1", timeout_seconds=None)["success"] is True


def test_timeout_disabled_with_none(conn):
    assert execute_query(conn, "SELECT 1", timeout_seconds=None)["success"] is True


def test_external_interrupt_is_not_swallowed_as_timeout(conn):
    # An interrupt that isn't ours (e.g. Ctrl-C) should propagate, not be reported as a timeout.
    threading.Timer(0.3, conn.interrupt).start()
    with pytest.raises(duckdb.InterruptException):
        execute_query(conn, SLOW_SQL, timeout_seconds=None)
