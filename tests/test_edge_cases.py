"""Paths the main suites don't reach: failure branches, races, and malformed input."""
import anthropic
import duckdb
import httpx2
import pytest

import duckdb_agent.sql as sql_module
from duckdb_agent.loop import describe_api_error, is_retryable_api_error
from duckdb_agent.sql import execute_query
from duckdb_agent.tools import describe_table, format_answer, list_tables, resolve_table, sample_rows, validate_answer
from test_api_errors import REQUEST, status_error
from test_tools import ANSWER


@pytest.fixture
def conn():
    conn = duckdb.connect()
    conn.execute("CREATE TABLE t AS SELECT range AS a FROM range(10)")
    # Binds fine, fails when run: the cast error happens per row at execution time.
    conn.execute("CREATE VIEW broken AS SELECT CAST(s AS INTEGER) AS n FROM (VALUES ('abc')) v(s)")
    return conn


# --- the timeout race guard ---

class RecordingConnection:
    """Wraps a DuckDB connection and records interrupt() calls."""

    def __init__(self, conn):
        self._conn = conn
        self.interrupts = 0

    def interrupt(self):
        self.interrupts += 1
        self._conn.interrupt()

    def __getattr__(self, name):
        return getattr(self._conn, name)


class CapturedTimer:
    """Stands in for threading.Timer: never fires on its own, so a test can fire it late."""
    created = []

    def __init__(self, interval, function):
        self.function = function
        self.daemon = False
        CapturedTimer.created.append(self)

    def start(self):
        pass

    def cancel(self):
        pass  # simulate the race: cancel() came too late to stop the timer thread


def test_late_timer_does_not_interrupt_later_queries(conn, monkeypatch):
    # A timer that fires just after its query finished must not interrupt the
    # connection, or it would cancel whatever query runs next.
    monkeypatch.setattr(sql_module.threading, "Timer", CapturedTimer)
    CapturedTimer.created.clear()
    recording = RecordingConnection(conn)
    assert execute_query(recording, "SELECT 1", timeout_seconds=30)["success"]

    (timer,) = CapturedTimer.created
    timer.function()  # fires after the query completed
    assert recording.interrupts == 0
    assert execute_query(recording, "SELECT 2", timeout_seconds=None)["rows"] == [[2]]


# --- failures after a table name resolves ---

def test_runtime_failure_in_view_is_reported(conn):
    for result in (sample_rows(conn, "broken"), execute_query(conn, "FROM broken")):
        assert result["success"] is False
        assert result["error_type"] == "ConversionException"
        assert "table" not in result


def test_sample_rows_unknown_table(conn):
    assert sample_rows(conn, "nope")["error_type"] == "table_not_found"


def test_catalog_lookup_failure_is_an_error_result(conn):
    conn.close()
    resolved, error = resolve_table(conn, "t")
    assert resolved is None
    assert error["error_type"] == "ConnectionException"
    assert describe_table(conn, "t")["success"] is False


def test_list_tables_failure_has_no_current_schema(conn):
    result = list_tables(conn, max_rows=10, max_chars=10, max_cell_chars=10)  # too small for any row
    assert result["success"] is True and result["row_count"] == 0
    conn.close()
    failed = list_tables(conn)
    assert failed["success"] is False
    assert "current_schema" not in failed


# --- submit_answer validation ---

@pytest.mark.parametrize("answer, problem", [
    ("just a string", "must be an object"),
    ({**ANSWER, "summary": "   "}, "non-empty summary"),
    ({**ANSWER, "key_findings": "one finding"}, "key_findings"),
    ({**ANSWER, "caveats": [1, 2]}, "caveats"),
    ({k: v for k, v in ANSWER.items() if k != "supporting_sql"}, "supporting_sql"),
])
def test_validate_answer_rejects(answer, problem):
    assert problem in validate_answer(answer)


def test_format_answer_omits_empty_sections():
    text = format_answer({**ANSWER, "key_findings": [], "caveats": []})
    assert text == ANSWER["summary"]


# --- API error helpers ---

def test_describe_api_error_without_request_id():
    resp = httpx2.Response(500, request=REQUEST)  # no request-id header
    error = anthropic.InternalServerError("boom", response=resp, body=None)
    assert describe_api_error(error) == "InternalServerError status=500: boom"


@pytest.mark.parametrize("error, retryable", [
    (anthropic.APIConnectionError(request=REQUEST), True),
    (anthropic.APITimeoutError(request=REQUEST), True),
    (status_error(anthropic.RateLimitError, 429), True),
    (status_error(anthropic.InternalServerError, 503), True),
    (status_error(anthropic.BadRequestError, 400), False),
    (status_error(anthropic.AuthenticationError, 401), False),
    # Neither a connection nor an HTTP status problem: the response didn't parse.
    (anthropic.APIResponseValidationError(httpx2.Response(200, request=REQUEST), body=None), False),
])
def test_is_retryable_api_error(error, retryable):
    assert is_retryable_api_error(error) is retryable


# --- known gaps, kept visible ---

@pytest.mark.xfail(strict=True, reason=(
    "Known gap: SELECT can read local files and URLs (read_csv, read_parquet, ...). "
    "Closing it needs SET enable_external_access = false, which may break DuckLake's "
    "access to its data files. When this starts passing, remove the xfail."
))
def test_select_cannot_read_local_files(conn, tmp_path):
    secret = tmp_path / "secret.csv"
    secret.write_text("password\nhunter2\n")
    result = execute_query(conn, f"SELECT * FROM read_csv('{secret}')")
    assert result["success"] is False
