"""Running Claude's SQL safely: the read-only check, timeouts, and result-size limits."""

import duckdb
import re
import threading

from .config import MAX_CELL_CHARS, MAX_RESULT_CHARS, MAX_RESULT_ROWS, QUERY_TIMEOUT_SECONDS
from .serialization import json_dumps, to_jsonable


READ_ONLY_ERROR = (
    "Only a single read-only query is allowed "
    "(SELECT/WITH/FROM/DESCRIBE/SHOW/SUMMARIZE, or EXPLAIN of one)."
)


# EXPLAIN [ANALYZE] [(options)] <statement> -- ANALYZE executes the inner statement.
EXPLAIN_PREFIX = re.compile(r"^\s*explain\s+(?:analyze\s+)?(?:\([^)]*\)\s*)?", re.IGNORECASE)


def check_read_only_sql(conn: duckdb.DuckDBPyConnection, sql: str) -> str | None:
    """Return an error message if SQL is not a single read-only statement, else None.

    Uses DuckDB's own parser, so the check sees the real statement type
    (e.g. `WITH ... INSERT` parses as INSERT). DESCRIBE/SHOW/SUMMARIZE/FROM-first
    all parse as SELECT. This is a second layer; attach production data READ_ONLY.
    """
    try:
        statements = conn.extract_statements(sql)
    except duckdb.Error as e:
        return f"SQL parse error: {e}"

    # DuckDB executes every statement in a batch, so reject stacked SQL.
    if len(statements) != 1:
        return READ_ONLY_ERROR

    statement_type = statements[0].type
    if statement_type == duckdb.StatementType.SELECT:
        return None
    if statement_type == duckdb.StatementType.EXPLAIN:
        inner_sql = EXPLAIN_PREFIX.sub("", sql, count=1)
        if inner_sql != sql and check_read_only_sql(conn, inner_sql) is None:
            return None
    return READ_ONLY_ERROR


def clip_long_text(value, max_cell_chars: int):
    """Clip long strings anywhere in a JSON-able value to max_cell_chars."""
    if isinstance(value, str) and len(value) > max_cell_chars:
        return value[:max_cell_chars] + f"…[clipped, {len(value):,} chars total]"
    if isinstance(value, list):
        return [clip_long_text(v, max_cell_chars) for v in value]
    if isinstance(value, dict):
        return {k: clip_long_text(v, max_cell_chars) for k, v in value.items()}
    return value


def cap_result_size(
    columns: list,
    rows: list,
    max_rows: int,
    max_chars: int,
    max_cell_chars: int = MAX_CELL_CHARS
) -> dict:
    """Build a query result that fits the row and character caps.

    `rows` may hold one extra row beyond `max_rows`; its presence means more rows exist.
    """
    more_rows_exist = len(rows) > max_rows
    rows = [clip_long_text(to_jsonable(row), max_cell_chars) for row in rows[:max_rows]]

    # Keep whole rows until the serialized payload would exceed the char budget.
    budget = max_chars - len(json_dumps(columns)) - 300  # room for keys and the note
    kept = []
    for row in rows:
        budget -= len(json_dumps(row)) + 1
        if budget < 0:
            break
        kept.append(row)

    result = {
        "success": True,
        "columns": columns,
        "rows": kept,
        "row_count": len(kept)
    }
    if more_rows_exist or len(kept) < len(rows):
        result["truncated"] = True
        if not kept:
            result["note"] = (
                "The first row alone exceeds the size limit. "
                "Select fewer columns or shorten values (e.g. left(col, 100))."
            )
        else:
            total = f"more than {max_rows}" if more_rows_exist else str(len(rows))
            result["note"] = (
                f"Showing {len(kept)} of {total} rows. "
                "Use aggregates, WHERE filters, or LIMIT to narrow the result."
            )
    return result


def tool_error(message: str, error_type: str, **extra) -> dict:
    """A failed tool result. `error` is what Claude sees; `error_type` is a stable class
    for traces (tool_calls.error_type), e.g. timeout, read_only_violation, table_not_found,
    or a DuckDB exception name such as BinderException."""
    return {"success": False, "error": message, "error_type": error_type, **extra}


def execute_query(
    conn: duckdb.DuckDBPyConnection,
    sql: str,
    max_rows: int = MAX_RESULT_ROWS,
    max_chars: int = MAX_RESULT_CHARS,
    max_cell_chars: int = MAX_CELL_CHARS,
    timeout_seconds: float | None = QUERY_TIMEOUT_SECONDS,
    params: list | None = None
) -> dict:
    """Execute SQL and return a size-capped result as dict. Pass timeout_seconds=None to disable."""
    read_only_error = check_read_only_sql(conn, sql)
    if read_only_error:
        error_type = "sql_parse_error" if read_only_error.startswith("SQL parse error") else "read_only_violation"
        return tool_error(read_only_error, error_type)

    # Cancel queries that run too long. The lock and `done` flag make sure a timer
    # firing just as the query finishes can't interrupt a later query on this connection.
    lock = threading.Lock()
    state = {"done": False, "timed_out": False}

    def interrupt():
        with lock:
            if not state["done"]:
                state["timed_out"] = True
                conn.interrupt()

    timer = None
    if timeout_seconds:
        timer = threading.Timer(timeout_seconds, interrupt)
        timer.daemon = True
        timer.start()

    try:
        try:
            # fetchmany streams: DuckDB stops producing rows once the cap is reached.
            rows = conn.execute(sql, params).fetchmany(max_rows + 1)
            columns = [desc[0] for desc in conn.description]
        finally:
            with lock:
                state["done"] = True
            if timer:
                timer.cancel()
        return cap_result_size(columns, rows, max_rows, max_chars, max_cell_chars)
    except duckdb.InterruptException:
        if not state["timed_out"]:
            raise  # interrupted by someone else (e.g. Ctrl-C), not our timeout
        return tool_error(
            f"Query timed out after {timeout_seconds}s and was cancelled. "
            "Narrow it: filter on date or partition columns, aggregate instead of "
            "scanning raw rows, or sample with USING SAMPLE.",
            "timeout",
            timed_out=True
        )
    except Exception as e:
        # DuckDB's exception class says what went wrong: BinderException (unknown
        # column), CatalogException (unknown table), ConversionException, ...
        return tool_error(str(e), type(e).__name__)
