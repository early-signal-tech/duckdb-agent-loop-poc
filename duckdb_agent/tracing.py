"""The traces database: schema and writers for runs, API calls, tool calls, and events."""

import duckdb
from datetime import datetime

from .usage import USAGE_FIELDS


def init_traces_db(traces_db_path: str = "agent_traces.duckdb") -> duckdb.DuckDBPyConnection:
    """Initialize traces database with schema."""
    traces_conn = duckdb.connect(traces_db_path)

    traces_conn.execute("CREATE SEQUENCE IF NOT EXISTS seq_traces START 1")

    traces_conn.execute("""
        CREATE TABLE IF NOT EXISTS agent_runs (
            run_id VARCHAR,
            timestamp TIMESTAMP,
            initial_sql VARCHAR,
            table_name VARCHAR,
            prompt VARCHAR,
            final_response VARCHAR,
            total_iterations INTEGER,
            PRIMARY KEY (run_id)
        )
    """)
    traces_conn.execute("""
        CREATE TABLE IF NOT EXISTS agent_traces (
            id INTEGER PRIMARY KEY DEFAULT nextval('seq_traces'),
            run_id VARCHAR,
            iteration INTEGER,
            timestamp TIMESTAMP,
            action VARCHAR,
            sql_query VARCHAR,
            query_result VARCHAR,
            claude_response VARCHAR,
            tool_calls VARCHAR,
            tool_name VARCHAR
        )
    """)
    traces_conn.execute(
        "ALTER TABLE agent_traces ADD COLUMN IF NOT EXISTS tool_name VARCHAR"
    )
    # Claude's summarized reasoning, on rows with action = 'thinking' (one per response).
    traces_conn.execute(
        "ALTER TABLE agent_traces ADD COLUMN IF NOT EXISTS thinking VARCHAR"
    )
    traces_conn.execute(
        "ALTER TABLE agent_runs ADD COLUMN IF NOT EXISTS table_name VARCHAR"
    )
    traces_conn.execute(
        "ALTER TABLE agent_runs ADD COLUMN IF NOT EXISTS status VARCHAR"
    )
    traces_conn.execute(
        "ALTER TABLE agent_runs ADD COLUMN IF NOT EXISTS stop_reason VARCHAR"
    )
    traces_conn.execute(
        "ALTER TABLE agent_runs ADD COLUMN IF NOT EXISTS error VARCHAR"
    )
    # The run's settings: model on its own for easy filtering, the full AgentConfig as
    # JSON (system prompt as a hash), so runs can be compared by configuration.
    traces_conn.execute(
        "ALTER TABLE agent_runs ADD COLUMN IF NOT EXISTS model VARCHAR"
    )
    traces_conn.execute(
        "ALTER TABLE agent_runs ADD COLUMN IF NOT EXISTS config JSON"
    )
    # Run timing and totals across all API calls. `timestamp` is kept for older
    # queries; it's written at the end of the run, so it equals ended_at.
    for column, column_type in (
        ("started_at", "TIMESTAMP"),
        ("ended_at", "TIMESTAMP"),
        ("duration_ms", "BIGINT"),
        ("api_calls", "INTEGER"),
        ("input_tokens", "BIGINT"),
        ("cache_creation_input_tokens", "BIGINT"),
        ("cache_read_input_tokens", "BIGINT"),
        ("output_tokens", "BIGINT"),
        ("estimated_cost_usd", "DOUBLE"),
    ):
        traces_conn.execute(f"ALTER TABLE agent_runs ADD COLUMN IF NOT EXISTS {column} {column_type}")

    # One row per API call (successful or failed), for per-iteration token, latency,
    # and cost breakdowns. request_id is what Anthropic support needs to look up a call.
    traces_conn.execute("CREATE SEQUENCE IF NOT EXISTS seq_api_calls START 1")
    traces_conn.execute("""
        CREATE TABLE IF NOT EXISTS api_calls (
            id INTEGER PRIMARY KEY DEFAULT nextval('seq_api_calls'),
            run_id VARCHAR,
            iteration INTEGER,
            started_at TIMESTAMP,
            latency_ms BIGINT,
            model VARCHAR,
            request_id VARCHAR,
            stop_reason VARCHAR,
            input_tokens BIGINT,
            cache_creation_input_tokens BIGINT,
            cache_read_input_tokens BIGINT,
            output_tokens BIGINT,
            estimated_cost_usd DOUBLE,
            error VARCHAR
        )
    """)

    traces_conn.execute("CREATE SEQUENCE IF NOT EXISTS seq_tool_calls START 1")

    traces_conn.execute("""
        CREATE TABLE IF NOT EXISTS tool_calls (
            id INTEGER PRIMARY KEY DEFAULT nextval('seq_tool_calls'),
            run_id VARCHAR,
            iteration INTEGER,
            tool_name VARCHAR,
            tool_call_id VARCHAR,
            input_sql VARCHAR,
            output_rows INTEGER,
            output_summary VARCHAR
        )
    """)
    # Full detail per tool call: timing, outcome, and how big the result sent to Claude
    # was (result_chars -- every result is resent on each later turn, so this is what
    # drives context growth). error_type is a stable class; error is the full message.
    for column, column_type in (
        ("started_at", "TIMESTAMP"),
        ("duration_ms", "BIGINT"),
        ("success", "BOOLEAN"),
        ("truncated", "BOOLEAN"),
        ("result_chars", "INTEGER"),
        ("error_type", "VARCHAR"),
        ("error", "VARCHAR"),
    ):
        traces_conn.execute(f"ALTER TABLE tool_calls ADD COLUMN IF NOT EXISTS {column} {column_type}")

    return traces_conn


def log_trace(
    traces_conn: duckdb.DuckDBPyConnection,
    run_id: str,
    iteration: int,
    action: str,
    sql_query: str = None,
    query_result: str = None,
    claude_response: str = None,
    tool_calls: str = None,
    tool_name: str = None,
    thinking: str = None
):
    """Log an agent trace to the database."""
    traces_conn.execute("""
        INSERT INTO agent_traces
        (run_id, iteration, timestamp, action, sql_query, query_result, claude_response, tool_calls, tool_name, thinking)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
    """, [
        run_id,
        iteration,
        datetime.now(),
        action,
        sql_query,
        query_result,
        claude_response,
        tool_calls,
        tool_name,
        thinking
    ])


def log_tool_call(
    traces_conn: duckdb.DuckDBPyConnection,
    run_id: str,
    iteration: int,
    tool_name: str,
    tool_call_id: str,
    input_sql: str = None,
    output_rows: int = None,
    output_summary: str = None,
    started_at: datetime = None,
    duration_ms: int = None,
    result: dict = None,
    result_chars: int = None
):
    """Log a tool call execution to the database. `result` is the tool's result dict;
    success, truncation, and error details are taken from it."""
    result = result or {}
    traces_conn.execute("""
        INSERT INTO tool_calls
        (run_id, iteration, tool_name, tool_call_id, input_sql, output_rows, output_summary,
         started_at, duration_ms, success, truncated, result_chars, error_type, error)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
    """, [
        run_id,
        iteration,
        tool_name,
        tool_call_id,
        input_sql,
        output_rows,
        output_summary,
        started_at,
        duration_ms,
        result.get("success"),
        bool(result.get("truncated")) if result else None,
        result_chars,
        result.get("error_type"),
        result.get("error")
    ])


def log_run_summary(
    traces_conn: duckdb.DuckDBPyConnection,
    run_id: str,
    table_name: str,
    prompt: str,
    final_response: str,
    total_iterations: int,
    status: str = None,
    stop_reason: str = None,
    error: str = None,
    model: str = None,
    config: str = None,
    started_at: datetime = None,
    duration_ms: int = None,
    api_calls: int = None,
    usage: dict = None,
    estimated_cost_usd: float = None
):
    """Log the summary of an agent run. `config` is AgentConfig.to_json(); `usage` is
    AgentResult.usage (token totals by USAGE_FIELDS name)."""
    ended_at = datetime.now()
    usage = usage or {}
    traces_conn.execute("""
        INSERT INTO agent_runs
        (run_id, timestamp, initial_sql, table_name, prompt, final_response, total_iterations,
         status, stop_reason, error, model, config,
         started_at, ended_at, duration_ms, api_calls, input_tokens, cache_creation_input_tokens,
         cache_read_input_tokens, output_tokens, estimated_cost_usd)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
    """, [
        run_id,
        ended_at,
        None,
        table_name,
        prompt,
        final_response,
        total_iterations,
        status,
        stop_reason,
        error,
        model,
        config,
        started_at,
        ended_at,
        duration_ms,
        api_calls,
        usage.get("input_tokens"),
        usage.get("cache_creation_input_tokens"),
        usage.get("cache_read_input_tokens"),
        usage.get("output_tokens"),
        estimated_cost_usd
    ])


def log_api_call(
    traces_conn: duckdb.DuckDBPyConnection,
    run_id: str,
    iteration: int,
    started_at: datetime,
    latency_ms: int,
    model: str,
    request_id: str = None,
    stop_reason: str = None,
    usage=None,
    estimated_cost_usd: float = None,
    error: str = None
):
    """Log one API call. `usage` is the response's usage object (None for failed calls)."""
    traces_conn.execute("""
        INSERT INTO api_calls
        (run_id, iteration, started_at, latency_ms, model, request_id, stop_reason, input_tokens,
         cache_creation_input_tokens, cache_read_input_tokens, output_tokens, estimated_cost_usd, error)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
    """, [
        run_id,
        iteration,
        started_at,
        latency_ms,
        model,
        request_id,
        stop_reason,
        # Unused cache fields come back as None; store 0 so sums work.
        *((getattr(usage, name, None) or 0) if usage is not None else None for name in USAGE_FIELDS),
        estimated_cost_usd,
        error
    ])
