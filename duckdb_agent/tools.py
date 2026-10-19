"""The tools Claude can call: definitions, catalog lookups, dispatch, and submit_answer."""

import duckdb

from .config import AgentConfig, DEFAULT_CONFIG, MAX_CELL_CHARS, MAX_RESULT_CHARS, MAX_RESULT_ROWS, QUERY_TIMEOUT_SECONDS
from .serialization import json_dumps
from .sql import execute_query, tool_error


# sample_rows returns this many rows by default, and never more than the max.
DEFAULT_SAMPLE_ROWS = 5


MAX_SAMPLE_ROWS = 20


def query_database_description(max_rows: int, max_chars: int, max_cell_chars: int, timeout_seconds: float) -> str:
    """query_database's description, which states the limits a run enforces."""
    return (
        "Execute a read-only DuckDB SQL query against the database. "
        f"Results are capped at {max_rows} rows and about {max_chars:,} characters, "
        f"and text values longer than {max_cell_chars} characters are clipped. "
        f"Queries running longer than {timeout_seconds}s are cancelled. "
        "When a result is cut off it includes \"truncated\": true. "
        "Aggregate, filter, or LIMIT in SQL instead of fetching raw rows."
    )


# Every tool is strict: the API guarantees tool inputs match input_schema (right
# types, required keys present, no extra keys). Strict schemas need
# additionalProperties: false on every object and don't support numeric/length
# constraints (minimum, maxLength, ...), so ranges are stated in descriptions and
# enforced in the tool code.
TOOLS = [
    {
        "name": "list_tables",
        "strict": True,
        "description": (
            "List the tables and views in the current database, with schema, type, column count, "
            "estimated row count, and comment. Also returns the current schema: tables in it can be "
            "referenced without a schema prefix. Use this first when no tables were provided."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "schema": {
                    "type": "string",
                    "description": "Only list tables in this schema. Omit to list all schemas."
                }
            },
            "additionalProperties": False
        }
    },
    {
        "name": "describe_table",
        "strict": True,
        "description": (
            "Get the columns of a table or view: name, type, nullability, and comment. "
            "Accepts a bare name (events), schema.table, or database.schema.table."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "table": {
                    "type": "string",
                    "description": "Table or view name"
                }
            },
            "required": ["table"],
            "additionalProperties": False
        }
    },
    {
        "name": "sample_rows",
        "strict": True,
        "description": (
            f"Return the first few rows of a table (default {DEFAULT_SAMPLE_ROWS}, max {MAX_SAMPLE_ROWS}) "
            "to see what values look like. These are rows in storage order, not a random sample; "
            "for a random sample use query_database with USING SAMPLE."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "table": {
                    "type": "string",
                    "description": "Table or view name"
                },
                "n": {
                    "type": "integer",
                    "description": f"Number of rows, 1-{MAX_SAMPLE_ROWS}"
                }
            },
            "required": ["table"],
            "additionalProperties": False
        }
    },
    {
        "name": "query_database",
        "strict": True,
        "description": query_database_description(
            MAX_RESULT_ROWS, MAX_RESULT_CHARS, MAX_CELL_CHARS, QUERY_TIMEOUT_SECONDS
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "sql": {
                    "type": "string",
                    "description": "SQL query to execute"
                }
            },
            "required": ["sql"],
            "additionalProperties": False
        }
    }
]


# Offered only when run_agent(structured_answer=True). Calling it ends the run, and its
# input becomes AgentResult.answer -- a fixed shape callers can rely on.
SUBMIT_ANSWER_TOOL = {
    "name": "submit_answer",
    "strict": True,
    "description": (
        "Submit your final answer and end the analysis. Call it on its own, after you have "
        "seen the results of the queries the answer relies on, not alongside other tools."
    ),
    "input_schema": {
        "type": "object",
        "properties": {
            "summary": {
                "type": "string",
                "description": "Direct answer to the question, in a few sentences."
            },
            "key_findings": {
                "type": "array",
                "items": {"type": "string"},
                "description": "The key numbers and facts supporting the answer, each taken from a query result."
            },
            "caveats": {
                "type": "array",
                "items": {"type": "string"},
                "description": "Data gaps, assumptions, and anything you could not verify. Empty if none."
            },
            "supporting_sql": {
                "type": "array",
                "items": {"type": "string"},
                "description": "The SQL queries whose results the answer relies on."
            }
        },
        "required": ["summary", "key_findings", "caveats", "supporting_sql"],
        "additionalProperties": False
    }
}


TOOL_LABELS = {
    "list_tables": "List Tables",
    "describe_table": "Describe Table",
    "sample_rows": "Sample Rows",
    "query_database": "Query Database",
    "submit_answer": "Submit Answer",
}


# Tables and views in the current database (the one selected with USE), so attached
# metadata catalogs and system tables stay out of view.
CATALOG_SQL = """
    SELECT schema_name, table_name, 'table' AS type, column_count, estimated_size, comment
    FROM duckdb_tables() WHERE database_name = current_database() AND NOT internal
    UNION ALL
    SELECT schema_name, view_name, 'view', NULL, NULL, comment
    FROM duckdb_views() WHERE database_name = current_database() AND NOT internal
"""


def quote_identifier(name: str) -> str:
    return '"' + name.replace('"', '""') + '"'


def resolve_table(conn: duckdb.DuckDBPyConnection, name: str) -> tuple[tuple[str, str] | None, dict | None]:
    """Find a table or view by bare, schema-qualified, or fully-qualified name.

    Returns ((schema, table), None) or (None, error result). Matching is case-insensitive,
    like DuckDB's own. A bare name found in several schemas resolves to the current
    schema, as DuckDB would; otherwise it is reported as ambiguous.
    """
    lookup = name.replace('"', "").strip()
    try:
        matches = conn.execute(f"""
            SELECT schema_name, table_name, schema_name = current_schema() AS in_current_schema
            FROM ({CATALOG_SQL})
            WHERE lower(?) IN (
                lower(table_name),
                lower(schema_name || '.' || table_name),
                lower(current_database() || '.' || schema_name || '.' || table_name)
            )
        """, [lookup]).fetchall()
    except duckdb.Error as e:
        return None, tool_error(f"Could not look up table {name!r}: {e}", type(e).__name__)

    if len(matches) > 1:
        matches = [m for m in matches if m[2]] or matches
    if not matches:
        return None, tool_error(
            f"Table not found: {name}. Call list_tables to see the available tables.", "table_not_found"
        )
    if len(matches) > 1:
        options = ", ".join(f"{schema}.{table}" for schema, table, _ in matches)
        return None, tool_error(f"Table name {name} is ambiguous. Use one of: {options}.", "ambiguous_table")
    schema, table, _ = matches[0]
    return (schema, table), None


def list_tables(conn: duckdb.DuckDBPyConnection, schema: str | None = None, **limits) -> dict:
    """Tables and views in the current database, optionally limited to one schema.
    `limits` go to execute_query."""
    sql = f"""
        SELECT schema_name AS schema, table_name AS name, type, column_count,
               estimated_size AS estimated_rows, comment
        FROM ({CATALOG_SQL})
    """
    params = None
    if schema:
        sql += " WHERE lower(schema_name) = lower(?)"
        params = [schema]
    result = execute_query(conn, sql + " ORDER BY schema, name", params=params, **limits)
    if result["success"]:
        result["current_schema"] = conn.execute("SELECT current_schema()").fetchone()[0]
    return result


def describe_table(
    conn: duckdb.DuckDBPyConnection,
    table: str,
    max_rows: int = MAX_RESULT_ROWS,
    max_chars: int = MAX_RESULT_CHARS,
    max_cell_chars: int = MAX_CELL_CHARS
) -> dict:
    """Columns of a table or view: name, type, nullability, comment."""
    resolved, error = resolve_table(conn, table)
    if error:
        return error
    schema, name = resolved
    result = execute_query(
        conn,
        """
        SELECT column_name AS "column", data_type AS type, is_nullable AS nullable, comment
        FROM duckdb_columns()
        WHERE database_name = current_database() AND schema_name = ? AND table_name = ?
        ORDER BY column_index
        """,
        max_rows=max_rows,
        max_chars=max_chars,
        max_cell_chars=max_cell_chars,
        params=[schema, name]
    )
    if result["success"]:
        result["table"] = f"{schema}.{name}"
    return result


def sample_rows(conn: duckdb.DuckDBPyConnection, table: str, n=DEFAULT_SAMPLE_ROWS, **limits) -> dict:
    """The first n rows of a table (storage order, not random). `limits` go to execute_query."""
    try:
        n = min(max(int(n), 1), MAX_SAMPLE_ROWS)
    except (TypeError, ValueError):
        return tool_error(f"n must be an integer from 1 to {MAX_SAMPLE_ROWS}.", "invalid_input")
    resolved, error = resolve_table(conn, table)
    if error:
        return error
    schema, name = resolved
    result = execute_query(conn, f"SELECT * FROM {quote_identifier(schema)}.{quote_identifier(name)} LIMIT {n}", **limits)
    if result["success"]:
        result["table"] = f"{schema}.{name}"
    return result


def run_tool(
    conn: duckdb.DuckDBPyConnection,
    name: str,
    tool_input: dict,
    config: "AgentConfig | None" = None
) -> dict:
    """Run one of the data tools by name, with the config's result limits and timeout.
    Never raises for bad input; returns an error result.

    Strict schemas mean the real API won't send bad input, but this check stays: it
    covers fake clients, proxies that drop `strict`, and future non-strict tools.
    """
    config = config or DEFAULT_CONFIG
    limits = {
        "max_rows": config.max_result_rows,
        "max_chars": config.max_result_chars,
        "max_cell_chars": config.max_cell_chars,
    }
    schema = next((tool["input_schema"] for tool in TOOLS if tool["name"] == name), None)
    if schema is None:
        return tool_error(f"Unknown tool: {name}", "unknown_tool")
    missing = [key for key in schema.get("required", []) if not tool_input.get(key)]
    if missing:
        return tool_error(f"{name} requires: {', '.join(missing)}", "invalid_input")

    if name == "list_tables":
        return list_tables(conn, tool_input.get("schema"), **limits)
    if name == "describe_table":
        return describe_table(conn, tool_input["table"], **limits)
    timeout = {"timeout_seconds": config.query_timeout_seconds}
    if name == "sample_rows":
        return sample_rows(conn, tool_input["table"], tool_input.get("n", DEFAULT_SAMPLE_ROWS), **limits, **timeout)
    return execute_query(conn, tool_input["sql"], **limits, **timeout)


def validate_answer(answer) -> str | None:
    """Return an error if a submit_answer input doesn't have the required shape, else None."""
    if not isinstance(answer, dict):
        return "submit_answer input must be an object."
    if not isinstance(answer.get("summary"), str) or not answer["summary"].strip():
        return "submit_answer requires a non-empty summary."
    for key in ("key_findings", "caveats", "supporting_sql"):
        value = answer.get(key)
        if not isinstance(value, list) or not all(isinstance(item, str) for item in value):
            return f"submit_answer requires {key} as a list of strings."
    return None


def format_answer(answer: dict) -> str:
    """Render a submitted answer as plain text for AgentResult.text and the run summary."""
    sections = [answer["summary"].strip()]
    for title, key in (("Key findings", "key_findings"), ("Caveats", "caveats")):
        if answer[key]:
            sections.append(f"{title}:\n" + "\n".join(f"- {item}" for item in answer[key]))
    return "\n\n".join(sections)


def build_tools(config: AgentConfig) -> list[dict]:
    """The tools for a run: query_database's description states the run's limits, and
    submit_answer is added when structured_answer is on."""
    description = query_database_description(
        config.max_result_rows, config.max_result_chars, config.max_cell_chars, config.query_timeout_seconds
    )
    tools = [
        {**tool, "description": description} if tool["name"] == "query_database" else tool
        for tool in TOOLS
    ]
    return tools + [SUBMIT_ANSWER_TOOL] if config.structured_answer else tools


def tool_result_block(tool_use_id: str, result: dict) -> dict:
    """Build a tool_result block: JSON data on success, error text with is_error on failure."""
    if not result["success"]:
        return {
            "type": "tool_result",
            "tool_use_id": tool_use_id,
            "content": result["error"],
            "is_error": True
        }
    payload = {key: value for key, value in result.items() if key != "success"}
    return {
        "type": "tool_result",
        "tool_use_id": tool_use_id,
        "content": json_dumps(payload)
    }
