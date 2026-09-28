import anthropic
import duckdb
from dotenv import load_dotenv
from datetime import datetime
import uuid
import json


def json_dumps(obj) -> str:
    """Serialize query results, including DuckDB datetime/date/decimal values."""
    def default(o):
        if hasattr(o, "isoformat"):
            return o.isoformat()
        return str(o)

    return json.dumps(obj, default=default)

load_dotenv()

TOOLS = [
    {
        "name": "query_database",
        "description": "Execute a SQL query against the database",
        "input_schema": {
            "type": "object",
            "properties": {
                "sql": {
                    "type": "string",
                    "description": "SQL query to execute"
                }
            },
            "required": ["sql"]
        }
    }
]

TOOL_LABELS = {
    "query_database": "Query Database",
}

READ_ONLY_PREFIXES = ("select", "with", "describe", "show", "explain", "pragma")


def is_read_only_sql(sql: str) -> bool:
    """Check if SQL is read-only (no write/delete/drop operations)."""
    stripped = sql.strip()
    # DuckDB executes every statement in a batch, so reject stacked SQL.
    if ";" in stripped.rstrip(";"):
        return False
    return stripped.rstrip(";").strip().lstrip("(").strip().lower().startswith(
        READ_ONLY_PREFIXES
    )


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
    traces_conn.execute(
        "ALTER TABLE agent_runs ADD COLUMN IF NOT EXISTS table_name VARCHAR"
    )

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
    tool_name: str = None
):
    """Log an agent trace to the database."""
    traces_conn.execute("""
        INSERT INTO agent_traces
        (run_id, iteration, timestamp, action, sql_query, query_result, claude_response, tool_calls, tool_name)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
    """, [
        run_id,
        iteration,
        datetime.now(),
        action,
        sql_query,
        query_result,
        claude_response,
        tool_calls,
        tool_name
    ])


def log_tool_call(
    traces_conn: duckdb.DuckDBPyConnection,
    run_id: str,
    iteration: int,
    tool_name: str,
    tool_call_id: str,
    input_sql: str = None,
    output_rows: int = None,
    output_summary: str = None
):
    """Log a tool call execution to the database."""
    traces_conn.execute("""
        INSERT INTO tool_calls
        (run_id, iteration, tool_name, tool_call_id, input_sql, output_rows, output_summary)
        VALUES (?, ?, ?, ?, ?, ?, ?)
    """, [
        run_id,
        iteration,
        tool_name,
        tool_call_id,
        input_sql,
        output_rows,
        output_summary
    ])


def log_run_summary(
    traces_conn: duckdb.DuckDBPyConnection,
    run_id: str,
    table_name: str,
    prompt: str,
    final_response: str,
    total_iterations: int
):
    """Log the summary of an agent run."""
    traces_conn.execute("""
        INSERT INTO agent_runs
        (run_id, timestamp, initial_sql, table_name, prompt, final_response, total_iterations)
        VALUES (?, ?, ?, ?, ?, ?, ?)
    """, [
        run_id,
        datetime.now(),
        None,
        table_name,
        prompt,
        final_response,
        total_iterations
    ])


def execute_query(conn: duckdb.DuckDBPyConnection, sql: str) -> dict:
    """Execute SQL and return results as dict."""
    if not is_read_only_sql(sql):
        return {
            "success": False,
            "error": "Only read-only queries are allowed (SELECT/WITH/DESCRIBE/SHOW/EXPLAIN/PRAGMA)."
        }

    try:
        result = conn.execute(sql).fetchall()
        columns = [desc[0] for desc in conn.description]
        return {
            "success": True,
            "columns": columns,
            "rows": result,
            "row_count": len(result)
        }
    except Exception as e:
        return {
            "success": False,
            "error": str(e)
        }


def run_agent(
    conn: duckdb.DuckDBPyConnection,
    table_name: str,
    prompt: str,
    max_iterations: int = 25,
    traces_conn: duckdb.DuckDBPyConnection = None
) -> str:
    """Run the SQL-to-LLM agent loop."""
    client = anthropic.Anthropic()
    run_id = str(uuid.uuid4())

    print(f"\n{'='*60}")
    print(f"Starting agent loop (Run ID: {run_id})")
    print(f"{'='*60}")

    # Get table schema
    schema_result = execute_query(conn, f'DESCRIBE "{table_name}"')
    print(f"\n[Iteration 0] Schema lookup")
    print(f"  Columns: {schema_result.get('row_count', 'error')}")

    if traces_conn:
        log_trace(
            traces_conn,
            run_id,
            0,
            "schema_lookup",
            sql_query=f'DESCRIBE "{table_name}"',
            query_result=json_dumps(schema_result)
        )

    schema_str = "\n".join(
        f"  - {row[0]} ({row[1]})"
        for row in schema_result.get("rows", [])
    ) if schema_result.get("success") else "Error retrieving schema"

    messages = [
        {
            "role": "user",
            "content": f"You have access to a table named '{table_name}' with this schema:\n{schema_str}\n\n"
                      f"User prompt: {prompt}\n\n"
                      f"Use the query_database tool to explore and analyze this table as needed."
        }
    ]

    for iteration in range(1, max_iterations + 1):
        print(f"\n[Iteration {iteration}] Calling Claude...")

        response = client.messages.create(
            model="claude-haiku-4-5-20251001",
            max_tokens=1024,
            tools=TOOLS,
            messages=messages
        )

        # Check for tool use
        has_tool_use = any(block.type == "tool_use" for block in response.content)

        if not has_tool_use:
            # Extract text response
            text_response = next(
                (block.text for block in response.content if hasattr(block, "text")),
                "No response"
            )
            print(f"\n[Iteration {iteration}] Final response (no additional tool call)")
            print(f"{'='*60}")
            print(f"Response:\n{text_response}")
            print(f"{'='*60}")

            if traces_conn:
                log_trace(
                    traces_conn,
                    run_id,
                    iteration,
                    "claude_response",
                    claude_response=text_response
                )
                log_run_summary(
                    traces_conn,
                    run_id,
                    table_name,
                    prompt,
                    text_response,
                    iteration
                )

            return text_response

        # Handle tool calls
        print(f"  Claude requested tool use")
        tool_results = []
        for block in response.content:
            if block.type == "tool_use":
                sql = block.input.get("sql", "")
                tool_name = TOOL_LABELS.get(block.name, block.name)
                print(f"  Tool: {tool_name} ({block.id})")
                print(f"  Executing: {sql[:80]}...")
                result = execute_query(conn, sql)
                row_count = result.get('row_count', 0)
                print(f"  Result: {row_count} rows")
                tool_results.append({
                    "type": "tool_result",
                    "tool_use_id": block.id,
                    "content": str(result)
                })

                if traces_conn:
                    log_trace(
                        traces_conn,
                        run_id,
                        iteration,
                        "tool_call",
                        sql_query=sql,
                        query_result=json_dumps(result),
                        tool_calls=block.id,
                        tool_name=tool_name
                    )
                    log_tool_call(
                        traces_conn,
                        run_id,
                        iteration,
                        tool_name,
                        block.id,
                        input_sql=sql,
                        output_rows=row_count,
                        output_summary=f"Success" if result.get("success") else f"Error: {result.get('error')}"
                    )

        # Add to message history
        messages.append({"role": "assistant", "content": response.content})
        messages.append({"role": "user", "content": tool_results})

    print(f"\n[Final] Max iterations ({max_iterations}) reached")
    print(f"{'='*60}")

    if traces_conn:
        log_run_summary(
            traces_conn,
            run_id,
            table_name,
            prompt,
            "Max iterations reached",
            max_iterations
        )

    return "Max iterations reached"
