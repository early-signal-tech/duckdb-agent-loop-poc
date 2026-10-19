"""Run the agent from the command line against a DuckDB file or DuckLake catalog.

    uv run agent_harness.py                                  # the demo question on DuckLake prod
    uv run agent_harness.py --prompt "Which event types grew most last week?"
    uv run agent_harness.py --db analytics.duckdb --schema main --table orders --table customers
    uv run agent_harness.py --effort high --structured --quiet

The database is always attached READ_ONLY. Exit status is 0 when the run completes
normally and 1 otherwise (max iterations, API error, refusal, ...).
"""
import argparse
import sys

import duckdb
from dotenv import load_dotenv

from duckdb_agent import AgentConfig, configure_logging, init_traces_db, run_agent
from duckdb_agent.config import DEFAULT_EFFORT, DEFAULT_MAX_ITERATIONS, DEFAULT_MODEL, EFFORT_LEVELS

DEFAULT_PROMPT = (
    "Compare the most recent day of data against the prior day and run the most important analysis on it."
)


def parse_args(argv=None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Ask Claude a question about DuckDB data.")
    parser.add_argument("--prompt", default=DEFAULT_PROMPT, help="The question to answer.")
    parser.add_argument(
        "--table", action="append", dest="tables", metavar="TABLE",
        help="A table to focus on; its schema is preloaded. Repeatable. "
             "Default: events_rich_ml_gold when using the default --db; otherwise none "
             "(the agent finds tables with list_tables)."
    )
    parser.add_argument(
        "--no-table", action="store_true",
        help="Preload no schema, even with the default --db: the agent must find tables with list_tables."
    )
    parser.add_argument(
        "--db", default="ducklake:ducklake_prod",
        help="Database to attach read-only: a DuckDB file path or a ducklake: URI. "
             "Default: ducklake:ducklake_prod."
    )
    parser.add_argument("--schema", help="Schema to USE in that database. Default: gold for the default --db.")
    parser.add_argument("--traces-db", default="agent_traces.duckdb", help="Where to write traces.")
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--effort", default=DEFAULT_EFFORT, choices=EFFORT_LEVELS)
    parser.add_argument("--max-iterations", type=int, default=DEFAULT_MAX_ITERATIONS)
    parser.add_argument("--structured", action="store_true", help="End with a structured answer (submit_answer).")
    parser.add_argument(
        "--quiet", action="store_true",
        help="Log progress and metadata only (no SQL, thinking, or results), then print the answer."
    )
    args = parser.parse_args(argv)
    if args.no_table and args.tables:
        parser.error("--no-table and --table can't be used together")
    if args.db == parser.get_default("db"):
        args.schema = args.schema or "gold"
        if not args.no_table:
            args.tables = args.tables or ["events_rich_ml_gold"]
    return args


def connect(db: str, schema: str | None) -> duckdb.DuckDBPyConnection:
    """Attach the database READ_ONLY, so DuckDB rejects writes whatever SQL Claude sends."""
    conn = duckdb.connect()
    conn.execute(f"ATTACH '{db.replace(chr(39), chr(39) * 2)}' AS source (READ_ONLY)")
    conn.execute("USE source" + (f'."{schema}"' if schema else ""))
    return conn


def main(argv=None, client=None) -> int:
    args = parse_args(argv)
    load_dotenv()  # ANTHROPIC_API_KEY from .env
    configure_logging(verbose=not args.quiet)

    conn = connect(args.db, args.schema)
    traces_conn = init_traces_db(args.traces_db)
    config = AgentConfig(
        model=args.model,
        effort=args.effort,
        max_iterations=args.max_iterations,
        structured_answer=args.structured,
    )
    result = run_agent(
        conn, args.prompt, tables=args.tables, config=config, traces_conn=traces_conn, client=client
    )
    if args.quiet:
        print(result.text)  # verbose mode already logged it
    return 0 if result.status == "completed" else 1


if __name__ == "__main__":
    sys.exit(main())
