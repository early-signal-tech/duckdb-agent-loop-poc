"""A SQL-to-LLM agent: Claude answers questions about DuckDB data by writing and running
read-only SQL in a loop, with every step traced to a DuckDB database.

    from duckdb_agent import AgentConfig, run_agent, init_traces_db

    result = run_agent(conn, "What changed yesterday?", tables="events",
                       config=AgentConfig(), traces_conn=init_traces_db())

Modules:
    loop          run_agent and AgentResult -- the agent loop
    config        AgentConfig and the defaults behind it
    tools         tool definitions, catalog lookups, dispatch, submit_answer
    sql           read-only check, query execution, timeouts, result-size limits
    prompts       system prompt and wrap-up instructions
    tracing       traces database schema and writers
    usage         token totals and cost estimates
    log           logger and configure_logging
    serialization DuckDB values to JSON
"""

from .config import DEFAULT_CONFIG, AgentConfig
from .log import configure_logging
from .loop import AgentResult, run_agent
from .tracing import init_traces_db

__all__ = [
    "AgentConfig",
    "AgentResult",
    "DEFAULT_CONFIG",
    "configure_logging",
    "init_traces_db",
    "run_agent",
]
