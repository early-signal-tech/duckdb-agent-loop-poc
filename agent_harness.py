import duckdb
from agent import run_agent, init_traces_db


if __name__ == "__main__":
    # Set up database connection
    conn = duckdb.connect()
    conn.execute("ATTACH 'ducklake:ducklake_prod' as my_ducklake;")
    conn.execute("USE my_ducklake.gold;")

    # Set up traces database
    traces_conn = init_traces_db("agent_traces.duckdb")

    print("Connected to DuckLake production")
    print("Agent traces will be saved to 'agent_traces.duckdb'")

    # Run agent
    run_agent(
        conn=conn,
        table_name="events_rich_ml_gold",
        prompt="Compare the most recent day of data against the prior day and run the most important analysis on it.",
        traces_conn=traces_conn
    )
