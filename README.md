# Duck Harness

A SQL-to-LLM agent: Claude (Haiku 5.5) answers questions about DuckDB or DuckLake data by writing read-only SQL, reading the results, and iterating until it can answer with evidence. Every step is traced to a DuckDB database.

# Companion Video

[![Companion Video](https://github.com/user-attachments/assets/fffb2f69-a9ff-4c18-abf5-9d5dc004fd20)](https://www.youtube.com/watch?v=kVey_qIXAJk)

## Quick start

1. **API key.** Copy `.env.example` to `.env` and set `ANTHROPIC_API_KEY`.
2. **Install.** `uv sync`
3. **Run.**
   ```bash
   uv run agent_harness.py                      # demo question on DuckLake prod (gold.events_rich_ml_gold)
   uv run agent_harness.py --prompt "Which event types grew most last week?"
   uv run agent_harness.py --db analytics.duckdb --schema main --table orders --table customers
   uv run agent_harness.py --effort high --structured --quiet
   uv run agent_harness.py --help
   ```
   The database is always attached read-only. Without `--table`, the agent finds tables itself. `--quiet` logs progress only and prints the answer; the default shows the full trace (SQL, thinking, results). Exit status is 0 when the run completes, 1 otherwise.

## Using it from Python

```python
import duckdb
from duckdb_agent import AgentConfig, init_traces_db, run_agent

conn = duckdb.connect()
conn.execute("ATTACH 'ducklake:ducklake_prod' AS source (READ_ONLY)")  # always READ_ONLY
conn.execute("USE source.gold")

result = run_agent(
    conn,
    "Compare the most recent day against the prior day.",
    tables="events_rich_ml_gold",          # optional: schemas to preload
    config=AgentConfig(effort="medium"),   # model, effort, limits, features
    traces_conn=init_traces_db("agent_traces.duckdb"),
)
print(result.status, result.estimated_cost_usd)
print(result.text)
```

`result.status` is `completed` on a normal finish; anything else (`max_iterations`, `api_error`, `refusal`, ...) is explained in `AGENT_DESIGN.md`. With `AgentConfig(structured_answer=True)`, `result.answer` holds `summary`, `key_findings`, `caveats`, and `supporting_sql`.

The library logs through the `duckdb_agent` logger and adds no handlers; call `configure_logging(verbose=True)` for console output in scripts.

## Traces

Every run is recorded in `agent_traces.duckdb`:

```sql
-- Recent runs: outcome, cost, and settings
SELECT run_id, started_at, status, total_iterations, round(estimated_cost_usd, 4) AS cost,
       config->>'effort' AS effort, left(prompt, 60) AS prompt
FROM agent_runs ORDER BY started_at DESC LIMIT 10;

-- Per-call tokens, caching, and latency for one run
SELECT iteration, latency_ms, input_tokens, cache_read_input_tokens, output_tokens, request_id
FROM api_calls WHERE run_id = ? ORDER BY id;

-- What Claude ran, and how it went
SELECT iteration, tool_name, success, truncated, duration_ms, error_type, left(input_sql, 80)
FROM tool_calls WHERE run_id = ? ORDER BY id;

-- Claude's reasoning, turn by turn
SELECT iteration, thinking FROM agent_traces WHERE run_id = ? AND action = 'thinking' ORDER BY id;
```

The traces hold SQL, results, and answers, so protect the file like the data itself.

## Tests

```bash
uv run pytest                                    # no API key needed
uv run pytest --cov=duckdb_agent --cov-branch    # with coverage
```

## Docs

- `AGENT_DESIGN.md`: how it works (loop, tools, safety, caching, tracing, logging)
- `UPGRADE_PLAN.md`: the upgrades from the original prototype
- `TIER1_CHANGES.md`: the correctness and safety upgrades in detail
- `DESIGN_NOTES.md`: history of the original prototype

## Layout

```
├── agent_harness.py       command-line entry point
├── duckdb_agent/          the agent package (layout in AGENT_DESIGN.md)
├── tests/                 pytest suite
├── .env.example           environment template
└── pyproject.toml / uv.lock
```
