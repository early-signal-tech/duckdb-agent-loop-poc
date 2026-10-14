# Duck Harness

A simple SQL-to-LLM agent harness using DuckDB and Anthropic's Claude API.

# Companion Video

[![Companion Video](https://github.com/user-attachments/assets/fffb2f69-a9ff-4c18-abf5-9d5dc004fd20)](https://www.youtube.com/watch?v=kVey_qIXAJk)

## Quick Start

1. **Set up your API key**
   Copy `.env.example` to `.env` and add your Anthropic API key:
   ```bash
   cp .env.example .env
   # Edit .env and add your ANTHROPIC_API_KEY
   ```

2. **Install dependencies**
   ```bash
   uv sync
   ```

3. **Run the demo**
   ```bash
   uv run agent_harness.py
   ```

## Agent Tracing

Every agent run is automatically logged to `agent_traces.duckdb`. Query the traces:

```bash
duckdb agent_traces.duckdb
```

```sql
-- View all agent run summaries
SELECT run_id, timestamp, table_name, prompt, total_iterations FROM agent_runs;

-- View detailed traces for a specific run
SELECT iteration, action, sql_query FROM agent_traces 
WHERE run_id = 'your-run-id'
ORDER BY iteration;
```

This will:
- Fetch the schema of the table specified
- Send the schema to Claude with your prompt
- Let Claude request SQL queries as needed to explore and analyze
- Loop until Claude responds without tool calls (max 5 iterations)

## How It Works

See `AGENT_DESIGN.md` for full architecture and design documentation.

## Using Your Own Data

Edit `agent_harness.py` to:
1. Connect to your DuckDB/DuckLake database
2. Specify a table name and prompt

```python
import duckdb
from agent import run_agent

conn = duckdb.connect("path/to/your.db")
# Or for DuckLake:
# conn.execute("ATTACH 'ducklake:ducklake_prod' AS my_ducklake")

run_agent(
    conn=conn,
    table_name="your_table_name",
    prompt="Your analysis question here"
)
```

## File Structure

```
duck-harness/
├── README.md              (this file)
├── AGENT_DESIGN.md        (full architecture doc)
├── pyproject.toml         (project config & dependencies)
├── uv.lock                (locked dependencies)
├── agent.py               (core agent functions)
└── agent_harness.py       (demo application)
```
