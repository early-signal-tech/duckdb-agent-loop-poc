# SQL-to-LLM Agent Harness

A simple agent loop that queries data via DuckDB, sends it to Claude with a prompt, and handles tool calls for follow-up queries.

## Architecture Overview

```
┌─────────────────────────────────────────────────────────────┐
│                                                             │
│  1. Execute SQL Query                                      │
│     (DuckDB) → Get Results                                 │
│           ↓                                                │
│  2. Build Message                                          │
│     Results + Prompt → Claude API                          │
│           ↓                                                │
│  3. Check Response                                         │
│     ├─ Tool Call? → Execute Tool (DuckDB query)           │
│     │               Loop back to step 2                    │
│     └─ Text Response? → End (eval complete)               │
│           ↓                                                │
│  Repeat 2-3 for ~3-5 cycles max                           │
│                                                             │
└─────────────────────────────────────────────────────────────┘
```

## Programming Language: Python

**Why Python?**
- **Anthropic SDK**: Clean, well-documented Python client for Claude API
- **DuckDB Integration**: Excellent Python bindings with `duckdb` package
- **Simplicity**: Agent loops are straightforward to write and debug
- **Ecosystem**: Rich libraries for data handling, HTTP, JSON parsing
- **Speed to iterate**: Perfect for prototyping agent behavior

## How It Works (Step-by-Step)

### Phase 1: Initial Query
1. Connect to DuckDB (local or DuckLake)
2. Execute initial SQL query
3. Fetch results as structured data (dict/JSON)

### Phase 2: LLM Loop (Max 3-5 cycles)
1. Format query results + user prompt into a message
2. Call Claude API with tool definitions (via `tools` parameter)
3. Receive response with possible `tool_use` block

### Phase 3: Tool Handling
**If response contains tool_use:**
- Extract tool name and input parameters
- Execute the requested SQL query against DuckDB
- Add tool result back to message history
- Loop back to Phase 2

**If response is text-only:**
- Evaluation complete
- Return final response to user

### Phase 4: Guardrails
- Max iterations: 5 (prevents infinite loops)
- Tool allowlist: Only allow `query_database` tool
- Input validation: Sanitize SQL to prevent injection

---

## Code Structure

```
agent_harness.py
├── Config
│   ├── DuckDB connection params
│   ├── API key
│   └── Max iterations
│
├── SQLExecutor
│   └── execute_query(sql) → dict
│
├── ToolHandler
│   ├── tool definitions (JSON schema)
│   └── process_tool_call(name, params) → result
│
├── Agent
│   ├── run_loop(initial_sql, user_prompt) → final_response
│   ├── call_claude(messages, tools) → response
│   └── handle_tool_use(response) → tool_result
│
└── main()
    └── Demo with sample data
```

## Minimal Implementation Pattern

```python
import anthropic
import duckdb
from typing import Any

# 1. Set up clients
client = anthropic.Anthropic()
conn = duckdb.connect(":memory:")

# 2. Define tools
tools = [
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

# 3. Execute queries
def execute_query(sql: str) -> dict:
    result = conn.execute(sql).fetchall()
    columns = [desc[0] for desc in conn.description]
    return {"columns": columns, "rows": result}

# 4. Main loop
def agent_loop(initial_sql: str, prompt: str, max_iterations: int = 5):
    # Get initial data
    initial_data = execute_query(initial_sql)
    
    messages = [
        {
            "role": "user",
            "content": f"Data:\n{initial_data}\n\nPrompt: {prompt}"
        }
    ]
    
    for iteration in range(max_iterations):
        # Call Claude
        response = client.messages.create(
            model="claude-opus-5",
            max_tokens=1024,
            tools=tools,
            messages=messages
        )
        
        # Check for tool use
        has_tool_use = any(block.type == "tool_use" for block in response.content)
        
        if not has_tool_use:
            # Extract and return text response
            return next(
                (block.text for block in response.content if hasattr(block, "text")),
                "No response"
            )
        
        # Handle tool calls
        tool_results = []
        for block in response.content:
            if block.type == "tool_use":
                result = execute_query(block.input["sql"])
                tool_results.append({
                    "type": "tool_result",
                    "tool_use_id": block.id,
                    "content": str(result)
                })
        
        # Add assistant response and tool results to messages
        messages.append({"role": "assistant", "content": response.content})
        messages.append({"role": "user", "content": tool_results})
    
    return "Max iterations reached"
```

---

## Dataset Recommendation

### Option A: Production Data (Recommended)
Use your actual DuckLake production data:
```sql
ATTACH 'ducklake:ducklake_prod' AS my_ducklake;
USE my_ducklake;

-- Query one of your real tables
SELECT * FROM your_main_table LIMIT 100;
```

**Pros:**
- Realistic data patterns
- Tests actual query patterns agent might encounter
- Validates against real schema complexity

### Option B: Sample Dataset (Quick Start)
If you want a pre-built public dataset, consider:

```sql
-- NYC Taxi Data (realistic, complex)
CREATE TABLE taxi_data AS
SELECT * FROM read_parquet('s3://nyc-tlc/trip data/yellow_tripdata_2023-01.parquet');

-- OR: Create synthetic customer/sales data
CREATE TABLE customers AS
SELECT 
    generate_subscripts(range(1, 1000), 1) as customer_id,
    'Customer_' || customer_id as name,
    random() * 100000 as lifetime_value
FROM range(1) t(x);
```

**Good for:**
- Prototyping behavior
- Testing agent robustness
- Clean, predictable schema

### Example Agent Tasks

```
Task 1: "What are the top 5 customers by lifetime value?"
Task 2: "Find all transactions over $500 from 2024 and group by region"
Task 3: "Identify customers with declining activity in the last quarter"
```

---

## Key Design Decisions

| Decision | Choice | Rationale |
|----------|--------|-----------|
| Language | Python | Best SDK support + DuckDB integration |
| LLM | Claude Opus 5 | Good balance of reasoning + cost for agent tasks |
| Max Iterations | 5 | Prevents runaway loops, enough for most workflows |
| Tool Validation | Whitelist SQL patterns | Prevents injection, ensures safety |
| State Tracking | Message array | Native to Claude API, easy to debug |

---

## Next Steps

1. **Setup**: `pip install anthropic duckdb`
2. **Minimal test**: Copy the code pattern above, run with NYC taxi data
3. **Add safety**: SQL validation, query timeouts, output size limits
4. **Evaluate**: Compare tool-call cycles, response quality, latency
5. **Extend**: Add more tools (visualizations, aggregations, etc.)

---

## Testing the Loop

```python
if __name__ == "__main__":
    result = agent_loop(
        initial_sql="SELECT * FROM customers LIMIT 50",
        prompt="Who are my top 10 customers by spending?"
    )
    print(result)
```

This should:
1. Execute the initial query → 50 customer records
2. Send to Claude with the prompt
3. Claude might call the tool for aggregate query
4. Execute that query, return results
5. Claude provides final analysis

**Expected cycles**: 1-3 tool calls before text response
