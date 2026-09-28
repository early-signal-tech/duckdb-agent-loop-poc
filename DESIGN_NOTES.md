# Duck Harness: Design Notes & Decisions

## Timeline: Project Evolution

### **Phase 1: Initial Concept & Documentation**
- Defined goal: simple harness for SQL → LLM → tool calls → loop
- Created `AGENT_DESIGN.md` with architecture, code patterns, dataset recommendations
- Chose Python as implementation language
- Identified Anthropic Claude API as LLM provider

### **Phase 2: Project Structure Setup**
- Created `duck-harness/` folder
- Set up `pyproject.toml` with uv package management
- Created `.env.example` and `.gitignore`
- Removed `requirements.txt` in favor of uv

### **Phase 3: Initial Implementation (Classes)**
- Wrote `agent.py` with `SQLAgent` class
- Implemented `execute_query()`, `run_agent()` methods
- Created `agent_harness.py` demo with sample data
- Hardcoded model to `claude-opus-5`

### **Phase 4: Refactoring to Functions**
- Removed class wrapper for simplicity
- Converted to function-based API: `run_agent()`, `execute_query()`
- Flattened module structure
- Cleaner, more testable code

### **Phase 5: Secrets & Configuration**
- Added `dotenv` dependency
- Updated `agent.py` to load `.env` via `load_dotenv()`
- Switched model to `claude-haiku-4-5-20251001` (cheaper, faster)

### **Phase 6: Tracing Infrastructure**
- Created `agent_traces.duckdb` with two tables: `agent_runs`, `agent_traces`
- Implemented `init_traces_db()`, `log_trace()`, `log_run_summary()`
- Logged: initial query, iterations, tool calls, final response
- Run ID and iteration tracking for debugging

### **Phase 7: Real-World Testing**
- Tested against production DuckLake data (`events_rich_ml_gold` table)
- Demo harness ran successfully with multi-iteration loops
- **Discovered**: Initial SQL approach + large result sets consumed too many tokens

### **Phase 8: Problem: Context Window Blowup**
- User reported: "My query was too big for the context window"
- Root cause: Passing full initial query results as dict string representation
- Example: 100-row result = ~2KB string = thousands of tokens
- Realized: Harness was over-deciding by writing first query

### **Phase 9: Redesign: Schema-First Approach (Planning)**
- Planned architectural shift: `initial_sql` → `table_name`
- Rationale: 
  - Let Claude decide what to query
  - DESCRIBE output = ~300 tokens vs thousands for data
  - Production-safe from the start
- Created detailed plan with read-only guardrails

### **Phase 10: Implementation of Schema-First**
- Rewrote `run_agent(conn, table_name, prompt, ...)` signature
- Changed iteration 0: now runs DESCRIBE instead of user SQL
- Pass schema (column names + types) to Claude in first message
- Renamed iteration-0 trace action: `initial_query` → `schema_lookup`
- Updated README and examples

### **Phase 11: Safety Guardrail**
- Added `is_read_only_sql()` function
- Prefix allowlist: SELECT, WITH, DESCRIBE, SHOW, EXPLAIN, PRAGMA
- Blocks: INSERT, UPDATE, DELETE, DROP, TRUNCATE
- Applied at query-execution time (zero performance cost)
- Returned errors in same format as successful queries

### **Phase 12: Tool Call Tracking**
- User requested: "Track what the agent actually did at each step"
- Created third traces table: `tool_calls`
- Fields: `run_id`, `iteration`, `tool_name`, `input_sql`, `output_rows`, `output_summary`
- Separates "what Claude generated" from "what system did"
- Tested: Verified queries, row counts, and error states logged correctly

### **Phase 13: Documentation & Finalization**
- Updated README with new API examples
- Verified read-only guardrail with test cases
- Updated traces table examples in docs
- Created DESIGN_NOTES.md (this file)

---

## Pain Points Encountered

### 1. **Context Window Blowup**
- **Problem**: Passing raw result rows from large production queries to Claude consumed thousands of tokens
- **Example**: Production DuckLake table queries quickly filled context window before agent could analyze
- **Impact**: Limited harness to small sample queries, defeated the purpose of automated analysis

### 2. **Over-Deciding with Initial SQL**
- **Problem**: Harness was writing the first SQL query, Claude only made follow-up decisions
- **Impact**: Harness had to know query patterns in advance; Claude lost opportunity to decide what data to fetch first
- **Better approach**: Let Claude author all queries from iteration 1

### 3. **Production Safety Without Guardrails**
- **Problem**: When Claude writes queries directly against production DuckLake, accidental writes are possible
- **Example**: Claude could generate INSERT/UPDATE/DELETE/DROP statements
- **Solution**: Read-only filter on all `execute_query()` calls

### 4. **Schema-to-Query Gap**
- **Problem**: Passing only schema column names wasn't enough; Claude needs to understand data shapes
- **Decision**: Pass DESCRIBE output (21 columns × 2 fields) instead of sample rows
- **Benefit**: ~300 tokens vs thousands for sample data

### 5. **Sequence Creation Order**
- **Problem**: Tried to use `DEFAULT nextval('seq_tool_calls')` before sequence was created
- **Solution**: Create sequence BEFORE table definition

### 6. **Tracing Scope Creep**
- **Problem**: Initial traces table was logging everything (queries, results, responses) in one row
- **Decision**: Separate `tool_calls` table for structured tool execution history
- **Benefit**: Easier to query "what did Claude do" vs "what did the system log"

---

## Design Decisions Made

### **API Shape: Table Name Instead of Initial SQL**
**Decision**: Replace `run_agent(conn, initial_sql, prompt)` with `run_agent(conn, table_name, prompt)`

**Why**:
- Claude decides what to query, not the harness
- Schema fits in ~300 tokens instead of thousands
- Production-safe from day one (no sample data exposure)
- Simpler mental model: "analyze this table with this prompt"

**Trade-off**: Loses ability to pass complex pre-filtered data; requires Claude to write exploratory queries


### **Read-Only Guardrail**
**Decision**: Block INSERT/UPDATE/DELETE/DROP/TRUNCATE at query execution time

**Prefix Allowlist**:
```python
READ_ONLY_PREFIXES = ("select", "with", "describe", "show", "explain", "pragma")
```

**Why**:
- Production data is append-only; accidental modifications are catastrophic
- Cheap to enforce (string prefix check before execution)
- Graceful failure (returns error in same format as query success, no exceptions)

**Trade-off**: Restricts Claude's capabilities if future tool types need write access


### **Model Choice: Haiku**
**Decision**: Hardcoded to `claude-haiku-4-5-20251001`

**Why**:
- Fast iteration and testing
- Cheap for high token volumes (exploratory queries return many rows)
- Strong enough for SQL generation and analysis

**Alternative Considered**: Parameterized model selection via environment variables
**Deferred**: Not critical for MVP; can add if needed


### **Language & Structure: Python Functions**
**Decision**: Simple function-based architecture, no classes

**Why**:
- Minimal setup overhead
- Direct integration with DuckDB bindings
- Easy to test individual functions
- Deployment-friendly

**Not Used**: Streamlit UI (planned, then scoped out to focus on core harness)


### **Package Management: UV**
**Decision**: Use `uv` instead of pip/venv

**Why**:
- Faster resolution and installation
- `pyproject.toml` as single source of truth
- Works seamlessly with DuckDB/uv ecosystem

**Setup**: Simple `uv sync` → `uv run agent_harness.py`


### **Tracing Strategy: Multi-Table Design**
**Tables**:
- `agent_runs` — Run summary (table_name, prompt, response, iteration count)
- `agent_traces` — Detailed trace log (every action: schema lookup, tool call, response)
- `tool_calls` — Tool execution details (input query, output rows, success/error)

**Why**:
- `tool_calls` separates "what Claude did" from "what system logged"
- Easier to answer "show me all successful queries" vs scanning full traces
- Extensible for future tool types (not just query_database)

**Trade-off**: Three tables instead of one; requires joins for full story


### **Secrets Management: Environment Variables**
**Decision**: Load `.env` file using `python-dotenv`

**Why**:
- Standard Python practice
- `.env` ignored in git (in `.gitignore`)
- Can easily switch to other secret stores later

**Setup**: Copy `.env.example`, add `ANTHROPIC_API_KEY`


### **Iteration Limits: Fixed 5**
**Decision**: `max_iterations=5` hard default

**Why**:
- Prevents infinite loops if Claude gets stuck
- ~5 iterations covers most analysis: explore → refine → aggregate → analyze → return
- Production costs stay predictable

**Trade-off**: Complex queries might hit limit; user can increase if needed


### **Tool Call Format: Native Anthropic Tool Use**
**Decision**: Use Claude's native `tool_use` response blocks with JSON schema

**Why**:
- No string parsing needed
- Supports multiple concurrent tool calls (iteration 2 shows 3 queries in one iteration)
- Claude understands tool definitions at training time
- Built-in error handling

**Not Used**: Instruction-based "write SQL as a code block" approach


---

## Lessons Learned

### ✅ What Worked Well
1. **Schema-first approach** — Dramatically reduced token usage, kept Claude focused
2. **Read-only guardrail** — Simple, effective, zero false positives
3. **Separate tool_calls table** — Debugging and analysis are much cleaner
4. **Function-based design** — Low overhead, easy to extend

### 🔄 What Could Be Better
1. **Error handling** — Currently returns errors as strings; could use structured error codes
2. **Query validation** — Could add SQL parsing to warn about expensive queries before execution
3. **Streaming** — Large result sets still block; streaming chunks could help
4. **Tool diversity** — Currently only `query_database`; could add viz, export, etc.

### 📊 Performance Observations
- Haiku is fast enough for 5 iterations in ~20 seconds (including DuckLake roundtrips)
- DESCRIBE output is stable; rarely needs schema refresh
- Read-only check adds <1ms overhead per query
- Traces DB grows ~100KB per full run

---

## Future Directions (Out of Scope)

- [ ] Multiple tool types (visualize, export, cache)
- [ ] Parameterized model selection
- [ ] Query cost estimation before execution
- [ ] Streaming large result sets
- [ ] Web UI (Streamlit, FastAPI, etc.)
- [ ] Cross-table joins in initial context
- [ ] Cached schema for frequently-used tables
- [ ] Claude system prompt tuning for domain-specific analysis
