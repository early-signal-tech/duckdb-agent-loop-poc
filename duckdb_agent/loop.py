"""The agent loop: run_agent and the result it returns."""

import anthropic
import duckdb
import logging
import time
import uuid
from dataclasses import dataclass, field
from datetime import datetime

from .config import API_MAX_RETRIES, AgentConfig, CACHE_CONTROL, CONTEXT_MANAGEMENT, CONTEXT_MANAGEMENT_BETA, DEFAULT_CONFIG, THINKING
from .log import logger
from .prompts import MAX_ITERATIONS_WRAP_UP_PROMPT, build_system_prompt
from .serialization import json_dumps
from .sql import tool_error
from .tools import TOOL_LABELS, build_tools, describe_table, format_answer, run_tool, tool_result_block, validate_answer
from .tracing import log_api_call, log_run_summary, log_tool_call, log_trace
from .usage import MODEL_PRICING, add_usage, estimate_cost


@dataclass
class AgentResult:
    """Outcome of an agent run. `status` says how the run ended, not just what was said."""
    run_id: str
    # completed | max_tokens | refusal | context_window_exceeded | unexpected_stop
    # | max_iterations | api_error | error | interrupted
    # max_iterations: `text` is Claude's best answer from partial work, made with tools disabled.
    status: str
    text: str
    iterations: int
    stop_reason: str | None = None
    error: str | None = None
    retryable: bool = False  # api_error only: worth retrying the whole run later
    # structured_answer runs only: the submit_answer input (summary, key_findings,
    # caveats, supporting_sql). None if the run ended any other way.
    answer: dict | None = None
    # Token totals across every API call in the run (see USAGE_FIELDS). input_tokens
    # is only the uncached part; total prompt tokens = input + cache_creation + cache_read.
    usage: dict = field(default_factory=dict)
    # Sum of per-call estimates (see MODEL_PRICING); None if the model isn't priced.
    estimated_cost_usd: float | None = None
    api_calls: int = 0  # API calls made, including failed ones and the wrap-up call
    duration_ms: int = 0  # wall-clock time for the whole run


# Terminal stop reasons other than a normal finish, mapped to run status.
ABNORMAL_STOP_STATUS = {
    "max_tokens": "max_tokens",
    "refusal": "refusal",
    "model_context_window_exceeded": "context_window_exceeded",
}


def response_text(response) -> str:
    """Join every text block in a response (thinking and tool_use blocks are skipped)."""
    return "\n".join(block.text for block in response.content if block.type == "text")


def response_thinking(response) -> str:
    """Join the thinking summaries in a response. Redacted thinking has no text and is skipped."""
    return "\n\n".join(
        block.thinking for block in response.content if block.type == "thinking" and block.thinking
    )


def describe_api_error(error: anthropic.APIError) -> str:
    """One-line description of an API error, with status and request ID when available."""
    parts = [type(error).__name__]
    if isinstance(error, anthropic.APIStatusError):
        parts.append(f"status={error.status_code}")
        if error.request_id:
            parts.append(f"request_id={error.request_id}")
    return f"{' '.join(parts)}: {error.message}"


def is_retryable_api_error(error: anthropic.APIError) -> bool:
    """True for failures worth retrying the whole run later (rate limits, overload, outages, network)."""
    if isinstance(error, anthropic.APIConnectionError):  # includes APITimeoutError
        return True
    if isinstance(error, anthropic.APIStatusError):
        return error.status_code in (408, 409, 429) or error.status_code >= 500
    return False


def run_agent(
    conn: duckdb.DuckDBPyConnection,
    prompt: str,
    tables: str | list[str] | None = None,
    config: AgentConfig = DEFAULT_CONFIG,
    traces_conn: duckdb.DuckDBPyConnection = None,
    client: anthropic.Anthropic = None
) -> AgentResult:
    """Run the SQL-to-LLM agent loop.

    The arguments are what a run is about (the data, the question, where to log);
    `config` is how it runs (model, effort, limits, features -- see AgentConfig).

    `tables` names the tables to focus on; their schemas are put in the first message
    to save a round trip. Claude can still find and use any table in the current
    database. With no tables, it starts from list_tables.

    API failures end the run with status "api_error" instead of raising. Any other
    exception (a bug, Ctrl-C) is re-raised, but the run summary is written first.
    """
    # The SDK retries rate limits, overload, 5xx, and network errors with backoff.
    client = client or anthropic.Anthropic(max_retries=API_MAX_RETRIES)
    run_id = str(uuid.uuid4())
    summary_written = False
    tables = [tables] if isinstance(tables, str) else list(tables or [])
    table_name = ", ".join(tables) or None  # for the agent_runs trace
    started_at, start_clock = datetime.now(), time.monotonic()
    # Shared with the loop. cost stays None for models without pricing.
    progress = {
        "iteration": 0,
        "usage": {},
        "api_calls": 0,
        "cost": 0.0 if config.model in MODEL_PRICING else None,
    }

    def finish(
        status: str,
        text: str,
        iterations: int,
        stop_reason: str = None,
        error: str = None,
        retryable: bool = False,
        answer: dict = None
    ) -> AgentResult:
        nonlocal summary_written
        duration_ms = round((time.monotonic() - start_clock) * 1000)
        cost = progress["cost"]
        # Anything but a normal finish is a WARNING, so it stands out in service logs.
        level = logging.INFO if status == "completed" else logging.WARNING
        logger.log(level, "\n[Final] status=%s stop_reason=%s iterations=%s run_id=%s",
                   status, stop_reason, iterations, run_id)
        if error:
            logger.warning("  Error: %s", error)
        if progress["usage"]:
            logger.info("  Tokens: %s", " ".join(f"{name}={count:,}" for name, count in progress["usage"].items()))
        cost_text = f"${cost:.4f}" if cost is not None else "n/a"
        logger.info("  API calls: %s  Duration: %.1fs  Est. cost: %s",
                    progress["api_calls"], duration_ms / 1000, cost_text)
        logger.debug("%s\nResponse:\n%s\n%s", "=" * 60, text, "=" * 60)
        if traces_conn:
            log_run_summary(
                traces_conn,
                run_id,
                table_name,
                prompt,
                text,
                iterations,
                status=status,
                stop_reason=stop_reason,
                error=error,
                model=config.model,
                config=config.to_json(),
                started_at=started_at,
                duration_ms=duration_ms,
                api_calls=progress["api_calls"],
                usage=progress["usage"],
                estimated_cost_usd=cost
            )
        summary_written = True
        return AgentResult(
            run_id, status, text, iterations, stop_reason, error, retryable, answer,
            dict(progress["usage"]), cost, progress["api_calls"], duration_ms
        )

    try:
        return _run_agent_loop(conn, tables, prompt, config, traces_conn, client, run_id, finish, progress)
    except BaseException as e:
        if not summary_written:
            status = "interrupted" if isinstance(e, KeyboardInterrupt) else "error"
            try:
                finish(status, "", progress["iteration"], error=f"{type(e).__name__}: {e}")
            except Exception:
                # Never let a tracing failure hide the original exception.
                logger.exception("Failed to write run summary for run %s", run_id)
        raise


def _run_agent_loop(
    conn: duckdb.DuckDBPyConnection,
    tables: list[str],
    prompt: str,
    config: AgentConfig,
    traces_conn: duckdb.DuckDBPyConnection,
    client: anthropic.Anthropic,
    run_id: str,
    finish,
    progress: dict
) -> AgentResult:
    logger.info("\n%s\nStarting agent loop (Run ID: %s, model: %s, effort: %s)\n%s",
                "=" * 60, run_id, config.model, config.effort, "=" * 60)

    tools = build_tools(config)
    system_prompt = build_system_prompt(config)
    if config.prompt_caching:
        system = [{"type": "text", "text": system_prompt, "cache_control": CACHE_CONTROL}]
        cache_params = {"cache_control": CACHE_CONTROL}
    else:
        system, cache_params = system_prompt, {}
    # Context editing is a beta, so every call goes through the beta endpoint; with it
    # off, no beta is requested and the endpoint behaves like the standard one.
    context_params = {
        "betas": [CONTEXT_MANAGEMENT_BETA],
        "context_management": CONTEXT_MANAGEMENT,
    } if config.context_editing else {}

    # Preload the schema of each focus table, saving Claude a describe_table call.
    table_blocks = []
    for table in tables:
        # The schema must arrive whole: wide tables and long STRUCT types are not clipped.
        schema_result = describe_table(conn, table, max_rows=10_000, max_chars=1_000_000, max_cell_chars=100_000)
        logger.debug("\n[Iteration 0] Schema lookup: %s", table)
        logger.debug("  Columns: %s", schema_result.get("row_count", "error"))

        if traces_conn:
            log_trace(
                traces_conn,
                run_id,
                0,
                "schema_lookup",
                sql_query=f"describe_table {table}",
                query_result=json_dumps(schema_result)
            )

        if schema_result["success"]:
            schema_str = "\n".join(
                f"  - {column} ({data_type})" + (f": {comment}" if comment else "")
                for column, data_type, _, comment in schema_result["rows"]
            )
            table_blocks.append(f"<table name=\"{schema_result['table']}\">\n{schema_str}\n</table>")
        else:
            table_blocks.append(f"<table name=\"{table}\">\nError: {schema_result['error']}\n</table>")

    messages = [
        {
            "role": "user",
            "content": "".join(f"{block}\n\n" for block in table_blocks)
                      + f"<question>\n{prompt}\n</question>"
        }
    ]

    def call_claude(iteration: int, **extra):
        """Call the API. Returns (response, None), or (None, AgentResult) if the call failed."""
        progress["iteration"] = iteration
        progress["api_calls"] += 1
        logger.info("\n[Iteration %s] Calling Claude...", iteration)
        # Latency covers the SDK's own retries and backoff, i.e. what the run waited.
        call_started_at, call_clock = datetime.now(), time.monotonic()
        try:
            response = client.beta.messages.create(
                model=config.model,
                max_tokens=config.max_tokens,
                thinking=THINKING,
                output_config={"effort": config.effort},
                system=system,
                tools=tools,
                messages=messages,
                **cache_params,
                **context_params,
                **extra
            )
        except anthropic.APIError as e:
            # Reached only after the SDK's own retries are exhausted (or for
            # non-retryable errors like 400/401/403/404).
            latency_ms = round((time.monotonic() - call_clock) * 1000)
            error = describe_api_error(e)
            if traces_conn:
                log_trace(traces_conn, run_id, iteration, "api_error", claude_response=error)
                log_api_call(
                    traces_conn, run_id, iteration, call_started_at, latency_ms, config.model,
                    request_id=getattr(e, "request_id", None), error=error
                )
            return None, finish("api_error", "", iteration, error=error, retryable=is_retryable_api_error(e))
        latency_ms = round((time.monotonic() - call_clock) * 1000)
        # The SDK sets _request_id from the response's request-id header.
        request_id = getattr(response, "_request_id", None)
        usage = getattr(response, "usage", None)
        cost = estimate_cost(config.model, usage) if usage is not None else None
        if cost is not None and progress["cost"] is not None:
            progress["cost"] += cost
        logger.info("  stop_reason: %s  latency: %.1fs  request_id: %s",
                    response.stop_reason, latency_ms / 1000, request_id)
        if usage is not None:
            add_usage(progress["usage"], usage)
            logger.info(
                "  tokens: input=%s cache_read=%s cache_write=%s output=%s%s",
                f"{usage.input_tokens or 0:,}",
                f"{usage.cache_read_input_tokens or 0:,}",
                f"{usage.cache_creation_input_tokens or 0:,}",
                f"{usage.output_tokens or 0:,}",
                f"  est. cost: ${cost:.5f}" if cost is not None else ""
            )
        if traces_conn:
            log_api_call(
                traces_conn, run_id, iteration, call_started_at, latency_ms,
                # The model that actually served the call, as reported by the API.
                getattr(response, "model", None) or config.model,
                request_id=request_id,
                stop_reason=response.stop_reason,
                usage=usage,
                estimated_cost_usd=cost
            )
        context_management = getattr(response, "context_management", None)
        for edit in getattr(context_management, "applied_edits", None) or []:
            details = edit.model_dump() if hasattr(edit, "model_dump") else vars(edit)
            logger.info("  context edit: %s", details)
            if traces_conn:
                log_trace(traces_conn, run_id, iteration, "context_edit", claude_response=json_dumps(details))
        thinking = response_thinking(response)
        if thinking:
            first_line = thinking.strip().splitlines()[0]
            logger.debug("  thinking: %s%s", first_line[:100], "..." if len(thinking) > len(first_line[:100]) else "")
            if traces_conn:
                log_trace(traces_conn, run_id, iteration, "thinking", thinking=thinking)
        return response, None

    def finish_terminal_turn(response, iteration: int, finished_status: str) -> AgentResult:
        """End the run on a turn with no tools to run.

        `finished_status` applies when Claude finished normally; other stop reasons
        map to their own status. Tool calls in a max_tokens/refusal turn may be cut
        off mid-input, so they are never executed.
        """
        stop_reason = response.stop_reason
        text_response = response_text(response)
        if stop_reason in ("end_turn", "stop_sequence"):
            status = finished_status
        else:
            status = ABNORMAL_STOP_STATUS.get(stop_reason, "unexpected_stop")
        if stop_reason == "refusal":
            details = response.stop_details
            category = details.category if details else None
            text_response = text_response or f"Claude declined this request (category: {category})."

        if traces_conn:
            log_trace(
                traces_conn,
                run_id,
                iteration,
                "claude_response" if status == "completed" else status,
                claude_response=text_response
            )

        return finish(status, text_response, iteration, stop_reason)

    for iteration in range(1, config.max_iterations + 1):
        response, failed = call_claude(iteration)
        if failed:
            return failed

        if response.stop_reason == "pause_turn":
            # Server paused its own sampling loop; resend the turn as-is to resume.
            # No extra user message -- the API resumes from the trailing assistant content.
            messages.append({"role": "assistant", "content": response.content})
            continue

        if response.stop_reason != "tool_use":
            return finish_terminal_turn(response, iteration, "completed")

        # Handle tool calls
        logger.debug("  Claude requested tool use")
        tool_uses = [block for block in response.content if block.type == "tool_use"]

        # A lone submit_answer with valid input ends the run. Anything else falls
        # through and gets an error result, so Claude can fix it and resubmit.
        if config.structured_answer and len(tool_uses) == 1 and tool_uses[0].name == "submit_answer":
            answer = tool_uses[0].input
            if validate_answer(answer) is None:
                text_response = format_answer(answer)
                logger.info("  Tool: %s (%s)", TOOL_LABELS["submit_answer"], tool_uses[0].id)
                if traces_conn:
                    log_trace(
                        traces_conn,
                        run_id,
                        iteration,
                        "submit_answer",
                        claude_response=json_dumps(answer),
                        tool_calls=tool_uses[0].id,
                        tool_name=TOOL_LABELS["submit_answer"]
                    )
                return finish("completed", text_response, iteration, response.stop_reason, answer=answer)

        tool_results = []
        for block in tool_uses:
            tool_name = TOOL_LABELS.get(block.name, block.name)
            # query_database logs its SQL; the other tools log their JSON input.
            sql = block.input.get("sql") if block.name == "query_database" else json_dumps(block.input)
            logger.info("  Tool: %s (%s)", tool_name, block.id)
            logger.debug("  Input: %s...", (sql or "")[:80])
            tool_started_at, tool_clock = datetime.now(), time.monotonic()
            if config.structured_answer and block.name == "submit_answer":
                if len(tool_uses) == 1:
                    result = tool_error(validate_answer(block.input), "invalid_answer")
                else:
                    result = tool_error(
                        "submit_answer must be called on its own, after you have seen the "
                        "results of the queries your answer relies on. Your other tool calls "
                        "in this turn were run; review their results, then submit again.",
                        "submit_not_alone"
                    )
            else:
                result = run_tool(conn, block.name, block.input, config)
            duration_ms = round((time.monotonic() - tool_clock) * 1000)
            result_block = tool_result_block(block.id, result)
            result_chars = len(result_block["content"])
            row_count = result.get('row_count', 0)
            outcome = (
                f"{row_count} rows{' (truncated)' if result.get('truncated') else ''}"
                if result["success"] else f"Error [{result.get('error_type')}]"
            )
            logger.info("  Result: %s  (%.2fs, %s chars)", outcome, duration_ms / 1000, f"{result_chars:,}")
            if not result["success"]:
                # DuckDB messages can quote data values, so the text is DEBUG-only.
                logger.debug("  Error message: %s", result["error"])
            tool_results.append(result_block)

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
                    output_summary="Success" if result.get("success") else f"Error: {result.get('error')}",
                    started_at=tool_started_at,
                    duration_ms=duration_ms,
                    result=result,
                    result_chars=result_chars
                )

        # Add to message history
        messages.append({"role": "assistant", "content": response.content})
        messages.append({"role": "user", "content": tool_results})

    # Out of iterations. Instead of discarding the work, make one last call with
    # tools disabled and ask Claude to answer from the results it already has.
    logger.warning("\n[Final] Max iterations (%s) reached; asking for a final answer", config.max_iterations)
    # Sent as a mid-conversation system message: the operator channel for harness
    # instructions, which can't be confused with (or spoofed by) data in tool results.
    # It goes after the history, so the cached prefix is untouched. The API requires it
    # to follow a user turn (here, the last tool results) or an assistant turn that
    # ended in server-tool use (the only way a pause_turn can happen).
    messages.append({"role": "system", "content": MAX_ITERATIONS_WRAP_UP_PROMPT})

    wrap_up_iteration = config.max_iterations + 1
    # Changing tool_choice keeps the tools + system cache but invalidates the
    # conversation cache, so this one call re-writes the history. It happens once per run.
    response, failed = call_claude(wrap_up_iteration, tool_choice={"type": "none"})
    if failed:
        return failed
    return finish_terminal_turn(response, wrap_up_iteration, "max_iterations")
