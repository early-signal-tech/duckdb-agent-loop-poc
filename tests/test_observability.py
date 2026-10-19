import anthropic
import duckdb
import pytest

from duckdb_agent.config import AgentConfig
from duckdb_agent.loop import run_agent
from duckdb_agent.tracing import init_traces_db
from duckdb_agent.usage import estimate_cost
from test_api_errors import RaisingClient, status_error
from test_prompt_caching import usage
from test_stop_reasons import FakeClient, response, text, tool_use


@pytest.fixture
def conn():
    conn = duckdb.connect()
    conn.execute("CREATE TABLE t AS SELECT 1 AS a")
    return conn


@pytest.fixture
def traces_conn(tmp_path):
    return init_traces_db(str(tmp_path / "traces.duckdb"))


def call(resp, request_id, **token_counts):
    """A scripted response with usage and a request ID, as the SDK would return."""
    resp.usage = usage(**token_counts)
    resp._request_id = request_id
    return resp


def run(conn, client, traces_conn, **settings):
    return run_agent(conn, "analyze", tables="t", client=client, traces_conn=traces_conn, config=AgentConfig(**settings))


# --- cost estimates ---

def test_cost_standard_rates():
    # $0.10 / $0.50 per million input / output tokens for prompts up to 100K tokens.
    cost = estimate_cost("claude-haiku-5-5", usage(input=50_000, output=50_000))
    assert cost == pytest.approx(50_000 * 0.10 / 1e6 + 50_000 * 0.50 / 1e6)


def test_cost_cache_multipliers():
    # Cache writes cost 1.25x input; cache reads 0.1x.
    write = estimate_cost("claude-haiku-5-5", usage(cache_write=50_000))
    read = estimate_cost("claude-haiku-5-5", usage(cache_read=50_000))
    assert write == pytest.approx(50_000 * 0.10 * 1.25 / 1e6)
    assert read == pytest.approx(50_000 * 0.10 * 0.1 / 1e6)


def test_cost_long_prompt_rate_card_counts_cached_tokens():
    # The rate card is chosen by total prompt size, cached tokens included.
    short = estimate_cost("claude-haiku-5-5", usage(input=1_000, cache_read=99_000, output=0))
    long = estimate_cost("claude-haiku-5-5", usage(input=1_000, cache_read=100_000, output=0))
    assert short == pytest.approx((1_000 * 0.10 + 99_000 * 0.01) / 1_000_000)
    assert long == pytest.approx((1_000 * 0.50 + 100_000 * 0.05) / 1_000_000)


def test_cost_unknown_model_is_none():
    assert estimate_cost("claude-opus-5-5", usage(input=100)) is None


# --- per-call rows ---

def test_each_api_call_is_logged(conn, traces_conn):
    client = FakeClient(
        call(response("tool_use", tool_use("SELECT 1")), "req_1", input=4_000, cache_write=3_000, output=200),
        call(response("end_turn", text("ok")), "req_2", input=300, cache_read=7_000, output=400),
    )
    result = run(conn, client, traces_conn)
    rows = traces_conn.execute("""
        SELECT iteration, model, request_id, stop_reason, input_tokens, cache_creation_input_tokens,
               cache_read_input_tokens, output_tokens, latency_ms >= 0, estimated_cost_usd > 0, error
        FROM api_calls WHERE run_id = ? ORDER BY id
    """, [result.run_id]).fetchall()
    assert rows == [
        (1, "claude-haiku-5-5", "req_1", "tool_use", 4_000, 3_000, 0, 200, True, True, None),
        (2, "claude-haiku-5-5", "req_2", "end_turn", 300, 0, 7_000, 400, True, True, None),
    ]


def test_failed_call_is_logged_with_request_id(conn, traces_conn):
    client = RaisingClient(status_error(anthropic.InternalServerError, 500))
    result = run(conn, client, traces_conn)
    assert result.status == "api_error"
    request_id, stop_reason, tokens, error = traces_conn.execute(
        "SELECT request_id, stop_reason, input_tokens, error FROM api_calls WHERE run_id = ?", [result.run_id]
    ).fetchone()
    assert request_id == "req_123"
    assert stop_reason is None and tokens is None
    assert "InternalServerError" in error


def test_logged_model_is_the_one_the_api_reports(conn, traces_conn):
    resp = call(response("end_turn", text("ok")), "req_1", input=10, output=5)
    resp.model = "claude-haiku-5-5-20261001"
    result = run(conn, FakeClient(resp), traces_conn)
    model = traces_conn.execute("SELECT model FROM api_calls WHERE run_id = ?", [result.run_id]).fetchone()[0]
    assert model == "claude-haiku-5-5-20261001"


# --- per-run totals ---

def test_run_totals_match_the_sum_of_calls(conn, traces_conn):
    client = FakeClient(
        call(response("tool_use", tool_use("SELECT 1")), "req_1", input=4_000, cache_write=3_000, output=200),
        call(response("end_turn", text("ok")), "req_2", input=300, cache_read=7_000, output=400),
    )
    result = run(conn, client, traces_conn)
    run_row = traces_conn.execute("""
        SELECT api_calls, input_tokens, cache_creation_input_tokens, cache_read_input_tokens,
               output_tokens, estimated_cost_usd
        FROM agent_runs WHERE run_id = ?
    """, [result.run_id]).fetchone()
    summed = traces_conn.execute("""
        SELECT count(*), sum(input_tokens), sum(cache_creation_input_tokens), sum(cache_read_input_tokens),
               sum(output_tokens), sum(estimated_cost_usd)
        FROM api_calls WHERE run_id = ?
    """, [result.run_id]).fetchone()
    assert run_row[:5] == summed[:5] == (2, 4_300, 3_000, 7_000, 600)
    assert run_row[5] == pytest.approx(summed[5])
    assert result.estimated_cost_usd == pytest.approx(summed[5])
    assert result.api_calls == 2


def test_run_timing_is_recorded(conn, traces_conn):
    result = run(conn, FakeClient(response("end_turn", text("ok"))), traces_conn)
    started_at, ended_at, duration_ms, timestamp = traces_conn.execute(
        "SELECT started_at, ended_at, duration_ms, timestamp FROM agent_runs WHERE run_id = ?", [result.run_id]
    ).fetchone()
    assert started_at <= ended_at
    assert timestamp == ended_at
    assert duration_ms == result.duration_ms >= 0


def test_wrap_up_call_is_counted(conn, traces_conn):
    client = FakeClient(response("tool_use", tool_use("SELECT 1")), response("end_turn", text("summary")))
    result = run(conn, client, traces_conn, max_iterations=1)
    assert result.api_calls == 2
    assert traces_conn.execute("SELECT count(*) FROM api_calls WHERE run_id = ?", [result.run_id]).fetchone()[0] == 2


def test_unpriced_model_has_no_cost(conn, traces_conn):
    client = FakeClient(call(response("end_turn", text("ok")), "req_1", input=10, output=5))
    result = run(conn, client, traces_conn, model="claude-opus-5-5")
    assert result.estimated_cost_usd is None
    cost = traces_conn.execute("SELECT estimated_cost_usd FROM agent_runs WHERE run_id = ?", [result.run_id]).fetchone()[0]
    assert cost is None


def test_existing_traces_db_gains_new_columns(tmp_path):
    path = str(tmp_path / "old.duckdb")
    old = duckdb.connect(path)
    old.execute("CREATE TABLE agent_runs (run_id VARCHAR PRIMARY KEY, timestamp TIMESTAMP, initial_sql VARCHAR, "
                "prompt VARCHAR, final_response VARCHAR, total_iterations INTEGER)")
    old.close()
    traces_conn = init_traces_db(path)
    columns = {row[0] for row in traces_conn.execute("DESCRIBE agent_runs").fetchall()}
    assert {"started_at", "ended_at", "duration_ms", "api_calls", "estimated_cost_usd", "output_tokens"} <= columns
