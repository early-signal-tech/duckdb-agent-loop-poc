import anthropic
import duckdb
import httpx2
import pytest

from duckdb_agent.config import API_MAX_RETRIES
from duckdb_agent.loop import run_agent
from duckdb_agent.tracing import init_traces_db
from test_stop_reasons import FakeClient, response, text, tool_use

REQUEST = httpx2.Request("POST", "https://api.anthropic.com/v1/messages")


def status_error(cls, status_code, message="boom"):
    resp = httpx2.Response(status_code, request=REQUEST, headers={"request-id": "req_123"})
    return cls(message, response=resp, body=None)


class RaisingClient(FakeClient):
    """Returns scripted responses, raising any that are exceptions."""

    def create(self, **kwargs):
        item = super().create(**kwargs)
        if isinstance(item, BaseException):
            raise item
        return item


@pytest.fixture
def conn():
    conn = duckdb.connect()
    conn.execute("CREATE TABLE t AS SELECT 1 AS a")
    return conn


@pytest.fixture
def traces_conn(tmp_path):
    return init_traces_db(str(tmp_path / "traces.duckdb"))


def run(conn, client, traces_conn=None):
    return run_agent(conn, "analyze", tables="t", traces_conn=traces_conn, client=client)


@pytest.mark.parametrize("error, retryable", [
    (status_error(anthropic.RateLimitError, 429), True),
    (status_error(anthropic.InternalServerError, 500), True),
    (status_error(anthropic.OverloadedError, 529), True),
    (anthropic.APIConnectionError(request=REQUEST), True),
    (anthropic.APITimeoutError(request=REQUEST), True),
    (status_error(anthropic.BadRequestError, 400), False),
    (status_error(anthropic.AuthenticationError, 401), False),
    (status_error(anthropic.NotFoundError, 404), False),
])
def test_api_errors_end_run_without_raising(conn, error, retryable):
    result = run(conn, RaisingClient(error))
    assert result.status == "api_error"
    assert result.retryable is retryable
    assert type(error).__name__ in result.error


def test_api_error_message_includes_status_and_request_id(conn):
    result = run(conn, RaisingClient(status_error(anthropic.RateLimitError, 429, "slow down")))
    assert result.error == "RateLimitError status=429 request_id=req_123: slow down"


def test_api_error_mid_run_is_traced(conn, traces_conn):
    client = RaisingClient(
        response("tool_use", tool_use("SELECT 1")),
        status_error(anthropic.InternalServerError, 500),
    )
    result = run(conn, client, traces_conn)
    assert result.iterations == 2
    row = traces_conn.execute(
        "SELECT status, error, total_iterations FROM agent_runs WHERE run_id = ?", [result.run_id]
    ).fetchone()
    assert row[0] == "api_error" and "InternalServerError" in row[1] and row[2] == 2
    actions = [a for (a,) in traces_conn.execute(
        "SELECT action FROM agent_traces WHERE run_id = ? ORDER BY id", [result.run_id]
    ).fetchall()]
    assert actions == ["schema_lookup", "tool_call", "api_error"]


def test_unexpected_exception_is_raised_after_writing_summary(conn, traces_conn):
    client = RaisingClient(RuntimeError("bug in our code"))
    with pytest.raises(RuntimeError, match="bug in our code"):
        run(conn, client, traces_conn)
    row = traces_conn.execute("SELECT status, error FROM agent_runs").fetchone()
    assert row == ("error", "RuntimeError: bug in our code")


def test_keyboard_interrupt_is_recorded_as_interrupted(conn, traces_conn):
    client = RaisingClient(KeyboardInterrupt())
    with pytest.raises(KeyboardInterrupt):
        run(conn, client, traces_conn)
    assert traces_conn.execute("SELECT status FROM agent_runs").fetchone() == ("interrupted",)


def test_tracing_failure_does_not_hide_original_error(conn, traces_conn):
    client = RaisingClient(RuntimeError("original"))
    traces_conn.execute("DROP TABLE agent_runs")  # make the summary write fail
    with pytest.raises(RuntimeError, match="original"):
        run(conn, client, traces_conn)


def test_successful_run_has_no_error(conn):
    result = run(conn, RaisingClient(response("end_turn", text("ok"))))
    assert result.status == "completed"
    assert result.error is None and result.retryable is False


def test_default_client_uses_higher_retry_count(conn, monkeypatch):
    created = {}

    class SpyAnthropic:
        def __init__(self, **kwargs):
            created.update(kwargs)
            self.beta = FakeClient(response("end_turn", text("ok"))).beta

    monkeypatch.setattr(anthropic, "Anthropic", SpyAnthropic)
    run_agent(conn, "analyze", tables="t")
    assert created["max_retries"] == API_MAX_RETRIES
