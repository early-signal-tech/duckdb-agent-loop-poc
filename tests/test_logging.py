import io
import logging
from types import SimpleNamespace

import anthropic
import duckdb
import pytest

from duckdb_agent.config import AgentConfig
from duckdb_agent.log import configure_logging, logger, LOGGER_NAME
from duckdb_agent.loop import run_agent
from duckdb_agent.tracing import init_traces_db
from test_api_errors import RaisingClient, status_error
from test_stop_reasons import FakeClient, response, text, tool_use

SECRET_SQL = "SELECT 'secret_value' AS v"
THINKING_TEXT = "Reasoning about revenue_by_region"
ANSWER_TEXT = "Revenue rose 12% in EMEA."


@pytest.fixture(autouse=True)
def restore_logger():
    """configure_logging changes global logger state; put it back after each test."""
    handlers, level, propagate = list(logger.handlers), logger.level, logger.propagate
    yield
    logger.handlers[:] = handlers
    logger.setLevel(level)
    logger.propagate = propagate


@pytest.fixture
def conn():
    conn = duckdb.connect()
    conn.execute("CREATE TABLE t AS SELECT 1 AS a")
    return conn


def thinking(summary):
    return SimpleNamespace(type="thinking", thinking=summary, signature="sig")


def typical_run(conn, **kwargs):
    client = FakeClient(
        response("tool_use", thinking(THINKING_TEXT), tool_use(SECRET_SQL)),
        response("end_turn", text(ANSWER_TEXT)),
    )
    return run_agent(conn, "analyze", tables="t", client=client, **kwargs)


def logged_text(caplog, level):
    return "\n".join(r.getMessage() for r in caplog.records if r.levelno >= level)


def test_nothing_is_printed(conn, capsys):
    typical_run(conn)
    captured = capsys.readouterr()
    assert captured.out == "" and captured.err == ""


def test_info_has_progress_but_no_content(conn, caplog):
    caplog.set_level(logging.INFO, logger=LOGGER_NAME)
    typical_run(conn)
    info = logged_text(caplog, logging.INFO)
    assert "Starting agent loop" in info
    assert "[Iteration 1] Calling Claude" in info
    assert "Tool: Query Database" in info
    assert "Result: 1 rows" in info
    assert "status=completed" in info
    # Query data, SQL, thinking, and the answer stay out of INFO.
    for content in ("secret_value", THINKING_TEXT, ANSWER_TEXT):
        assert content not in info


def test_debug_adds_the_content(conn, caplog):
    caplog.set_level(logging.DEBUG, logger=LOGGER_NAME)
    typical_run(conn)
    debug = logged_text(caplog, logging.DEBUG)
    for content in ("secret_value", THINKING_TEXT, ANSWER_TEXT, "Schema lookup: t"):
        assert content in debug


def test_tool_error_message_is_debug_only(conn, caplog):
    caplog.set_level(logging.DEBUG, logger=LOGGER_NAME)
    client = FakeClient(
        response("tool_use", tool_use("SELECT 'secret_value'::INTEGER")),
        response("end_turn", text("ok")),
    )
    run_agent(conn, "analyze", tables="t", client=client)
    assert "Error [ConversionException]" in logged_text(caplog, logging.INFO)
    assert "secret_value" not in logged_text(caplog, logging.INFO)
    assert "secret_value" in logged_text(caplog, logging.DEBUG)


def test_completed_run_has_no_warnings(conn, caplog):
    caplog.set_level(logging.DEBUG, logger=LOGGER_NAME)
    typical_run(conn)
    assert not [r for r in caplog.records if r.levelno >= logging.WARNING]


def test_api_error_is_a_warning(conn, caplog):
    caplog.set_level(logging.INFO, logger=LOGGER_NAME)
    run_agent(conn, "analyze", tables="t", client=RaisingClient(status_error(anthropic.InternalServerError, 500)))
    warnings = logged_text(caplog, logging.WARNING)
    assert "status=api_error" in warnings
    assert "InternalServerError" in warnings


def test_max_iterations_is_a_warning(conn, caplog):
    caplog.set_level(logging.INFO, logger=LOGGER_NAME)
    client = FakeClient(response("tool_use", tool_use("SELECT 1")), response("end_turn", text("summary")))
    run_agent(conn, "analyze", tables="t", client=client, config=AgentConfig(max_iterations=1))
    warnings = logged_text(caplog, logging.WARNING)
    assert "Max iterations (1) reached" in warnings
    assert "status=max_iterations" in warnings


def test_failed_summary_write_is_logged_and_original_error_raised(conn, caplog, tmp_path):
    caplog.set_level(logging.INFO, logger=LOGGER_NAME)
    traces_conn = init_traces_db(str(tmp_path / "traces.duckdb"))
    traces_conn.close()  # every trace write now fails
    with pytest.raises(duckdb.ConnectionException):
        run_agent(conn, "analyze", tables="t", client=FakeClient(), traces_conn=traces_conn)
    errors = [r for r in caplog.records if r.levelno == logging.ERROR]
    assert len(errors) == 1
    assert "Failed to write run summary" in errors[0].getMessage()
    assert errors[0].exc_info is not None


# --- configure_logging ---

def test_configure_logging_writes_plain_messages(conn):
    stream = io.StringIO()
    configure_logging(stream=stream)
    typical_run(conn)
    output = stream.getvalue()
    assert "[Iteration 1] Calling Claude..." in output
    assert "INFO" not in output  # no level/name prefixes: same look as the old prints
    assert ANSWER_TEXT not in output


def test_configure_logging_verbose_shows_everything(conn):
    stream = io.StringIO()
    configure_logging(verbose=True, stream=stream)
    typical_run(conn)
    output = stream.getvalue()
    assert "secret_value" in output and THINKING_TEXT in output and ANSWER_TEXT in output


def test_configure_logging_is_idempotent():
    configure_logging()
    configure_logging(verbose=True)
    console = [h for h in logger.handlers if getattr(h, "_agent_console", False)]
    assert len(console) == 1
    assert logger.level == logging.DEBUG
    assert logger.propagate is False


def test_library_adds_no_output_by_default():
    # Importing the agent must not configure logging for the host application.
    assert all(isinstance(h, logging.NullHandler) for h in logger.handlers)
