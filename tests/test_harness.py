import duckdb
import pytest

import agent_harness
from duckdb_agent.log import logger
from test_stop_reasons import FakeClient, response, text, tool_use


@pytest.fixture(autouse=True)
def restore_logger():
    handlers, level, propagate = list(logger.handlers), logger.level, logger.propagate
    yield
    logger.handlers[:] = handlers
    logger.setLevel(level)
    logger.propagate = propagate


@pytest.fixture
def db(tmp_path):
    path = tmp_path / "shop.duckdb"
    conn = duckdb.connect(str(path))
    conn.execute("CREATE SCHEMA sales")
    conn.execute("CREATE TABLE sales.orders AS SELECT range AS id FROM range(3)")
    conn.close()
    return str(path)


def test_defaults_target_ducklake_prod():
    args = agent_harness.parse_args([])
    assert args.db == "ducklake:ducklake_prod"
    assert args.schema == "gold"
    assert args.tables == ["events_rich_ml_gold"]
    assert args.effort == "medium" and not args.quiet and not args.structured


def test_custom_db_has_no_default_table_or_schema(db):
    args = agent_harness.parse_args(["--db", db])
    assert args.schema is None and args.tables is None


def test_repeatable_tables_and_settings():
    args = agent_harness.parse_args(["--table", "a", "--table", "b", "--effort", "high", "--max-iterations", "5"])
    assert args.tables == ["a", "b"]
    assert (args.effort, args.max_iterations) == ("high", 5)


def test_no_table_skips_the_default_preload():
    args = agent_harness.parse_args(["--no-table"])
    assert args.tables is None
    assert args.schema == "gold"


def test_no_table_conflicts_with_table():
    with pytest.raises(SystemExit):
        agent_harness.parse_args(["--no-table", "--table", "a"])


def test_invalid_effort_is_rejected():
    with pytest.raises(SystemExit):
        agent_harness.parse_args(["--effort", "extreme"])


def test_connection_is_read_only(db):
    conn = agent_harness.connect(db, "sales")
    assert conn.execute("SELECT count(*) FROM orders").fetchone() == (3,)
    with pytest.raises(duckdb.Error, match="read-only"):
        conn.execute("DELETE FROM orders")


def test_main_runs_end_to_end(db, tmp_path, capsys):
    client = FakeClient(response("tool_use", tool_use("SELECT count(*) FROM orders")), response("end_turn", text("3 orders.")))
    traces = str(tmp_path / "traces.duckdb")
    status = agent_harness.main(
        ["--db", db, "--schema", "sales", "--table", "orders", "--traces-db", traces, "--quiet"], client=client
    )
    assert status == 0
    assert capsys.readouterr().out.strip() == "3 orders."  # --quiet prints the answer to stdout
    runs = duckdb.connect(traces).execute("SELECT status, table_name FROM agent_runs").fetchall()
    assert runs == [("completed", "orders")]


def test_main_exit_status_is_1_when_not_completed(db, tmp_path):
    client = FakeClient(response("max_tokens", text("cut off")))
    status = agent_harness.main(["--db", db, "--traces-db", str(tmp_path / "t.duckdb"), "--quiet"], client=client)
    assert status == 1
