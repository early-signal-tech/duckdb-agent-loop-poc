import duckdb
import pytest

from duckdb_agent.sql import check_read_only_sql, execute_query


@pytest.fixture
def conn():
    conn = duckdb.connect()
    conn.execute("CREATE TABLE t AS SELECT 1 AS a")
    return conn


ALLOWED = [
    "SELECT 1",
    "select * from t;",
    "(SELECT 1)",
    "WITH x AS (SELECT 1) SELECT * FROM x",
    "FROM t",
    "DESCRIBE t",
    "SHOW TABLES",
    "SUMMARIZE t",
    "EXPLAIN SELECT * FROM t",
    "EXPLAIN ANALYZE SELECT * FROM t",
    "SELECT * FROM pragma_table_info('t')",
]

BLOCKED = [
    "DELETE FROM t",
    "DROP TABLE t",
    "INSERT INTO t VALUES (2)",
    "UPDATE t SET a = 2",
    "WITH x AS (SELECT 1) INSERT INTO t SELECT * FROM x",
    "EXPLAIN ANALYZE DELETE FROM t",
    "EXPLAIN DELETE FROM t",
    "SELECT 1; DROP TABLE t",
    "PRAGMA threads=1",
    "SET threads=1",
    "CALL pragma_version()",
    "COPY t TO 'out.csv'",
    "ATTACH ':memory:' AS other",
    "CREATE TABLE u AS SELECT 1",
    "",
]


@pytest.mark.parametrize("sql", ALLOWED)
def test_allows_read_only(conn, sql):
    assert check_read_only_sql(conn, sql) is None


@pytest.mark.parametrize("sql", BLOCKED)
def test_blocks_writes_and_side_effects(conn, sql):
    assert check_read_only_sql(conn, sql) is not None


def test_parse_error_is_reported(conn):
    assert check_read_only_sql(conn, "SELEC 1").startswith("SQL parse error")


@pytest.mark.parametrize("sql", BLOCKED)
def test_blocked_sql_does_not_change_data(conn, sql):
    result = execute_query(conn, sql)
    assert result["success"] is False
    assert conn.execute("SELECT count(*) FROM t").fetchone() == (1,)


def test_read_only_attach_blocks_writes_even_if_check_is_bypassed(tmp_path):
    db_path = str(tmp_path / "prod.duckdb")
    setup = duckdb.connect(db_path)
    setup.execute("CREATE TABLE t AS SELECT 1 AS a")
    setup.close()

    conn = duckdb.connect()
    conn.execute(f"ATTACH '{db_path}' AS prod (READ_ONLY)")
    with pytest.raises(duckdb.Error, match="read-only"):
        conn.execute("EXPLAIN ANALYZE DELETE FROM prod.t")
    assert conn.execute("SELECT count(*) FROM prod.t").fetchone() == (1,)
